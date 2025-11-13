#ifndef CUDA_NO_BFLOAT16
#define CUDA_NO_BFLOAT16 1
#endif

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cutlass/gemm/device/gemm.h"

// TODO: Move to utils
#define CHECK_CUDA_ERROR(callable)                                        \
  {                                                                       \
    auto codeError = callable;                                            \
    if (codeError != cudaSuccess) {                                       \
      std::cerr << "\033[1;31merror\033[0m: ";                            \
      std::cerr << cudaGetErrorString(codeError) << '\n';                 \
      std::cerr << "code error: " << static_cast<int>(codeError) << '\n'; \
      std::cerr << "loc: " << __FILE__ << '(' << __LINE__ << ")\n";       \
      std::exit(codeError);                                               \
    }                                                                     \
  }

namespace {
std::vector<__half> make_input_matrix(std::size_t n) {
  static std::mt19937 gen_eng{42};

  std::normal_distribution<float> dist{};
  auto gen = [&]() { return static_cast<__half>(dist(gen_eng)); };

  std::vector<__half> matrix(n * n * 2);

  std::generate(matrix.begin(), matrix.end(), gen);

  return matrix;
}

// TODO: use intrinsics
std::vector<float> run_openmp_reference(const std::vector<__half>& matrix,
                                        std::size_t n) {
  const size_t size = n * n;

  std::vector<float> result(size, 0.0f);

  const __half* a = matrix.data();
  const __half* b = matrix.data() + size;

#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    float* row = &result[i * n];
    for (size_t k = 0; k < n; ++k) {
      for (size_t j = 0; j < n; ++j) {
        row[j] += __half2float(a[i * n + k]) * __half2float(b[k * n + j]);
      }
    }

    float rowsum = 0.0f;
    for (std::size_t j = 0; j < n; j++) {
      const float val = std::exp(row[j]);
      rowsum += val;
      // row[j] = val;
    }
    const float inv_rowsum = 1.0f / rowsum;
    for (std::size_t j = 0; j < n; j++) {
      row[j] = std::exp(row[j]) * inv_rowsum;
      // row[j] *= row_sum;
    }
  }

  return result;
}

__global__ void softmax_kernel(const float* input, float* output, size_t n) {
  unsigned int data_row = blockIdx.x * n;
  unsigned int tid = threadIdx.x;
  unsigned int block_size = blockDim.x;

  extern __shared__ float sdata[];
  const float* row = &input[data_row];
  float* out = &output[data_row];

  float tid_sum = 0.0f;
  for (unsigned int j = tid; j < n; j += block_size) {
    tid_sum += __expf(row[j]);
  }
  sdata[tid] = tid_sum;
  __syncthreads();

  for (unsigned int s = block_size >> 1; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  __syncthreads();
  float inv_rowsum = 1.0f / sdata[0];

  for (unsigned int j = tid; j < n; j += block_size) {
    out[j] = __expf(row[j]) * inv_rowsum;
  }
}

#include <mma.h>

using namespace nvcuda;

constexpr std::size_t WMMA_SIZE = 16;
constexpr std::size_t WARP_SIZE = 32;

__global__ void GEMMv4(const __half* a, const __half* b, float* c,
                       std::size_t n) {
  wmma::fragment<wmma::matrix_a, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half,
                 wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, half,
                 wmma::row_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_SIZE, WMMA_SIZE, WMMA_SIZE, float>
      acc_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  int row = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_SIZE;
  int col = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE * WMMA_SIZE;

  if (row >= n || col >= n) return;

  for (int k = 0; k < n; k += WMMA_SIZE) {
    wmma::load_matrix_sync(a_frag, &a[row * n + k], n);
    wmma::load_matrix_sync(b_frag, &b[k * n + col], n);

    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  wmma::store_matrix_sync(&c[row * n + col], acc_frag, n, wmma::mem_row_major);
}

std::vector<float> run_wmma(const std::vector<__half>& matrix, std::size_t n) {
  if (n <= 16) return run_openmp_reference(matrix, n);

  const size_t size = n * n;
  std::vector<float> res(size);

  half *a_dev = nullptr, *b_dev = nullptr;
  float* c_dev = nullptr;
  float* temp_dev = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(&a_dev, size * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMalloc(&b_dev, size * sizeof(half)));
  CHECK_CUDA_ERROR(cudaMalloc(&c_dev, size * sizeof(float)));
  CHECK_CUDA_ERROR(cudaMalloc(&temp_dev, size * sizeof(float)));

  CHECK_CUDA_ERROR(cudaMemcpy(a_dev, matrix.data(), size * sizeof(half),
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(b_dev, matrix.data() + size, size * sizeof(half),
                              cudaMemcpyHostToDevice));

  constexpr int Y_THREADS = 5;
  const int warps_per_block = Y_THREADS;
  dim3 block_size(Y_THREADS * WARP_SIZE, Y_THREADS);
  const int tiles_per_dim = static_cast<int>((n + WMMA_SIZE - 1) / WMMA_SIZE);
  dim3 block_count((tiles_per_dim + warps_per_block - 1) / warps_per_block,
                   (tiles_per_dim + block_size.y - 1) / block_size.y);

  GEMMv4<<<block_count, block_size>>>(a_dev, b_dev, temp_dev, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());

  const int softmax_threads = 256;
  dim3 softmax_block(softmax_threads);
  dim3 softmax_grid(n);
  size_t softmax_shared = softmax_threads * sizeof(float);
  softmax_kernel<<<softmax_grid, softmax_block, softmax_shared>>>(temp_dev,
                                                                  c_dev, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  std::cout << "block_size: " << block_size.x << " " << block_size.y << " "
            << "block_count: " << block_count.x << " " << block_count.y << " "
            << "softmax_block: " << softmax_block.x << " "
            << "softmax_grid: " << softmax_grid.x << '\n';

  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  CHECK_CUDA_ERROR(cudaEventRecord(start));
  GEMMv4<<<block_count, block_size>>>(a_dev, b_dev, temp_dev, n);
  softmax_kernel<<<softmax_grid, softmax_block, softmax_shared>>>(temp_dev,
                                                                  c_dev, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
  float milliseconds = 0.0f;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "Time (wmma): " << (milliseconds / 1000.0f) << " sec"
            << std::endl;
  CHECK_CUDA_ERROR(cudaEventDestroy(start));
  CHECK_CUDA_ERROR(cudaEventDestroy(stop));
  CHECK_CUDA_ERROR(cudaMemcpy(res.data(), c_dev, size * sizeof(float),
                              cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(a_dev));
  CHECK_CUDA_ERROR(cudaFree(b_dev));
  CHECK_CUDA_ERROR(cudaFree(c_dev));
  CHECK_CUDA_ERROR(cudaFree(temp_dev));

  return res;
}

void warmup_wmma(const std::vector<__half>& matrix, std::size_t n) {
  run_wmma(matrix, n);
}

cudaError_t CutlassGEMM(const cutlass::half_t* a, const cutlass::half_t* b,
                        float* c, int n) {
  using RowMajor = cutlass::layout::RowMajor;
  using OpClassTensorOp = cutlass::arch::OpClassTensorOp;
  using Sm75 = cutlass::arch::Sm75;
  using CutlassGemm =
      cutlass::gemm::device::Gemm<cutlass::half_t,  // Data-type of A matrix
                                  RowMajor,         // Layout of A matrix
                                  cutlass::half_t,  // Data-type of B matrix
                                  RowMajor,         // Layout of B matrix
                                  float,            // Data-type of C matrix
                                  RowMajor,         // Layout of C matrix
                                  float,            // Element Accumulator
                                  OpClassTensorOp,  // Tag indicating Tensor
                                                    // Cores
                                  Sm75>;            // SM architecture
  CutlassGemm::Arguments args({n, n, n},            // Gemm Problem dimensions
                              {a, n},   // Tensor-ref for source matrix A
                              {b, n},   // Tensor-ref for source matrix B
                              {c, n},   // Tensor-ref for source matrix C
                              {c, n},   // Tensor-ref for destination matrix D
                              {1, 0});  // Scalars used in the Epilogue
  CutlassGemm gemm_operator;
  cutlass::Status status = gemm_operator(args);
  return status == cutlass::Status::kSuccess ? cudaSuccess : cudaErrorUnknown;
}

std::vector<float> run_cutlass(const std::vector<__half>& matrix,
                               std::size_t n) {
  if (n <= 16) return run_openmp_reference(matrix, n);

  const size_t size = n * n;
  std::vector<float> res(size);

  cutlass::half_t *a_dev = nullptr, *b_dev = nullptr;
  float* c_dev = nullptr;
  float* temp_dev = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(&a_dev, size * sizeof(*a_dev)));
  CHECK_CUDA_ERROR(cudaMalloc(&b_dev, size * sizeof(*b_dev)));
  CHECK_CUDA_ERROR(cudaMalloc(&c_dev, size * sizeof(*c_dev)));
  CHECK_CUDA_ERROR(cudaMalloc(&temp_dev, size * sizeof(*temp_dev)));

  CHECK_CUDA_ERROR(cudaMemcpy(a_dev, matrix.data(),
                              size * sizeof(*matrix.data()),
                              cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(cudaMemcpy(b_dev, matrix.data() + size,
                              size * sizeof(*matrix.data()),
                              cudaMemcpyHostToDevice));

  cudaError_t status = CutlassGEMM(a_dev, b_dev, temp_dev, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  const int softmax_threads = 256;
  dim3 softmax_block(softmax_threads);
  dim3 softmax_grid(n);
  size_t softmax_shared = softmax_threads * sizeof(*temp_dev);
  softmax_kernel<<<softmax_grid, softmax_block, softmax_shared>>>(temp_dev,
                                                                  c_dev, n);

  cudaEvent_t start, stop;
  CHECK_CUDA_ERROR(cudaEventCreate(&start));
  CHECK_CUDA_ERROR(cudaEventCreate(&stop));
  CHECK_CUDA_ERROR(cudaEventRecord(start));

  auto start_host = std::chrono::system_clock::now();

  status = CutlassGEMM(a_dev, b_dev, c_dev, n);
  softmax_kernel<<<softmax_grid, softmax_block, softmax_shared>>>(temp_dev,
                                                                  c_dev, n);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaEventRecord(stop));
  CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

  auto stop_host = std::chrono::system_clock::now();
  std::chrono::duration<float> elapsed = stop_host - start_host;
  std::cout << "Time (cutlass_host): " << elapsed.count() << " sec"
            << std::endl;

  float milliseconds = 0.0f;
  CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));
  std::cout << "Time (cutlass_device): " << (milliseconds / 1000.0f) << " sec"
            << std::endl;

  CHECK_CUDA_ERROR(cudaMemcpy(res.data(), c_dev, size * sizeof(*c_dev),
                              cudaMemcpyDeviceToHost));

  CHECK_CUDA_ERROR(cudaFree(a_dev));
  CHECK_CUDA_ERROR(cudaFree(b_dev));
  CHECK_CUDA_ERROR(cudaFree(c_dev));

  return res;
}

void warmup_cutlass(const std::vector<__half>& matrix, std::size_t n) {
  run_cutlass(matrix, n);
}

double measure_seconds(const std::function<std::vector<float>()>& work,
                       std::vector<float>& result_store) {
  const auto start = std::chrono::high_resolution_clock::now();
  result_store = work();
  const auto stop = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(stop - start).count();
}

float max_abs_diff(const std::vector<float>& baseline,
                   const std::vector<float>& candidate) {
  if (baseline.size() != candidate.size()) {
    throw std::runtime_error(
        "Result size mismatch while validating correctness");
  }
  float max_diff = 0.0f;
  int c = 0;
  for (std::size_t i = 0; i < baseline.size(); ++i) {
    const auto diff = std::abs(baseline[i] - candidate[i]);
    max_diff = std::max(max_diff, diff);
    if (diff > 1e-3f) {
      c++;
    }
  }
  std::cout << "Num mismatches: " << c << " / " << baseline.size() << " ("
            << static_cast<float>(c) / baseline.size() << ")\n";
  return max_diff;
}

// TODO: Create basic utils file
struct RunResult {
  double seconds = 0.0;
  float diff = 0.0f;
  bool success = false;
  explicit operator bool() const noexcept { return success; }
};

std::string format_time(double seconds) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << seconds;
  return oss.str();
}

std::string format_diff(float diff) {
  std::ostringstream oss;
  oss << std::defaultfloat << std::setprecision(1) << diff;
  return oss.str();
}

void print_report(std::string_view testName, const RunResult& result) {
  if (result) {
    std::cout << testName << ": " << format_time(result.seconds)
              << " sec (diff: " << format_diff(result.diff) << ")\n";
  } else {
    std::cout << testName << ": n/a (diff: n/a)\n";
  }
}
}  // namespace

int main(int argc, char* argv[]) {
  try {
    if (argc != 2) {
      std::cerr << "Usage: " << argv[0] << " <matrix_size_n>\n";
      return EXIT_FAILURE;
    }

    const std::size_t n = static_cast<std::size_t>(std::stoul(argv[1]));
    if (n == 0) {
      throw std::invalid_argument("Matrix size must be positive");
    }

    const auto input = make_input_matrix(n);
    std::vector<float> openmp_result;
    const double openmp_seconds = measure_seconds(
        [&]() { return run_openmp_reference(input, n); }, openmp_result);

    RunResult wmma_res;
    try {
      std::vector<float> result;
      warmup_wmma(input, n);
      wmma_res.seconds =
          measure_seconds([&]() { return run_wmma(input, n); }, result);
      wmma_res.diff = max_abs_diff(openmp_result, result);
      wmma_res.success = true;
    } catch (const std::exception& ex) {
      std::cerr << "WMMA method failed: " << ex.what() << '\n';
    }

    RunResult cutlass_res;
    try {
      std::vector<float> result;
      warmup_cutlass(input, n);
      cutlass_res.seconds =
          measure_seconds([&]() { return run_cutlass(input, n); }, result);
      cutlass_res.diff = max_abs_diff(openmp_result, result);
      cutlass_res.success = true;
    } catch (const std::exception& ex) {
      std::cerr << "CUTLASS method failed: " << ex.what() << '\n';
    }

    std::cout << "OpenMP: " << format_time(openmp_seconds) << " sec\n";
    print_report("WMMA", wmma_res);
    print_report("CUTLASS", cutlass_res);

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}
