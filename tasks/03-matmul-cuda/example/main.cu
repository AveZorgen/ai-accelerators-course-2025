#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

  std::normal_distribution<__half> dist{};
  auto gen = [&]() { return dist(gen_eng); };

  std::vector<__half> matrix(n * n);

  std::generate(matrix.begin(), matrix.end(), gen);

  return matrix;
}

std::vector<float> run_openmp_reference(const std::vector<__half>& matrix,
                                        std::size_t n) {
  std::vector<float> result(n * n, 0.0f);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < n; ++k) {
      for (int j = 0; j < n; ++j) {
        result[i * n + j] +=
            static_cast<float>(matrix[i * n + k] * matrix[k * n + j]);
      }
    }
    float rowsum = 0.0f;
    for (int j = 0; j < n; ++j) {
      result[i * n + j] = std::exp(result[i * n + j]);
      rowsum += result[i * n + j];
    }
    const float inv_rowsum = 1.0f / rowsum;
    for (int j = 0; j < n; ++j) {
      result[i * n + j] *= inv_rowsum;
    }
  }

  return result;
}

void warmup_wmma(const std::vector<__half>& matrix, std::size_t n) {
  throw std::runtime_error("WMMA warm-up not implemented");
}

std::vector<float> run_wmma(const std::vector<__half>& matrix, std::size_t n) {
  throw std::runtime_error("WMMA method not implemented");
}

void warmup_cutlass(const std::vector<__half>& matrix, std::size_t n) {
  throw std::runtime_error("CUTLASS warm-up not implemented");
}

std::vector<float> run_cutlass(const std::vector<__half>& matrix,
                               std::size_t n) {
  throw std::runtime_error("CUTLASS method not implemented");
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
  for (std::size_t i = 0; i < baseline.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(baseline[i] - candidate[i]));
  }
  return max_diff;
}

// TODO: Create basic utils file
struct RunResult {
  std::vector<float> result;
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
      warmup_wmma(input, n);
      wmma_res.seconds = measure_seconds([&]() { return run_wmma(input, n); },
                                         wmma_res.result);
      wmma_res.diff = max_abs_diff(openmp_result, wmma_res.result);
      wmma_res.success = true;
    } catch (const std::exception& ex) {
      std::cerr << "WMMA method failed: " << ex.what() << '\n';
    }

    RunResult cutlass_res;
    try {
      warmup_cutlass(input, n);
      cutlass_res.seconds = measure_seconds(
          [&]() { return run_cutlass(input, n); }, cutlass_res.result);
      cutlass_res.diff = max_abs_diff(openmp_result, cutlass_res.result);
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
