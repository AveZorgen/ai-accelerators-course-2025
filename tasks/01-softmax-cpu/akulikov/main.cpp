#include <immintrin.h>

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
#include <string_view>
#include <vector>

namespace {
std::vector<float> make_matrix(std::size_t n) {
  static std::mt19937 gen_eng{42};

  std::normal_distribution<float> dist{};
  auto gen = [&]() { return dist(gen_eng); };

  std::vector<float> matrix(n * n);

  std::generate(matrix.begin(), matrix.end(), gen);

  return matrix;
}

static inline void process_row(const float* row, std::size_t n, float* out) {
  float rowsum = 0.0f;
  for (std::size_t j = 0; j < n; j++) {
    rowsum += std::exp(row[j]);
  }
  const float inv_rowsum = 1.0f / rowsum;
  for (std::size_t j = 0; j < n; j++) {
    out[j] = std::exp(row[j]) * inv_rowsum;
  }
}

__m256 exp256_ps(__m256 x);

static inline void process_row_simd(const float* row, std::size_t n,
                                    float* out) {
  auto y = _mm256_setzero_ps();
  std::size_t j = 0;
  for (; j + 8 <= n; j += 8) {
    auto x = _mm256_loadu_ps(&row[j]);
    x = exp256_ps(x);
    y = _mm256_add_ps(y, x);
  }
  float tail_sum = 0.0f;
  for (; j < n; j++) {
    tail_sum += std::exp(row[j]);
  }

  float rowsum[8];
  _mm256_storeu_ps(rowsum, y);
  const float inv_rowsum =
      1.0f / (rowsum[0] + rowsum[1] + rowsum[2] + rowsum[3] + rowsum[4] +
              rowsum[5] + rowsum[6] + rowsum[7] + tail_sum);

  const auto inv_rowsum_vec = _mm256_set1_ps(inv_rowsum);
  for (j = 0; j + 8 <= n; j += 8) {
    auto x = _mm256_loadu_ps(&row[j]);
    x = exp256_ps(x);
    x = _mm256_mul_ps(x, inv_rowsum_vec);
    _mm256_storeu_ps(&out[j], x);
  }
  for (; j < n; j++) {
    out[j] = std::exp(row[j]) * inv_rowsum;
  }
}

std::vector<float> run_sequential(const std::vector<float>& matrix,
                                  std::size_t n) {
  std::vector<float> result(n * n, 0.0f);

#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < n; ++k) {
        sum += static_cast<float>(matrix[i * n + k] * matrix[k * n + j]);
      }
      result[i * n + j] = sum;
    }
    process_row(&result[i * n], n, &result[i * n]);
  }

  return result;
}

std::vector<float> run_openmp(const std::vector<float>& matrix, std::size_t n) {
  std::vector<float> result(n * n, 0.0f);

#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    float rowsum = 0.0f;
    for (size_t j = 0; j < n; ++j) {
      float sum = 0.0f;
      for (size_t k = 0; k < n; ++k) {
        sum += static_cast<float>(matrix[i * n + k] * matrix[k * n + j]);
      }
      const float exp_val = std::exp(sum);
      rowsum += exp_val;
      result[i * n + j] = exp_val;
    }
    const float inv_rowsum = 1.0f / rowsum;
    for (size_t j = 0; j < n; ++j) {
      result[i * n + j] *= inv_rowsum;
    }
  }

  return result;
}

std::vector<float> run_simd(const std::vector<float>& matrix, std::size_t n) {
  std::vector<float> result(n * n, 0.0f);

#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    float rowsum = 0.0f;
    for (size_t k = 0; k < n; ++k) {
      for (size_t j = 0; j < n; ++j) {
        float el = static_cast<float>(matrix[i * n + k] * matrix[k * n + j]);
        result[i * n + j] += el;
      }
    }
    for (size_t j = 0; j < n; ++j) {
      result[i * n + j] = std::exp(result[i * n + j]);
      rowsum += result[i * n + j];
    }
    const float inv_rowsum = 1.0f / rowsum;
    for (size_t j = 0; j < n; ++j) {
      result[i * n + j] *= inv_rowsum;
    }
  }

  return result;
}

std::vector<float> run_openmp_simd(const std::vector<float>& matrix,
                                   std::size_t n) {
  std::vector<float> result(n * n, 0.0f);

#pragma omp parallel for
  for (size_t i = 0; i < n; ++i) {
    for (size_t k = 0; k < n; ++k) {
      for (size_t j = 0; j < n; ++j) {
        result[i * n + j] +=
            static_cast<float>(matrix[i * n + k] * matrix[k * n + j]);
      }
    }
    process_row_simd(&result[i * n], n, &result[i * n]);
  }

  return result;
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

void print_report(const std::string_view& testName, const RunResult& result) {
  if (result) {
    std::cout << testName << ": " << format_time(result.seconds)
              << " sec (diff: " << format_diff(result.diff) << ")\n";
  } else {
    std::cout << testName << ": n/a (diff: n/a)\n";
  }
}

RunResult run_test_case(const std::function<std::vector<float>()>& runner,
                        const std::vector<float>& baseline,
                        const std::string_view& methodName) {
  RunResult result;
  try {
    result.seconds = measure_seconds(runner, result.result);
    result.diff = max_abs_diff(baseline, result.result);
    result.success = true;
  } catch (const std::exception& ex) {
    std::cerr << methodName << " method failed: " << ex.what() << '\n';
  }
  return result;
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

    const auto input = make_matrix(n);

    std::vector<float> sequential_result;
    const double sequential_seconds = measure_seconds(
        [&]() { return run_sequential(input, n); }, sequential_result);

    auto omp_res = run_test_case([&] { return run_openmp(input, n); },
                                 sequential_result, "OpenMP");
    auto simd_res = run_test_case([&] { return run_simd(input, n); },
                                  sequential_result, "SIMD");
    auto omp_simd_res = run_test_case([&] { return run_openmp_simd(input, n); },
                                      sequential_result, "OpenMP + SIMD");

    std::cout << "Sequential: " << format_time(sequential_seconds) << " sec\n";
    print_report("OpenMP", omp_res);
    print_report("SIMD", simd_res);
    print_report("OpenMP + SIMD", omp_simd_res);

    return EXIT_SUCCESS;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << '\n';
  } catch (...) {
    std::cerr << "Unknown error\n";
  }

  return EXIT_FAILURE;
}

namespace {
__m256 exp256_ps(__m256 x) {
  /* Modified code. The original code is here:
    https://github.com/reyoung/avx_mathfun

     AVX implementation of exp
     Based on "sse_mathfun.h", by Julien Pommier
     http://gruntthepeon.free.fr/ssemath/
     Copyright (C) 2012 Giovanni Garberoglio
     Interdisciplinary Laboratory for Computational Science (LISC)
     Fondazione Bruno Kessler and University of Trento
     via Sommarive, 18
     I-38123 Trento (Italy)
    This software is provided 'as-is', without any express or implied
    warranty.  In no event will the authors be held liable for any damages
    arising from the use of this software.
    Permission is granted to anyone to use this software for any purpose,
    including commercial applications, and to alter it and redistribute it
    freely, subject to the following restrictions:
    1. The origin of this software must not be misrepresented; you must not
       claim that you wrote the original software. If you use this software
       in a product, an acknowledgment in the product documentation would be
       appreciated but is not required.
    2. Altered source versions must be plainly marked as such, and must not be
       misrepresented as being the original software.
    3. This notice may not be removed or altered from any source distribution.
    (this is the zlib license)
  */
  /*
    To increase the compatibility across different compilers the original code
    is converted to plain AVX2 intrinsics code without ingenious macro's, gcc
    style alignment attributes etc. The modified code requires AVX2
  */
  __m256 exp_hi = _mm256_set1_ps(88.3762626647949f);
  __m256 exp_lo = _mm256_set1_ps(-88.3762626647949f);

  __m256 cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341);
  __m256 cephes_exp_C1 = _mm256_set1_ps(0.693359375);
  __m256 cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4);

  __m256 cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
  __m256 cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
  __m256 cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
  __m256 cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
  __m256 cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
  __m256 cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
  __m256 tmp = _mm256_setzero_ps(), fx;
  __m256i imm0;
  __m256 one = _mm256_set1_ps(1.0f);

  x = _mm256_min_ps(x, exp_hi);
  x = _mm256_max_ps(x, exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _mm256_mul_ps(x, cephes_LOG2EF);
  fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
  tmp = _mm256_floor_ps(fx);
  __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
  mask = _mm256_and_ps(mask, one);
  fx = _mm256_sub_ps(tmp, mask);
  tmp = _mm256_mul_ps(fx, cephes_exp_C1);
  __m256 z = _mm256_mul_ps(fx, cephes_exp_C2);
  x = _mm256_sub_ps(x, tmp);
  x = _mm256_sub_ps(x, z);
  z = _mm256_mul_ps(x, x);

  __m256 y = cephes_exp_p0;
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p1);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p2);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p3);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p4);
  y = _mm256_mul_ps(y, x);
  y = _mm256_add_ps(y, cephes_exp_p5);
  y = _mm256_mul_ps(y, z);
  y = _mm256_add_ps(y, x);
  y = _mm256_add_ps(y, one);

  /* build 2^n */
  imm0 = _mm256_cvttps_epi32(fx);
  imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
  imm0 = _mm256_slli_epi32(imm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(imm0);
  y = _mm256_mul_ps(y, pow2n);
  return y;
}
}  // namespace
