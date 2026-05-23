#include "zcutlass/gemm.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

#define CHECK_CUDA(expr)                                                         \
  do {                                                                           \
    cudaError_t status = (expr);                                                 \
    if (status != cudaSuccess) {                                                 \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(status) << std::endl;                     \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

#define CHECK_CUBLAS(expr)                                                       \
  do {                                                                           \
    cublasStatus_t status = (expr);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << ": "   \
                << static_cast<int>(status) << std::endl;                       \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

template <typename T>
T from_float(float value);

template <>
half from_float<half>(float value) {
  return __float2half_rn(value);
}

template <>
__nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
float to_float(T value);

template <>
float to_float<half>(half value) {
  return __half2float(value);
}

template <>
float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
zcutlass::DType dtype();

template <>
zcutlass::DType dtype<half>() {
  return zcutlass::DType::F16;
}

template <>
zcutlass::DType dtype<__nv_bfloat16>() {
  return zcutlass::DType::BF16;
}

template <typename T>
cudaDataType cuda_type();

template <>
cudaDataType cuda_type<half>() {
  return CUDA_R_16F;
}

template <>
cudaDataType cuda_type<__nv_bfloat16>() {
  return CUDA_R_16BF;
}

template <typename T>
float tolerance();

template <>
float tolerance<half>() {
  return 0.16f;
}

template <>
float tolerance<__nv_bfloat16>() {
  return 0.55f;
}

template <typename T>
float sentinel_value();

template <>
float sentinel_value<half>() {
  return -7.5f;
}

template <>
float sentinel_value<__nv_bfloat16>() {
  return -6.5f;
}

template <typename T>
std::vector<T> make_tensor(int64_t rows, int64_t cols, int64_t ld, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.35f, 0.35f);
  std::vector<T> values(rows * ld);
  for (int64_t i = 0; i < rows; ++i) {
    for (int64_t j = 0; j < ld; ++j) {
      values[i * ld + j] = from_float<T>(j < cols ? dist(rng) : 0.0f);
    }
  }
  return values;
}

template <typename T>
std::vector<float> reference(const std::vector<T>& A,
                             const std::vector<T>& B,
                             const std::vector<T>& C,
                             const std::vector<T>& bias,
                             int64_t m,
                             int64_t n,
                             int64_t k,
                             int64_t lda,
                             int64_t ldb,
                             int64_t ldc,
                             float alpha,
                             float beta,
                             bool use_bias) {
  std::vector<float> out(m * n, 0.0f);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      float accum = 0.0f;
      for (int64_t kk = 0; kk < k; ++kk) {
        accum += to_float<T>(A[row * lda + kk]) * to_float<T>(B[kk * ldb + col]);
      }
      float value = alpha * accum;
      if (beta != 0.0f) {
        value += beta * to_float<T>(C[row * ldc + col]);
      }
      if (use_bias) {
        value += to_float<T>(bias[col]);
      }
      out[row * n + col] = to_float<T>(from_float<T>(value));
    }
  }
  return out;
}

void expect_status(const std::string& name,
                   zcutlass::Status actual,
                   zcutlass::Status expected) {
  if (actual != expected) {
    std::cerr << "Expected " << name << " to be "
              << zcutlass::status_to_string(expected) << ", got "
              << zcutlass::status_to_string(actual) << std::endl;
    std::exit(1);
  }
}

void expect_kernel_contains(const std::string& name,
                            const zcutlass::GemmDesc& desc,
                            const std::string& expected) {
  const char* kernel_name = zcutlass::selected_kernel_name(desc);
  const std::string selected = kernel_name != nullptr ? kernel_name : "null";
  if (selected.find(expected) == std::string::npos) {
    std::cerr << "Expected " << name << " to select a kernel containing '" << expected
              << "', got " << selected << std::endl;
    std::exit(1);
  }
}

void expect_kernel_not_contains(const std::string& name,
                                const zcutlass::GemmDesc& desc,
                                const std::string& unexpected) {
  const char* kernel_name = zcutlass::selected_kernel_name(desc);
  const std::string selected = kernel_name != nullptr ? kernel_name : "null";
  if (selected.find(unexpected) != std::string::npos) {
    std::cerr << "Expected " << name << " to avoid kernels containing '" << unexpected
              << "', got " << selected << std::endl;
    std::exit(1);
  }
}

void expect_family(const std::string& name,
                   const zcutlass::GemmDesc& desc,
                   const std::string& expected) {
  const std::string family = zcutlass::selected_kernel_family(desc);
  if (family != expected) {
    std::cerr << "Expected " << name << " family '" << expected << "', got "
              << family << std::endl;
    std::exit(1);
  }
}

void expect_path(const std::string& name,
                 const zcutlass::GemmDesc& desc,
                 const std::string& expected) {
  const std::string path = zcutlass::selected_kernel_path(desc);
  if (path != expected) {
    std::cerr << "Expected " << name << " path '" << expected << "', got "
              << path << std::endl;
    std::exit(1);
  }
}

void expect_pipeline(const std::string& name,
                     const zcutlass::GemmDesc& desc,
                     int expected) {
  const int stages = zcutlass::selected_kernel_pipeline_stages(desc);
  if (stages != expected) {
    std::cerr << "Expected " << name << " pipeline stages " << expected << ", got "
              << stages << std::endl;
    std::exit(1);
  }
}

void expect_epilogue(const std::string& name,
                     const zcutlass::GemmDesc& desc,
                     const std::string& expected) {
  const std::string epilogue = zcutlass::selected_kernel_epilogue_kind(desc);
  if (epilogue != expected) {
    std::cerr << "Expected " << name << " epilogue '" << expected << "', got "
              << epilogue << std::endl;
    std::exit(1);
  }
}

void set_experimental_kernel_filter(const char* filter) {
#if defined(_WIN32)
  if (filter == nullptr) {
    _putenv_s("ZCUTLASS_EXPERIMENTAL_KERNELS", "");
    _putenv_s("ZCUTLASS_EXPERIMENTAL_KERNEL", "");
  } else {
    _putenv_s("ZCUTLASS_EXPERIMENTAL_KERNELS", "1");
    _putenv_s("ZCUTLASS_EXPERIMENTAL_KERNEL", filter);
  }
#else
  if (filter == nullptr) {
    unsetenv("ZCUTLASS_EXPERIMENTAL_KERNELS");
    unsetenv("ZCUTLASS_EXPERIMENTAL_KERNEL");
  } else {
    setenv("ZCUTLASS_EXPERIMENTAL_KERNELS", "1", 1);
    setenv("ZCUTLASS_EXPERIMENTAL_KERNEL", filter, 1);
  }
#endif
}

template <typename T>
void run_case(const std::string& name,
              cublasHandle_t handle,
              int64_t m,
              int64_t n,
              int64_t k,
              float alpha,
              float beta,
              bool use_bias,
              int64_t lda_pad,
              int64_t ldb_pad,
              int64_t ldc_pad,
              int64_t ldd_pad) {
  const int64_t lda = k + lda_pad;
  const int64_t ldb = n + ldb_pad;
  const int64_t ldc = n + ldc_pad;
  const int64_t ldd = n + ldd_pad;

  std::vector<T> h_a = make_tensor<T>(m, k, lda, 17);
  std::vector<T> h_b = make_tensor<T>(k, n, ldb, 29);
  std::vector<T> h_c = make_tensor<T>(m, n, ldc, 41);
  std::vector<T> h_d(m * ldd, from_float<T>(sentinel_value<T>()));
  std::vector<T> h_bias = make_tensor<T>(1, n, n, 53);

  T* d_a = nullptr;
  T* d_b = nullptr;
  T* d_c = nullptr;
  T* d_d = nullptr;
  T* d_blas = nullptr;
  T* d_bias = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, h_a.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_b, h_b.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_c, h_c.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_d, h_d.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_blas, h_d.size() * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_bias, h_bias.size() * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), h_b.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), h_c.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_d, h_d.data(), h_d.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_blas, h_d.data(), h_d.size() * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_bias, h_bias.data(), h_bias.size() * sizeof(T), cudaMemcpyHostToDevice));

  zcutlass::GemmDesc desc{
      m, n, k, lda, ldb, ldc, ldd, dtype<T>(), dtype<T>(), dtype<T>(), dtype<T>(),
      alpha, beta, use_bias ? static_cast<const void*>(d_bias) : nullptr, nullptr};

  const char* kernel_name = zcutlass::selected_kernel_name(desc);
  if (kernel_name == nullptr || std::string(kernel_name) == "none") {
    std::cerr << "Expected a selected kernel for " << name << std::endl;
    std::exit(1);
  }

  const zcutlass::Status status = zcutlass::gemm(desc, d_a, d_b, beta != 0.0f ? d_c : nullptr, d_d);
  if (status != zcutlass::Status::Success) {
    std::cerr << "zcutlass::gemm failed in " << name << ": "
              << zcutlass::status_to_string(status) << std::endl;
    std::exit(1);
  }
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(h_d.data(), d_d, h_d.size() * sizeof(T), cudaMemcpyDeviceToHost));

  std::vector<T> h_blas;
  if (!use_bias && ldc == ldd) {
    if (beta != 0.0f) {
      CHECK_CUDA(cudaMemcpy(d_blas, h_c.data(), h_c.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              static_cast<int>(n),
                              static_cast<int>(m),
                              static_cast<int>(k),
                              &alpha,
                              d_b,
                              cuda_type<T>(),
                              static_cast<int>(ldb),
                              d_a,
                              cuda_type<T>(),
                              static_cast<int>(lda),
                              &beta,
                              d_blas,
                              cuda_type<T>(),
                              static_cast<int>(ldd),
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CHECK_CUDA(cudaDeviceSynchronize());
    h_blas.resize(h_d.size());
    CHECK_CUDA(cudaMemcpy(h_blas.data(), d_blas, h_blas.size() * sizeof(T), cudaMemcpyDeviceToHost));
  }

  std::vector<float> expected =
      reference<T>(h_a, h_b, h_c, h_bias, m, n, k, lda, ldb, ldc, alpha, beta, use_bias);

  float max_abs = 0.0f;
  int64_t bad = -1;
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < n; ++col) {
      const float actual = to_float<T>(h_d[row * ldd + col]);
      const float diff = std::abs(actual - expected[row * n + col]);
      if (diff > max_abs) {
        max_abs = diff;
        bad = row * n + col;
      }
      if (diff > tolerance<T>()) {
        std::cerr << "Mismatch in " << name << " at (" << row << ", " << col
                  << "): actual=" << actual << " expected=" << expected[row * n + col]
                  << " diff=" << diff << " tol=" << tolerance<T>() << std::endl;
        std::exit(1);
      }
      if (!h_blas.empty()) {
        const float blas = to_float<T>(h_blas[row * ldd + col]);
        const float blas_diff = std::abs(actual - blas);
        if (blas_diff > tolerance<T>()) {
          std::cerr << "cuBLAS mismatch in " << name << " at (" << row << ", " << col
                    << "): actual=" << actual << " cublas=" << blas
                    << " diff=" << blas_diff << " tol=" << tolerance<T>() << std::endl;
          std::exit(1);
        }
      }
    }
  }

  std::cout << "[pass] " << name << " max_abs=" << max_abs;
  if (bad >= 0) {
    std::cout << " worst_index=" << bad;
  }
  std::cout << std::endl;

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = n; col < ldd; ++col) {
      const float actual = to_float<T>(h_d[row * ldd + col]);
      const float expected_sentinel = to_float<T>(from_float<T>(sentinel_value<T>()));
      if (std::abs(actual - expected_sentinel) > 0.0f) {
        std::cerr << "Output padding overwrite in " << name << " at (" << row << ", "
                  << col << "): actual=" << actual
                  << " expected=" << expected_sentinel << std::endl;
        std::exit(1);
      }
    }
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_d));
  CHECK_CUDA(cudaFree(d_blas));
  CHECK_CUDA(cudaFree(d_bias));
}


template <typename T>
void run_case(const std::string& name,
              cublasHandle_t handle,
              int64_t m,
              int64_t n,
              int64_t k,
              float alpha,
              float beta,
              bool use_bias,
              bool padded) {
  run_case<T>(name,
              handle,
              m,
              n,
              k,
              alpha,
              beta,
              use_bias,
              padded ? 3 : 0,
              padded ? 5 : 0,
              padded ? 7 : 0,
              padded ? 11 : 0);
}

template <typename T>
void run_zero_size_tests() {
  zcutlass::GemmDesc desc{0,
                          16,
                          16,
                          16,
                          16,
                          16,
                          16,
                          dtype<T>(),
                          dtype<T>(),
                          dtype<T>(),
                          dtype<T>(),
                          1.0f,
                          0.0f,
                          nullptr,
                          nullptr};
  expect_status("zero m with null tensors",
                zcutlass::gemm(desc, nullptr, nullptr, nullptr, nullptr),
                zcutlass::Status::Success);

  desc.m = 16;
  desc.n = 0;
  expect_status("zero n with null tensors",
                zcutlass::gemm(desc, nullptr, nullptr, nullptr, nullptr),
                zcutlass::Status::Success);

  desc.n = 16;
  desc.k = 0;
  desc.beta = 1.0f;
  expect_status("zero k with null tensors and beta",
                zcutlass::gemm(desc, nullptr, nullptr, nullptr, nullptr),
                zcutlass::Status::Success);

  std::cout << "[pass] " << zcutlass::dtype_name(dtype<T>()) << " zero-size checks" << std::endl;
}

void run_invalid_argument_tests() {
  zcutlass::GemmDesc desc{};
  desc.m = 16;
  desc.n = 16;
  desc.k = 16;
  desc.lda = 16;
  desc.ldb = 16;
  desc.ldc = 16;
  desc.ldd = 16;
  desc.a_type = zcutlass::DType::F16;
  desc.b_type = zcutlass::DType::F16;
  desc.c_type = zcutlass::DType::F16;
  desc.d_type = zcutlass::DType::F16;
  desc.alpha = 1.0f;
  desc.beta = 0.0f;

  half* ptr = reinterpret_cast<half*>(0x1);
  expect_status("null A", zcutlass::gemm(desc, nullptr, ptr, nullptr, ptr),
                zcutlass::Status::InvalidArgument);
  expect_status("null B", zcutlass::gemm(desc, ptr, nullptr, nullptr, ptr),
                zcutlass::Status::InvalidArgument);
  expect_status("null D", zcutlass::gemm(desc, ptr, ptr, nullptr, nullptr),
                zcutlass::Status::InvalidArgument);

  desc.beta = 1.0f;
  expect_status("missing C with beta != 0", zcutlass::gemm(desc, ptr, ptr, nullptr, ptr),
                zcutlass::Status::InvalidArgument);

  desc.beta = 0.0f;
  desc.m = -1;
  expect_status("negative m", zcutlass::gemm(desc, ptr, ptr, nullptr, ptr),
                zcutlass::Status::InvalidArgument);

  desc.m = 16;
  desc.lda = 15;
  expect_status("lda smaller than k", zcutlass::gemm(desc, ptr, ptr, nullptr, ptr),
                zcutlass::Status::InvalidArgument);

  desc.lda = 16;
  desc.ldb = 15;
  expect_status("ldb smaller than n", zcutlass::gemm(desc, ptr, ptr, nullptr, ptr),
                zcutlass::Status::InvalidArgument);

  desc.ldb = 16;
  desc.ldd = 15;
  expect_status("ldd smaller than n", zcutlass::gemm(desc, ptr, ptr, nullptr, ptr),
                zcutlass::Status::InvalidArgument);

  desc.ldd = 16;
  desc.beta = 1.0f;
  desc.ldc = 15;
  expect_status("ldc smaller than n with beta", zcutlass::gemm(desc, ptr, ptr, ptr, ptr),
                zcutlass::Status::InvalidArgument);

  desc.ldc = 16;
  desc.beta = 0.0f;
  desc.b_type = zcutlass::DType::BF16;
  expect_status("mixed dtypes", zcutlass::gemm(desc, ptr, ptr, nullptr, ptr),
                zcutlass::Status::NotSupported);

  desc.b_type = zcutlass::DType::F16;
  desc.a_type = zcutlass::DType::F32;
  desc.b_type = zcutlass::DType::F32;
  desc.c_type = zcutlass::DType::F32;
  desc.d_type = zcutlass::DType::F32;
  expect_status("unsupported f32 storage", zcutlass::gemm(desc, ptr, ptr, nullptr, ptr),
                zcutlass::Status::NotSupported);

  desc.a_type = zcutlass::DType::F16;
  desc.b_type = zcutlass::DType::F16;
  desc.c_type = zcutlass::DType::F16;
  desc.d_type = zcutlass::DType::F16;
  desc.a_layout = zcutlass::layout::LayoutKind::ColumnMajor;
  expect_status("column-major layout", zcutlass::gemm(desc, ptr, ptr, nullptr, ptr),
                zcutlass::Status::NotSupported);

  zcutlass::gemm_api::GemmArguments args{};
  args.problem = {16, 16, 16};
  args.A = {ptr, zcutlass::DType::F16, zcutlass::layout::LayoutKind::RowMajor, 16};
  args.B = {ptr, zcutlass::DType::F16, zcutlass::layout::LayoutKind::RowMajor, 16};
  args.C = {nullptr, zcutlass::DType::F16, zcutlass::layout::LayoutKind::RowMajor, 16};
  args.D = {ptr, zcutlass::DType::F16, zcutlass::layout::LayoutKind::RowMajor, 16};
  args.alpha = 1.0f;
  args.beta = 0.0f;
  expect_status("GemmArguments row-major f16 path", zcutlass::can_implement(args),
                zcutlass::Status::Success);
  if (zcutlass::get_workspace_size(args) != 0) {
    std::cerr << "Expected v1 GEMM workspace size to be zero" << std::endl;
    std::exit(1);
  }

  std::cout << "[pass] invalid argument checks" << std::endl;
}

void run_dispatch_tests() {
  zcutlass::GemmDesc desc{8,
                          256,
                          256,
                          256,
                          256,
                          256,
                          256,
                          zcutlass::DType::F16,
                          zcutlass::DType::F16,
                          zcutlass::DType::F16,
                          zcutlass::DType::F16,
                          1.0f,
                          0.0f,
                          nullptr,
                          nullptr};
  expect_kernel_contains("small-M f16 dispatch", desc, "64x128x16");
  expect_family("small-M f16 dispatch", desc, "fallback");

  desc.m = 64;
  expect_kernel_contains("M=64 f16 dispatch", desc, "64x128x16_aligned");
  expect_family("M=64 f16 dispatch", desc, "fallback");
  expect_path("M=64 f16 dispatch", desc, "fast");

  desc.m = 8;
  desc.n = 4096;
  desc.k = 4096;
  desc.lda = 4096;
  desc.ldb = 4096;
  desc.ldc = 4096;
  desc.ldd = 4096;
  expect_family("LLM decode canonical dispatch", desc, "decode");
  expect_path("LLM decode canonical dispatch", desc, "fallback");

  desc.m = 128;
  expect_kernel_contains("LLM prefill canonical dispatch", desc,
                         "64x128x64_aligned_prefill_n_le_k");
  expect_kernel_not_contains("LLM prefill canonical dispatch", desc, "experimental");
  expect_family("LLM prefill canonical dispatch", desc, "prefill");
  expect_path("LLM prefill canonical dispatch", desc, "fast");
  expect_pipeline("LLM prefill canonical dispatch", desc, 4);
  expect_epilogue("LLM prefill canonical dispatch", desc, "shared_accumulator");

  desc.n = 16384;
  desc.k = 4096;
  desc.lda = 4096;
  desc.ldb = 16384;
  desc.ldc = 16384;
  desc.ldd = 16384;
  expect_kernel_contains("LLM prefill N>K dispatch", desc, "64x128x32_aligned_prefill");
  expect_kernel_not_contains("LLM prefill N>K dispatch", desc, "64x128x64");
  expect_kernel_not_contains("LLM prefill N>K default dispatch", desc, "64x256x32");

  set_experimental_kernel_filter("64x256x32");
  expect_kernel_contains("LLM prefill N>K experimental dispatch", desc, "64x256x32");
  expect_path("LLM prefill N>K experimental dispatch", desc, "experimental_fast");
  set_experimental_kernel_filter(nullptr);

  set_experimental_kernel_filter("64x256x16");
  expect_kernel_contains("LLM prefill N>K KGroup1 experimental dispatch", desc, "64x256x16");
  expect_path("LLM prefill N>K KGroup1 experimental dispatch", desc, "experimental_fast");
  set_experimental_kernel_filter(nullptr);

  desc.n = 4096;
  desc.k = 16384;
  desc.lda = 16384;
  desc.ldb = 4096;
  desc.ldc = 4096;
  desc.ldd = 4096;
  expect_kernel_contains("LLM prefill N<=K dispatch", desc,
                         "64x128x64_aligned_prefill_n_le_k");

  desc.m = 4096;
  desc.k = 4096;
  desc.lda = 4096;
  expect_family("large canonical dispatch", desc, "large");
  expect_path("large canonical dispatch", desc, "fast");

  desc.m = 128;
  desc.n = 4096;
  desc.k = 4096;
  desc.lda = 4096;
  desc.ldb = 4096;
  desc.ldc = 4096;
  desc.ldd = 4096;
  desc.a_type = zcutlass::DType::BF16;
  desc.b_type = zcutlass::DType::BF16;
  desc.c_type = zcutlass::DType::BF16;
  desc.d_type = zcutlass::DType::BF16;
  expect_kernel_contains("BF16 prefill default dispatch", desc, "bf16_64x128x32_aligned_prefill");
  expect_path("BF16 prefill default dispatch", desc, "fast");
  expect_pipeline("BF16 prefill default dispatch", desc, 2);
  expect_epilogue("BF16 prefill default dispatch", desc, "shared_accumulator");

  set_experimental_kernel_filter("sm120_mma_bf16");
  expect_kernel_contains("BF16 explicit-MMA prefill dispatch", desc, "sm120_mma_bf16");
  expect_path("BF16 explicit-MMA prefill dispatch", desc, "experimental_fast");
  expect_pipeline("BF16 explicit-MMA prefill dispatch", desc, 2);
  expect_epilogue("BF16 explicit-MMA prefill dispatch", desc, "register_linear");
  set_experimental_kernel_filter(nullptr);

  set_experimental_kernel_filter("32x128x64");
  expect_kernel_contains("BF16 explicit-MMA M32 prefill dispatch", desc, "32x128x64");
  expect_path("BF16 explicit-MMA M32 prefill dispatch", desc, "experimental_fast");
  set_experimental_kernel_filter(nullptr);

  desc.a_type = zcutlass::DType::F16;
  desc.b_type = zcutlass::DType::F16;
  desc.c_type = zcutlass::DType::F16;
  desc.d_type = zcutlass::DType::F16;

  set_experimental_kernel_filter("sm120_mma_f16");
  expect_kernel_contains("F16 explicit-MMA prefill dispatch", desc, "sm120_mma_f16");
  expect_path("F16 explicit-MMA prefill dispatch", desc, "experimental_fast");
  expect_pipeline("F16 explicit-MMA prefill dispatch", desc, 2);
  expect_epilogue("F16 explicit-MMA prefill dispatch", desc, "register_linear");
  set_experimental_kernel_filter(nullptr);

  desc.m = 4096;
  desc.alpha = 0.5f;
  expect_kernel_contains("alpha f16 dispatch", desc, "64x128x16");
  expect_kernel_not_contains("alpha f16 dispatch", desc, "aligned");
  expect_family("alpha f16 dispatch", desc, "large");
  expect_path("alpha f16 dispatch", desc, "fallback");

  desc.alpha = 1.0f;
  desc.m = 64;
  desc.n = 256;
  desc.k = 256;
  desc.lda = 260;
  desc.ldb = 256;
  desc.ldc = 256;
  desc.ldd = 256;
  expect_kernel_contains("padded lda f16 dispatch", desc, "64x128x16");
  expect_kernel_not_contains("padded lda f16 dispatch", desc, "aligned");
  expect_family("padded lda f16 dispatch", desc, "fallback");

  desc.lda = 256;
  desc.n = 64;
  desc.ldb = 64;
  desc.ldc = 64;
  desc.ldd = 64;
  expect_kernel_contains("n=64 f16 dispatch", desc, "64x64x16");

  desc.a_type = zcutlass::DType::BF16;
  desc.b_type = zcutlass::DType::BF16;
  desc.c_type = zcutlass::DType::BF16;
  desc.d_type = zcutlass::DType::BF16;
  expect_kernel_contains("n=64 bf16 dispatch", desc, "bf16_64x64x16");

  std::cout << "[pass] dispatch checks" << std::endl;
}

}  // namespace

int main() {
  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  std::cout << "Running zcutlass tests on " << prop.name << " sm_" << prop.major << prop.minor
            << std::endl;

  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  run_invalid_argument_tests();
  run_dispatch_tests();
  run_zero_size_tests<half>();
  run_zero_size_tests<__nv_bfloat16>();

  run_case<half>("f16 tiny padded bias", handle, 15, 17, 19, 1.0f, 0.25f, true, true);
  run_case<half>("f16 16x16", handle, 16, 16, 16, 1.0f, 0.0f, false, false);
  run_case<half>("f16 rectangular", handle, 65, 129, 31, 0.75f, 0.5f, true, true);
  run_case<half>("f16 ragged alpha no-bias padded A/D", handle, 31, 63, 17, -0.5f, 0.0f, false, 2, 0, 0, 9);
  run_case<half>("f16 ragged beta no-bias", handle, 47, 64, 33, 0.5f, -0.25f, false, false);
  run_case<half>("f16 llm-smoke", handle, 8, 256, 256, 1.0f, 0.0f, false, false);
  set_experimental_kernel_filter("sm120_mma_f16");
  run_case<half>("f16 explicit-mma prefill smoke", handle, 64, 1024, 1024, 1.0f, 0.0f, false, false);
  set_experimental_kernel_filter(nullptr);
  set_experimental_kernel_filter("32x128x64");
  run_case<half>("f16 explicit-mma m32 prefill smoke", handle, 32, 1024, 1024, 1.0f, 0.0f, false, false);
  set_experimental_kernel_filter(nullptr);

  run_case<__nv_bfloat16>("bf16 tiny padded bias", handle, 13, 19, 23, 1.0f, 0.25f, true, true);
  run_case<__nv_bfloat16>("bf16 16x16", handle, 16, 16, 16, 1.0f, 0.0f, false, false);
  run_case<__nv_bfloat16>("bf16 rectangular", handle, 67, 127, 29, 0.75f, 0.5f, true, true);
  run_case<__nv_bfloat16>("bf16 ragged alpha no-bias padded B/C/D", handle, 29, 61, 15, -0.75f, 0.0f, false, 0, 4, 6, 10);
  run_case<__nv_bfloat16>("bf16 ragged beta no-bias", handle, 45, 64, 31, 0.5f, -0.25f, false, false);
  run_case<__nv_bfloat16>("bf16 llm-smoke", handle, 8, 256, 256, 1.0f, 0.0f, false, false);
  set_experimental_kernel_filter("sm120_mma_bf16");
  run_case<__nv_bfloat16>("bf16 explicit-mma prefill smoke", handle, 64, 1024, 1024, 1.0f, 0.0f, false, false);
  set_experimental_kernel_filter(nullptr);
  set_experimental_kernel_filter("32x128x64");
  run_case<__nv_bfloat16>("bf16 explicit-mma m32 prefill smoke", handle, 32, 1024, 1024, 1.0f, 0.0f, false, false);
  set_experimental_kernel_filter(nullptr);

  CHECK_CUBLAS(cublasDestroy(handle));
  std::cout << "All zcutlass tests passed." << std::endl;
  return 0;
}
