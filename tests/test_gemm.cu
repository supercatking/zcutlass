#include "zcutlass/gemm.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
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
  const int64_t lda = k + (padded ? 3 : 0);
  const int64_t ldb = n + (padded ? 5 : 0);
  const int64_t ldc = n + (padded ? 7 : 0);
  const int64_t ldd = n + (padded ? 11 : 0);

  std::vector<T> h_a = make_tensor<T>(m, k, lda, 17);
  std::vector<T> h_b = make_tensor<T>(k, n, ldb, 29);
  std::vector<T> h_c = make_tensor<T>(m, n, ldc, 41);
  std::vector<T> h_d(m * ldd, from_float<T>(0.0f));
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

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_d));
  CHECK_CUDA(cudaFree(d_blas));
  CHECK_CUDA(cudaFree(d_bias));
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
  if (zcutlass::gemm(desc, nullptr, ptr, nullptr, ptr) != zcutlass::Status::InvalidArgument) {
    std::cerr << "Expected null A to be InvalidArgument" << std::endl;
    std::exit(1);
  }

  desc.beta = 1.0f;
  if (zcutlass::gemm(desc, ptr, ptr, nullptr, ptr) != zcutlass::Status::InvalidArgument) {
    std::cerr << "Expected missing C with beta != 0 to be InvalidArgument" << std::endl;
    std::exit(1);
  }

  desc.beta = 0.0f;
  desc.b_type = zcutlass::DType::BF16;
  if (zcutlass::gemm(desc, ptr, ptr, nullptr, ptr) != zcutlass::Status::NotSupported) {
    std::cerr << "Expected mixed dtypes to be NotSupported" << std::endl;
    std::exit(1);
  }

  desc.b_type = zcutlass::DType::F16;
  desc.a_layout = zcutlass::layout::LayoutKind::ColumnMajor;
  if (zcutlass::gemm(desc, ptr, ptr, nullptr, ptr) != zcutlass::Status::NotSupported) {
    std::cerr << "Expected column-major layout to be NotSupported for v1" << std::endl;
    std::exit(1);
  }

  zcutlass::gemm_api::GemmArguments args{};
  args.problem = {16, 16, 16};
  args.A = {ptr, zcutlass::DType::F16, zcutlass::layout::LayoutKind::RowMajor, 16};
  args.B = {ptr, zcutlass::DType::F16, zcutlass::layout::LayoutKind::RowMajor, 16};
  args.C = {nullptr, zcutlass::DType::F16, zcutlass::layout::LayoutKind::RowMajor, 16};
  args.D = {ptr, zcutlass::DType::F16, zcutlass::layout::LayoutKind::RowMajor, 16};
  args.alpha = 1.0f;
  args.beta = 0.0f;
  if (zcutlass::can_implement(args) != zcutlass::Status::Success) {
    std::cerr << "Expected GemmArguments row-major f16 path to be implementable" << std::endl;
    std::exit(1);
  }
  if (zcutlass::get_workspace_size(args) != 0) {
    std::cerr << "Expected v1 GEMM workspace size to be zero" << std::endl;
    std::exit(1);
  }

  std::cout << "[pass] invalid argument checks" << std::endl;
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

  run_case<half>("f16 tiny padded bias", handle, 15, 17, 19, 1.0f, 0.25f, true, true);
  run_case<half>("f16 16x16", handle, 16, 16, 16, 1.0f, 0.0f, false, false);
  run_case<half>("f16 rectangular", handle, 65, 129, 31, 0.75f, 0.5f, true, true);
  run_case<half>("f16 llm-smoke", handle, 8, 256, 256, 1.0f, 0.0f, false, false);

  run_case<__nv_bfloat16>("bf16 tiny padded bias", handle, 13, 19, 23, 1.0f, 0.25f, true, true);
  run_case<__nv_bfloat16>("bf16 16x16", handle, 16, 16, 16, 1.0f, 0.0f, false, false);
  run_case<__nv_bfloat16>("bf16 rectangular", handle, 67, 127, 29, 0.75f, 0.5f, true, true);
  run_case<__nv_bfloat16>("bf16 llm-smoke", handle, 8, 256, 256, 1.0f, 0.0f, false, false);

  CHECK_CUBLAS(cublasDestroy(handle));
  std::cout << "All zcutlass tests passed." << std::endl;
  return 0;
}
