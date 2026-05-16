#include "zcutlass/gemm.hpp"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
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

struct Options {
  int64_t m = 256;
  int64_t n = 4096;
  int64_t k = 4096;
  std::string dtype = "f16";
  std::string suite = "single";
  int warmup = 10;
  int iterations = 50;
  bool json = false;
};

struct Shape {
  int64_t m;
  int64_t n;
  int64_t k;
};

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
zcutlass::DType z_dtype();

template <>
zcutlass::DType z_dtype<half>() {
  return zcutlass::DType::F16;
}

template <>
zcutlass::DType z_dtype<__nv_bfloat16>() {
  return zcutlass::DType::BF16;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* name) -> const char* {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << std::endl;
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--m") {
      options.m = std::atoll(require_value("--m"));
    } else if (arg == "--n") {
      options.n = std::atoll(require_value("--n"));
    } else if (arg == "--k") {
      options.k = std::atoll(require_value("--k"));
    } else if (arg == "--dtype") {
      options.dtype = require_value("--dtype");
    } else if (arg == "--suite") {
      options.suite = require_value("--suite");
    } else if (arg == "--warmup") {
      options.warmup = std::atoi(require_value("--warmup"));
    } else if (arg == "--iterations") {
      options.iterations = std::atoi(require_value("--iterations"));
    } else if (arg == "--json") {
      options.json = true;
    } else if (arg == "--help") {
      std::cout << "Usage: zcutlass_bench [--m M --n N --k K] [--dtype f16|bf16]\n"
                << "                      [--suite single|smoke|llm] [--json]\n"
                << "                      [--warmup N] [--iterations N]\n";
      std::exit(0);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::exit(2);
    }
  }
  return options;
}

std::vector<Shape> make_shapes(const Options& options) {
  if (options.suite == "single") {
    return {{options.m, options.n, options.k}};
  }
  if (options.suite == "smoke") {
    return {{1, 1024, 1024}, {8, 2048, 2048}, {64, 4096, 4096}, {256, 2048, 8192}};
  }
  if (options.suite == "llm") {
    std::vector<Shape> shapes;
    const int64_t hs[] = {1024, 2048, 4096, 8192};
    const int64_t ms[] = {1, 8, 16, 64, 256, 1024};
    for (int64_t h : hs) {
      for (int64_t m : ms) {
        shapes.push_back({m, h, h});
        shapes.push_back({m, 4 * h, h});
        shapes.push_back({m, h, 4 * h});
      }
    }
    shapes.push_back({1024, 1024, 1024});
    shapes.push_back({2048, 2048, 2048});
    shapes.push_back({4096, 4096, 4096});
    return shapes;
  }
  std::cerr << "Unknown suite: " << options.suite << std::endl;
  std::exit(2);
}

template <typename T>
std::vector<T> make_tensor(size_t count, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
  std::vector<T> values(count);
  for (T& value : values) {
    value = from_float<T>(dist(rng));
  }
  return values;
}

float median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

template <typename Fn>
float time_ms(Fn&& fn, int warmup, int iterations) {
  for (int i = 0; i < warmup; ++i) {
    fn();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  std::vector<float> samples;
  samples.reserve(iterations);
  for (int i = 0; i < iterations; ++i) {
    CHECK_CUDA(cudaEventRecord(start));
    fn();
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(ms);
  }
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return median(samples);
}

double tflops(const Shape& shape, float ms) {
  const double flops = 2.0 * static_cast<double>(shape.m) * shape.n * shape.k;
  return flops / (static_cast<double>(ms) * 1.0e-3) / 1.0e12;
}

template <typename T>
void run_shape(const Shape& shape, const Options& options, cublasHandle_t handle) {
  const int64_t lda = shape.k;
  const int64_t ldb = shape.n;
  const int64_t ldc = shape.n;
  const int64_t ldd = shape.n;
  const size_t a_count = static_cast<size_t>(shape.m * lda);
  const size_t b_count = static_cast<size_t>(shape.k * ldb);
  const size_t c_count = static_cast<size_t>(shape.m * ldc);
  const size_t d_count = static_cast<size_t>(shape.m * ldd);

  std::vector<T> h_a = make_tensor<T>(a_count, 11);
  std::vector<T> h_b = make_tensor<T>(b_count, 23);
  std::vector<T> h_c = make_tensor<T>(c_count, 37);

  T* d_a = nullptr;
  T* d_b = nullptr;
  T* d_c = nullptr;
  T* d_z = nullptr;
  T* d_blas = nullptr;
  CHECK_CUDA(cudaMalloc(&d_a, a_count * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_b, b_count * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_c, c_count * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_z, d_count * sizeof(T)));
  CHECK_CUDA(cudaMalloc(&d_blas, d_count * sizeof(T)));
  CHECK_CUDA(cudaMemcpy(d_a, h_a.data(), a_count * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, h_b.data(), b_count * sizeof(T), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_c, h_c.data(), c_count * sizeof(T), cudaMemcpyHostToDevice));

  zcutlass::GemmDesc desc{shape.m,
                          shape.n,
                          shape.k,
                          lda,
                          ldb,
                          ldc,
                          ldd,
                          z_dtype<T>(),
                          z_dtype<T>(),
                          z_dtype<T>(),
                          z_dtype<T>(),
                          1.0f,
                          0.0f,
                          nullptr,
                          nullptr};

  const auto z_fn = [&]() {
    const zcutlass::Status status = zcutlass::gemm(desc, d_a, d_b, nullptr, d_z);
    if (status != zcutlass::Status::Success) {
      std::cerr << "zcutlass::gemm failed: " << zcutlass::status_to_string(status) << std::endl;
      std::exit(1);
    }
  };

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const auto blas_fn = [&]() {
    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              static_cast<int>(shape.n),
                              static_cast<int>(shape.m),
                              static_cast<int>(shape.k),
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
  };

  const float z_ms = time_ms(z_fn, options.warmup, options.iterations);
  const float blas_ms = time_ms(blas_fn, options.warmup, options.iterations);
  const double z_tflops = tflops(shape, z_ms);
  const double blas_tflops = tflops(shape, blas_ms);
  const double speedup = blas_ms / z_ms;

  if (options.json) {
    std::cout << std::fixed << std::setprecision(4)
              << "{\"m\":" << shape.m << ",\"n\":" << shape.n << ",\"k\":" << shape.k
              << ",\"dtype\":\"" << options.dtype << "\",\"zcutlass_ms\":" << z_ms
              << ",\"cublas_ms\":" << blas_ms << ",\"zcutlass_tflops\":" << z_tflops
              << ",\"cublas_tflops\":" << blas_tflops << ",\"speedup\":" << speedup << "}"
              << std::endl;
  } else {
    std::cout << std::fixed << std::setprecision(4) << "m=" << shape.m << " n=" << shape.n
              << " k=" << shape.k << " dtype=" << options.dtype << " zcutlass=" << z_ms
              << " ms (" << z_tflops << " TFLOP/s) cublas=" << blas_ms << " ms ("
              << blas_tflops << " TFLOP/s) speedup=" << speedup << "x" << std::endl;
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));
  CHECK_CUDA(cudaFree(d_z));
  CHECK_CUDA(cudaFree(d_blas));
}

}  // namespace

int main(int argc, char** argv) {
  Options options = parse_options(argc, argv);
  if (options.iterations <= 0 || options.warmup < 0) {
    std::cerr << "Invalid warmup/iteration counts" << std::endl;
    return 2;
  }

  int device = 0;
  CHECK_CUDA(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
  if (!options.json) {
    std::cout << "GPU: " << prop.name << " sm_" << prop.major << prop.minor << std::endl;
  }

  cublasHandle_t handle = nullptr;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  const std::vector<Shape> shapes = make_shapes(options);
  for (const Shape& shape : shapes) {
    if (options.dtype == "f16") {
      run_shape<half>(shape, options, handle);
    } else if (options.dtype == "bf16") {
      run_shape<__nv_bfloat16>(shape, options, handle);
    } else {
      std::cerr << "Unsupported dtype: " << options.dtype << std::endl;
      return 2;
    }
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}

