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
#include <fstream>
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
  std::string providers = "zcutlass,cublas";
  std::string output;
  int warmup = 10;
  int iterations = 50;
  float alpha = 1.0f;
  float beta = 0.0f;
  bool json = false;
  bool append = false;
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
    } else if (arg == "--providers") {
      options.providers = require_value("--providers");
    } else if (arg == "--output") {
      options.output = require_value("--output");
    } else if (arg == "--append") {
      options.append = true;
    } else if (arg == "--warmup") {
      options.warmup = std::atoi(require_value("--warmup"));
    } else if (arg == "--iterations") {
      options.iterations = std::atoi(require_value("--iterations"));
    } else if (arg == "--alpha") {
      options.alpha = std::atof(require_value("--alpha"));
    } else if (arg == "--beta") {
      options.beta = std::atof(require_value("--beta"));
    } else if (arg == "--json") {
      options.json = true;
    } else if (arg == "--help") {
      std::cout << "Usage: zcutlass_bench [--m M --n N --k K] [--dtype f16|bf16]\n"
                << "                      [--suite single|correctness|smoke|llm|llm-decode|llm-prefill|square|ragged]\n"
                << "                      [--providers zcutlass,cublas] [--json] [--output FILE]\n"
                << "                      [--warmup N] [--iterations N] [--alpha X] [--beta X]\n";
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
  if (options.suite == "correctness") {
    return {{15, 17, 19}, {16, 16, 16}, {65, 129, 31}, {67, 127, 29}};
  }
  if (options.suite == "square") {
    return {{512, 512, 512}, {1024, 1024, 1024}, {2048, 2048, 2048}, {4096, 4096, 4096}};
  }
  if (options.suite == "ragged") {
    return {{63, 127, 65}, {65, 129, 127}, {127, 255, 129}, {257, 511, 255}};
  }
  if (options.suite == "llm" || options.suite == "llm-decode" ||
      options.suite == "llm-prefill") {
    std::vector<Shape> shapes;
    const int64_t hs[] = {1024, 2048, 4096, 8192};
    const int64_t decode_ms[] = {1, 2, 4, 8, 16};
    const int64_t prefill_ms[] = {64, 128, 256, 512, 1024};
    const int64_t all_ms[] = {1, 8, 16, 64, 256, 1024};
    for (int64_t h : hs) {
      const int64_t* ms = all_ms;
      int count = 6;
      if (options.suite == "llm-decode") {
        ms = decode_ms;
        count = 5;
      } else if (options.suite == "llm-prefill") {
        ms = prefill_ms;
        count = 5;
      }
      for (int i = 0; i < count; ++i) {
        const int64_t m = ms[i];
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

bool provider_enabled(const Options& options, const std::string& provider) {
  return options.providers == "all" || options.providers.find(provider) != std::string::npos;
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

void emit_schema_record(std::ostream& out,
                        const Options& options,
                        const Shape& shape,
                        const char* provider,
                        const char* kernel,
                        float ms,
                        double achieved_tflops,
                        const cudaDeviceProp& prop) {
  out << std::fixed << std::setprecision(4)
      << "{\"schema_version\":1"
      << ",\"problem\":{\"operation\":\"gemm\",\"m\":" << shape.m << ",\"n\":"
      << shape.n << ",\"k\":" << shape.k << ",\"dtype\":\"" << options.dtype
      << "\",\"layout\":\"row,row,row,row\",\"alpha\":" << options.alpha
      << ",\"beta\":" << options.beta << "}"
      << ",\"provider\":\"" << provider << "\",\"status\":\"success\""
      << ",\"kernel\":\"" << kernel << "\""
      << ",\"performance\":{\"warmup_iterations\":" << options.warmup
      << ",\"profiling_iterations\":" << options.iterations << ",\"median_ms\":"
      << ms << ",\"tflops\":" << achieved_tflops << "}"
      << ",\"environment\":{\"gpu_name\":\"" << prop.name << "\",\"sm\":"
      << prop.major << prop.minor << ",\"zcutlass_version\":\""
      << zcutlass::version_major() << "." << zcutlass::version_minor() << "."
      << zcutlass::version_patch() << "\",\"registered_gemm_operations\":"
      << zcutlass::registered_gemm_operation_count() << "}"
      << ",\"tags\":{\"suite\":\"" << options.suite << "\"}}"
      << std::endl;
}

template <typename T>
void run_shape(const Shape& shape,
               const Options& options,
               cublasHandle_t handle,
               const cudaDeviceProp& prop,
               std::ostream* output) {
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
                          options.alpha,
                          options.beta,
                          nullptr,
                          nullptr};

  const auto z_fn = [&]() {
    const zcutlass::Status status =
        zcutlass::gemm(desc, d_a, d_b, options.beta != 0.0f ? d_c : nullptr, d_z);
    if (status != zcutlass::Status::Success) {
      std::cerr << "zcutlass::gemm failed: " << zcutlass::status_to_string(status) << std::endl;
      std::exit(1);
    }
  };

  const float alpha = options.alpha;
  const float beta = options.beta;
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

  float z_ms = 0.0f;
  float blas_ms = 0.0f;
  double z_tflops = 0.0;
  double blas_tflops = 0.0;
  if (provider_enabled(options, "zcutlass")) {
    z_ms = time_ms(z_fn, options.warmup, options.iterations);
    z_tflops = tflops(shape, z_ms);
    if (output != nullptr) {
      emit_schema_record(*output, options, shape, "zcutlass", zcutlass::selected_kernel_name(desc),
                         z_ms, z_tflops, prop);
    }
  }
  if (provider_enabled(options, "cublas")) {
    blas_ms = time_ms(blas_fn, options.warmup, options.iterations);
    blas_tflops = tflops(shape, blas_ms);
    if (output != nullptr) {
      emit_schema_record(*output, options, shape, "cublas", "cublasGemmEx", blas_ms,
                         blas_tflops, prop);
    }
  }
  const double speedup = (z_ms > 0.0f && blas_ms > 0.0f) ? blas_ms / z_ms : 0.0;

  if (options.json) {
    std::cout << std::fixed << std::setprecision(4)
              << "{\"m\":" << shape.m << ",\"n\":" << shape.n << ",\"k\":" << shape.k
              << ",\"dtype\":\"" << options.dtype << "\",\"zcutlass_ms\":" << z_ms
              << ",\"kernel\":\"" << zcutlass::selected_kernel_name(desc)
              << "\",\"cublas_ms\":" << blas_ms << ",\"zcutlass_tflops\":" << z_tflops
              << ",\"cublas_tflops\":" << blas_tflops << ",\"speedup\":" << speedup << "}"
              << std::endl;
  } else {
    std::cout << std::fixed << std::setprecision(4) << "m=" << shape.m << " n=" << shape.n
              << " k=" << shape.k << " dtype=" << options.dtype << " kernel="
              << zcutlass::selected_kernel_name(desc) << " zcutlass=" << z_ms
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

  std::ofstream output_file;
  std::ostream* output = nullptr;
  if (!options.output.empty()) {
    output_file.open(options.output, options.append ? std::ios::app : std::ios::out);
    if (!output_file) {
      std::cerr << "Failed to open output file: " << options.output << std::endl;
      return 2;
    }
    output = &output_file;
  }

  const std::vector<Shape> shapes = make_shapes(options);
  for (const Shape& shape : shapes) {
    if (options.dtype == "f16") {
      run_shape<half>(shape, options, handle, prop, output);
    } else if (options.dtype == "bf16") {
      run_shape<__nv_bfloat16>(shape, options, handle, prop, output);
    } else if (options.dtype == "both") {
      Options f16_options = options;
      f16_options.dtype = "f16";
      run_shape<half>(shape, f16_options, handle, prop, output);
      Options bf16_options = options;
      bf16_options.dtype = "bf16";
      run_shape<__nv_bfloat16>(shape, bf16_options, handle, prop, output);
    } else {
      std::cerr << "Unsupported dtype: " << options.dtype << std::endl;
      return 2;
    }
  }

  CHECK_CUBLAS(cublasDestroy(handle));
  return 0;
}
