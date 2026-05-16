#include "zcutlass/gemm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>

namespace zcutlass {
namespace {

using nvcuda::wmma::col_major;
using nvcuda::wmma::fragment;
using nvcuda::wmma::load_matrix_sync;
using nvcuda::wmma::matrix_a;
using nvcuda::wmma::matrix_b;
using nvcuda::wmma::mma_sync;
using nvcuda::wmma::row_major;
using nvcuda::wmma::store_matrix_sync;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kWarpSize = 32;
constexpr int kWarpTileM = 32;
constexpr int kWarpTileN = 32;
constexpr int kMatricesPerWarp = 4;
constexpr int kMatrixElements = kWmmaM * kWmmaN;
constexpr int kTileElementsPerWarp = kMatricesPerWarp * kMatrixElements;

template <typename T>
__device__ T from_float(float value);

template <>
__device__ half from_float<half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <typename T>
__device__ float to_float(T value);

template <>
__device__ float to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T, int BlockM, int BlockN>
__global__ void wmma_gemm_kernel(GemmDesc desc,
                                 const T* __restrict__ A,
                                 const T* __restrict__ B,
                                 const T* __restrict__ C,
                                 T* __restrict__ D) {
  constexpr int kWarpTilesM = BlockM / kWarpTileM;
  constexpr int kWarpTilesN = BlockN / kWarpTileN;
  constexpr int kWarpsPerBlock = kWarpTilesM * kWarpTilesN;

  static_assert(BlockM % kWarpTileM == 0, "BlockM must be a multiple of 32");
  static_assert(BlockN % kWarpTileN == 0, "BlockN must be a multiple of 32");
  static_assert(kWarpsPerBlock <= 16, "This kernel keeps shared memory under 96 KiB");

  extern __shared__ __align__(16) unsigned char shared_storage[];
  T* a_tile = reinterpret_cast<T*>(shared_storage);
  T* b_tile = a_tile + BlockM * kWmmaK;
  float* accumulator_tiles =
      reinterpret_cast<float*>(b_tile + kWmmaK * BlockN);

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;

  if (warp_id >= kWarpsPerBlock) {
    return;
  }

  const int warp_m = warp_id / kWarpTilesN;
  const int warp_n = warp_id % kWarpTilesN;
  const int block_m = static_cast<int>(blockIdx.y) * BlockM;
  const int block_n = static_cast<int>(blockIdx.x) * BlockN;
  const int tile_m = block_m + warp_m * kWarpTileM;
  const int tile_n = block_n + warp_n * kWarpTileN;

  const int local_m = warp_m * kWarpTileM;
  const int local_n = warp_n * kWarpTileN;

  float* warp_accumulators = accumulator_tiles + warp_id * kTileElementsPerWarp;
  float* c00_smem = warp_accumulators;
  float* c01_smem = c00_smem + kMatrixElements;
  float* c10_smem = c01_smem + kMatrixElements;
  float* c11_smem = c10_smem + kMatrixElements;

  fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, T, row_major> a0_frag;
  fragment<matrix_a, kWmmaM, kWmmaN, kWmmaK, T, row_major> a1_frag;
  fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, T, row_major> b0_frag;
  fragment<matrix_b, kWmmaM, kWmmaN, kWmmaK, T, row_major> b1_frag;
  fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c00_frag;
  fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c01_frag;
  fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c10_frag;
  fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c11_frag;

  nvcuda::wmma::fill_fragment(c00_frag, 0.0f);
  nvcuda::wmma::fill_fragment(c01_frag, 0.0f);
  nvcuda::wmma::fill_fragment(c10_frag, 0.0f);
  nvcuda::wmma::fill_fragment(c11_frag, 0.0f);

  const T zero = from_float<T>(0.0f);

  for (int64_t k0 = 0; k0 < desc.k; k0 += kWmmaK) {
    for (int idx = tid; idx < BlockM * kWmmaK; idx += blockDim.x) {
      const int row = idx / kWmmaK;
      const int col = idx % kWmmaK;
      const int64_t gm = block_m + row;
      const int64_t gk = k0 + col;
      a_tile[idx] = (gm < desc.m && gk < desc.k) ? A[gm * desc.lda + gk] : zero;
    }

    for (int idx = tid; idx < kWmmaK * BlockN; idx += blockDim.x) {
      const int row = idx / BlockN;
      const int col = idx % BlockN;
      const int64_t gk = k0 + row;
      const int64_t gn = block_n + col;
      b_tile[idx] = (gk < desc.k && gn < desc.n) ? B[gk * desc.ldb + gn] : zero;
    }

    __syncthreads();
    load_matrix_sync(a0_frag, a_tile + local_m * kWmmaK, kWmmaK);
    load_matrix_sync(a1_frag, a_tile + (local_m + kWmmaM) * kWmmaK, kWmmaK);
    load_matrix_sync(b0_frag, b_tile + local_n, BlockN);
    load_matrix_sync(b1_frag, b_tile + local_n + kWmmaN, BlockN);

    mma_sync(c00_frag, a0_frag, b0_frag, c00_frag);
    mma_sync(c01_frag, a0_frag, b1_frag, c01_frag);
    mma_sync(c10_frag, a1_frag, b0_frag, c10_frag);
    mma_sync(c11_frag, a1_frag, b1_frag, c11_frag);
    __syncthreads();
  }

  store_matrix_sync(c00_smem, c00_frag, kWmmaN, nvcuda::wmma::mem_row_major);
  store_matrix_sync(c01_smem, c01_frag, kWmmaN, nvcuda::wmma::mem_row_major);
  store_matrix_sync(c10_smem, c10_frag, kWmmaN, nvcuda::wmma::mem_row_major);
  store_matrix_sync(c11_smem, c11_frag, kWmmaN, nvcuda::wmma::mem_row_major);
  __syncwarp();

  for (int idx = lane_id; idx < kTileElementsPerWarp; idx += kWarpSize) {
    const int matrix = idx / kMatrixElements;
    const int local = idx % kMatrixElements;
    const int row = local / kWmmaN;
    const int col = local % kWmmaN;

    int64_t gm = tile_m;
    int64_t gn = tile_n;
    const float* accum = c00_smem;
    if (matrix == 0) {
      gm += row;
      gn += col;
      accum = c00_smem;
    } else if (matrix == 1) {
      gm += row;
      gn += kWmmaN + col;
      accum = c01_smem;
    } else if (matrix == 2) {
      gm += kWmmaM + row;
      gn += col;
      accum = c10_smem;
    } else {
      gm += kWmmaM + row;
      gn += kWmmaN + col;
      accum = c11_smem;
    }

    if (gm < desc.m && gn < desc.n) {
      float value = desc.alpha * accum[local];
      if (desc.beta != 0.0f && C != nullptr) {
        value += desc.beta * to_float<T>(C[gm * desc.ldc + gn]);
      }
      if (desc.bias != nullptr) {
        const T* bias = static_cast<const T*>(desc.bias);
        value += to_float<T>(bias[gn]);
      }
      D[gm * desc.ldd + gn] = from_float<T>(value);
    }
  }
}

Status validate_desc(const GemmDesc& desc,
                     const void* A,
                     const void* B,
                     const void* C,
                     const void* D) {
  if (desc.m < 0 || desc.n < 0 || desc.k < 0) {
    return Status::InvalidArgument;
  }
  if (desc.m == 0 || desc.n == 0 || desc.k == 0) {
    return Status::Success;
  }
  if (A == nullptr || B == nullptr || D == nullptr) {
    return Status::InvalidArgument;
  }
  if (desc.beta != 0.0f && C == nullptr) {
    return Status::InvalidArgument;
  }
  if (desc.lda < desc.k || desc.ldb < desc.n || desc.ldd < desc.n) {
    return Status::InvalidArgument;
  }
  if (desc.beta != 0.0f && desc.ldc < desc.n) {
    return Status::InvalidArgument;
  }
  if (desc.a_type != desc.b_type || desc.a_type != desc.c_type ||
      desc.a_type != desc.d_type) {
    return Status::NotSupported;
  }
  if (!is_supported_gemm_storage_dtype(desc.a_type)) {
    return Status::NotSupported;
  }
  if (desc.a_layout != layout::LayoutKind::RowMajor ||
      desc.b_layout != layout::LayoutKind::RowMajor ||
      desc.c_layout != layout::LayoutKind::RowMajor ||
      desc.d_layout != layout::LayoutKind::RowMajor) {
    return Status::NotSupported;
  }
  return Status::Success;
}

template <typename T, int BlockM, int BlockN>
Status launch_wmma(const GemmDesc& desc, const void* A, const void* B, const void* C, void* D) {
  constexpr int kWarpsPerBlock = (BlockM / kWarpTileM) * (BlockN / kWarpTileN);
  constexpr int kThreads = kWarpsPerBlock * kWarpSize;
  const dim3 block(kThreads);
  const dim3 grid((static_cast<unsigned int>(desc.n) + BlockN - 1) / BlockN,
                  (static_cast<unsigned int>(desc.m) + BlockM - 1) / BlockM);
  const size_t shared_bytes =
      (BlockM * kWmmaK + kWmmaK * BlockN) * sizeof(T) +
      kWarpsPerBlock * kTileElementsPerWarp * sizeof(float);

  auto kernel = wmma_gemm_kernel<T, BlockM, BlockN>;
  const cudaError_t attr_status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(shared_bytes));
  if (attr_status != cudaSuccess && attr_status != cudaErrorInvalidValue) {
    return Status::RuntimeError;
  }

  kernel<<<grid, block, shared_bytes, desc.stream>>>(
      desc, static_cast<const T*>(A), static_cast<const T*>(B), static_cast<const T*>(C),
      static_cast<T*>(D));

  return cudaGetLastError() == cudaSuccess ? Status::Success : Status::RuntimeError;
}

template <typename T, int BlockM, int BlockN, DType Type, bool AlignedNoBetaBias>
class WmmaGemmOperation final : public gemm_api::GemmOperation {
 public:
  constexpr WmmaGemmOperation(const char* name) : description_{name,
                                                              Type,
                                                              Type,
                                                              Type,
                                                              Type,
                                                              DType::F32,
                                                              layout::LayoutKind::RowMajor,
                                                              layout::LayoutKind::RowMajor,
                                                              layout::LayoutKind::RowMajor,
                                                              layout::LayoutKind::RowMajor,
                                                              arch::ArchKind::Sm80,
                                                              arch::OpClass::TensorOp,
                                                              BlockM,
                                                              BlockN,
                                                              kWmmaK,
                                                              AlignedNoBetaBias,
                                                              !AlignedNoBetaBias,
                                                              !AlignedNoBetaBias} {}

  const gemm_api::GemmOperationDescription& description() const override {
    return description_;
  }

  bool can_implement(const gemm_api::GemmArguments& args,
                     const gemm_api::GemmPreference& pref) const override {
    (void)pref;
    if (args.A.dtype != Type || args.B.dtype != Type || args.C.dtype != Type ||
        args.D.dtype != Type) {
      return false;
    }
    if (args.A.layout != layout::LayoutKind::RowMajor ||
        args.B.layout != layout::LayoutKind::RowMajor ||
        args.C.layout != layout::LayoutKind::RowMajor ||
        args.D.layout != layout::LayoutKind::RowMajor) {
      return false;
    }
    if (BlockN == 128 && args.problem.n < 128) {
      return false;
    }
    if (BlockM == 32 && args.problem.m > 32) {
      return false;
    }
    if (AlignedNoBetaBias) {
      if (args.problem.m % BlockM != 0 || args.problem.n % BlockN != 0 ||
          args.problem.k % kWmmaK != 0) {
        return false;
      }
      if (args.alpha != 1.0f || args.beta != 0.0f || args.bias != nullptr) {
        return false;
      }
      if (args.A.ld != args.problem.k || args.B.ld != args.problem.n ||
          args.D.ld != args.problem.n) {
        return false;
      }
    }
    return arch::supports(arch::current_device_cc(), description_.min_arch);
  }

  Status run(const gemm_api::GemmArguments& args,
             const gemm_api::GemmPreference& pref) const override {
    (void)pref;
    GemmDesc desc{args.problem.m,
                  args.problem.n,
                  args.problem.k,
                  args.A.ld,
                  args.B.ld,
                  args.C.ld,
                  args.D.ld,
                  args.A.dtype,
                  args.B.dtype,
                  args.C.dtype,
                  args.D.dtype,
                  args.alpha,
                  args.beta,
                  args.bias,
                  args.stream,
                  args.A.layout,
                  args.B.layout,
                  args.C.layout,
                  args.D.layout};
    return launch_wmma<T, BlockM, BlockN>(desc, args.A.data, args.B.data, args.C.data,
                                          args.D.data);
  }

 private:
  gemm_api::GemmOperationDescription description_;
};

gemm_api::GemmArguments make_arguments(const GemmDesc& desc,
                                       const void* A,
                                       const void* B,
                                       const void* C,
                                       void* D) {
  gemm_api::GemmArguments args{};
  args.problem = {desc.m, desc.n, desc.k};
  args.A = {A, desc.a_type, desc.a_layout, desc.lda};
  args.B = {B, desc.b_type, desc.b_layout, desc.ldb};
  args.C = {C, desc.c_type, desc.c_layout, desc.ldc};
  args.D = {D, desc.d_type, desc.d_layout, desc.ldd};
  args.alpha = desc.alpha;
  args.beta = desc.beta;
  args.bias = desc.bias;
  args.stream = desc.stream;
  return args;
}

void ensure_builtin_operations_registered() {
  static bool initialized = false;
  if (initialized) {
    return;
  }

  static const WmmaGemmOperation<half, 64, 128, DType::F16, true> f16_64x128_aligned(
      "zcutlass_sm120_tensorop_f16_64x128x16_aligned");
  static const WmmaGemmOperation<__nv_bfloat16, 64, 128, DType::BF16, true>
      bf16_64x128_aligned("zcutlass_sm120_tensorop_bf16_64x128x16_aligned");
  static const WmmaGemmOperation<half, 32, 128, DType::F16, false> f16_32x128(
      "zcutlass_sm120_tensorop_f16_32x128x16");
  static const WmmaGemmOperation<half, 32, 64, DType::F16, false> f16_32x64(
      "zcutlass_sm120_tensorop_f16_32x64x16");
  static const WmmaGemmOperation<__nv_bfloat16, 32, 128, DType::BF16, false> bf16_32x128(
      "zcutlass_sm120_tensorop_bf16_32x128x16");
  static const WmmaGemmOperation<__nv_bfloat16, 32, 64, DType::BF16, false> bf16_32x64(
      "zcutlass_sm120_tensorop_bf16_32x64x16");
  static const WmmaGemmOperation<half, 64, 128, DType::F16, false> f16_64x128(
      "zcutlass_sm120_tensorop_f16_64x128x16");
  static const WmmaGemmOperation<half, 64, 64, DType::F16, false> f16_64x64(
      "zcutlass_sm120_tensorop_f16_64x64x16");
  static const WmmaGemmOperation<__nv_bfloat16, 64, 128, DType::BF16, false> bf16_64x128(
      "zcutlass_sm120_tensorop_bf16_64x128x16");
  static const WmmaGemmOperation<__nv_bfloat16, 64, 64, DType::BF16, false> bf16_64x64(
      "zcutlass_sm120_tensorop_bf16_64x64x16");

  auto& manifest = gemm_api::global_manifest();
  manifest.append(&f16_64x128_aligned);
  manifest.append(&bf16_64x128_aligned);
  manifest.append(&f16_64x128);
  manifest.append(&f16_64x64);
  manifest.append(&bf16_64x128);
  manifest.append(&bf16_64x64);
  manifest.append(&f16_32x128);
  manifest.append(&f16_32x64);
  manifest.append(&bf16_32x128);
  manifest.append(&bf16_32x64);
  initialized = true;
}

const char* select_builtin_kernel_name(const GemmDesc& desc) {
  ensure_builtin_operations_registered();
  const gemm_api::GemmArguments args = make_arguments(
      desc, reinterpret_cast<const void*>(1), reinterpret_cast<const void*>(1),
      desc.beta != 0.0f ? reinterpret_cast<const void*>(1) : nullptr,
      reinterpret_cast<void*>(1));
  const auto* operation = gemm_api::global_manifest().find_best(args, {});
  if (operation != nullptr) {
    return operation->description().name;
  }
  return "none";
}

template <typename T>
Status dispatch_typed(const GemmDesc& desc, const void* A, const void* B, const void* C, void* D) {
  if (desc.n >= 128) {
    return launch_wmma<T, 64, 128>(desc, A, B, C, D);
  }
  return launch_wmma<T, 64, 64>(desc, A, B, C, D);
}

}  // namespace

Status gemm(const GemmDesc& desc,
            const void* A,
            const void* B,
            const void* C,
            void* D) {
  const Status validation = validate_desc(desc, A, B, C, D);
  if (validation != Status::Success) {
    return validation;
  }
  if (desc.m == 0 || desc.n == 0 || desc.k == 0) {
    return Status::Success;
  }

  ensure_builtin_operations_registered();
  const gemm_api::GemmArguments args = make_arguments(desc, A, B, C, D);
  const gemm_api::GemmOperation* operation = gemm_api::global_manifest().find_best(args, {});
  if (operation == nullptr) {
    return Status::NotSupported;
  }
  return operation->run(args, {});
}

const char* selected_kernel_name(const GemmDesc& desc) {
  const Status validation = validate_desc(desc, reinterpret_cast<const void*>(1),
                                          reinterpret_cast<const void*>(1),
                                          desc.beta != 0.0f ? reinterpret_cast<const void*>(1) : nullptr,
                                          reinterpret_cast<const void*>(1));
  return validation == Status::Success ? select_builtin_kernel_name(desc) : "none";
}

int registered_gemm_operation_count() {
  ensure_builtin_operations_registered();
  return gemm_api::global_manifest().size();
}

Status gemm(const gemm_api::GemmArguments& args) {
  GemmDesc desc{args.problem.m,
                args.problem.n,
                args.problem.k,
                args.A.ld,
                args.B.ld,
                args.C.ld,
                args.D.ld,
                args.A.dtype,
                args.B.dtype,
                args.C.dtype,
                args.D.dtype,
                args.alpha,
                args.beta,
                args.bias,
                args.stream,
                args.A.layout,
                args.B.layout,
                args.C.layout,
                args.D.layout};
  return gemm(desc, args.A.data, args.B.data, args.C.data, args.D.data);
}

Status can_implement(const gemm_api::GemmArguments& args) {
  GemmDesc desc{args.problem.m,
                args.problem.n,
                args.problem.k,
                args.A.ld,
                args.B.ld,
                args.C.ld,
                args.D.ld,
                args.A.dtype,
                args.B.dtype,
                args.C.dtype,
                args.D.dtype,
                args.alpha,
                args.beta,
                args.bias,
                args.stream,
                args.A.layout,
                args.B.layout,
                args.C.layout,
                args.D.layout};
  return validate_desc(desc, args.A.data, args.B.data, args.C.data, args.D.data);
}

size_t get_workspace_size(const gemm_api::GemmArguments& args) {
  (void)args;
  return 0;
}

int version_major() {
  return 0;
}

int version_minor() {
  return 1;
}

int version_patch() {
  return 0;
}

}  // namespace zcutlass
