#include "gemm_sm120_mma_prefill.cuh"

#include "zcutlass/gemm.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdlib>
#include <cstring>

namespace zcutlass {
namespace {

constexpr int kBlockM = gemm_sm120::Sm120MmaPrefill64x128x64Config::kBlockM;
constexpr int kBlockN = gemm_sm120::Sm120MmaPrefill64x128x64Config::kBlockN;
constexpr int kBlockK = gemm_sm120::Sm120MmaPrefill64x128x64Config::kBlockK;
constexpr int kWarpSize = 32;
constexpr int kWarpTileM = 16;
constexpr int kWarpTileN = 16;
constexpr int kWarpTilesM = kBlockM / kWarpTileM;
constexpr int kWarpTilesN = kBlockN / kWarpTileN;
constexpr int kWarpsPerBlock = kWarpTilesM * kWarpTilesN;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kWarpSize;
constexpr int kWarpTileN32 = 32;
constexpr int kWarpTilesN32 = kBlockN / kWarpTileN32;
constexpr int kWarpsPerBlockN32 = kWarpTilesM * kWarpTilesN32;
constexpr int kThreadsPerBlockN32 = kWarpsPerBlockN32 * kWarpSize;
constexpr int kWarpTileN64 = 64;
constexpr int kWarpTilesN64 = kBlockN / kWarpTileN64;
constexpr int kWarpsPerBlockN64 = kWarpTilesM * kWarpTilesN64;
constexpr int kThreadsPerBlockN64 = kWarpsPerBlockN64 * kWarpSize;

static_assert(kThreadsPerBlock == 1024, "The prototype uses one warp per 16x16 output tile");
static_assert(kThreadsPerBlockN32 == 512, "The warp16x32 prototype uses 16 warps per CTA");
static_assert(kThreadsPerBlockN64 == 256, "The warp16x64 prototype uses 8 warps per CTA");

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

__device__ unsigned short storage_bits(half value) {
  return __half_raw(value).x;
}

__device__ unsigned short storage_bits(__nv_bfloat16 value) {
  return __nv_bfloat16_raw(value).x;
}

template <typename T>
__device__ unsigned pack_pair(T lo, T hi) {
  return static_cast<unsigned>(storage_bits(lo)) |
         (static_cast<unsigned>(storage_bits(hi)) << 16);
}

template <typename T>
__device__ void mma_m16n8k16(float acc[4], const unsigned a[4], const unsigned b[2]);

template <>
__device__ void mma_m16n8k16<half>(float acc[4], const unsigned a[4], const unsigned b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <>
__device__ void mma_m16n8k16<__nv_bfloat16>(float acc[4],
                                            const unsigned a[4],
                                            const unsigned b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ int mma_a_row(int lane_group, int element) {
  return ((element >= 2 && element < 4) || element >= 6) ? lane_group + 8 : lane_group;
}

__device__ int mma_a_col(int lane_in_group, int element) {
  return lane_in_group * 2 + (element & 1) + (element >= 4 ? 8 : 0);
}

__device__ int mma_b_row(int lane_in_group, int element) {
  return lane_in_group * 2 + (element & 1) + (element >= 2 ? 8 : 0);
}

__device__ int mma_c_row(int lane_group, int element) {
  return element >= 2 ? lane_group + 8 : lane_group;
}

__device__ int mma_c_col(int lane_in_group, int element) {
  return lane_in_group * 2 + (element & 1);
}

template <typename T>
__device__ void load_a_fragment(const T* __restrict__ A,
                                const GemmDesc& desc,
                                int64_t tile_m,
                                int64_t tile_k,
                                int lane_group,
                                int lane_in_group,
                                unsigned a[4]) {
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    const int e0 = reg * 2;
    const int e1 = e0 + 1;
    const int row0 = mma_a_row(lane_group, e0);
    const int col0 = mma_a_col(lane_in_group, e0);
    const int row1 = mma_a_row(lane_group, e1);
    const int col1 = mma_a_col(lane_in_group, e1);
    const T lo = A[(tile_m + row0) * desc.lda + tile_k + col0];
    const T hi = A[(tile_m + row1) * desc.lda + tile_k + col1];
    a[reg] = pack_pair(lo, hi);
  }
}

template <typename T>
__device__ void load_a_fragment_smem(const T* __restrict__ a_tile,
                                     int warp_m,
                                     int k_stage,
                                     int lane_group,
                                     int lane_in_group,
                                     unsigned a[4]) {
#pragma unroll
  for (int reg = 0; reg < 4; ++reg) {
    const int e0 = reg * 2;
    const int e1 = e0 + 1;
    const int row0 = warp_m * kWarpTileM + mma_a_row(lane_group, e0);
    const int col0 = k_stage + mma_a_col(lane_in_group, e0);
    const int row1 = warp_m * kWarpTileM + mma_a_row(lane_group, e1);
    const int col1 = k_stage + mma_a_col(lane_in_group, e1);
    a[reg] = pack_pair(a_tile[row0 * kBlockK + col0], a_tile[row1 * kBlockK + col1]);
  }
}

template <typename T>
__device__ void load_b_fragment(const T* __restrict__ B,
                                const GemmDesc& desc,
                                int64_t tile_n,
                                int64_t tile_k,
                                int n_offset,
                                int lane_group,
                                int lane_in_group,
                                unsigned b[2]) {
#pragma unroll
  for (int reg = 0; reg < 2; ++reg) {
    const int e0 = reg * 2;
    const int e1 = e0 + 1;
    const int row0 = mma_b_row(lane_in_group, e0);
    const int row1 = mma_b_row(lane_in_group, e1);
    const int col = lane_group;
    const T lo = B[(tile_k + row0) * desc.ldb + tile_n + n_offset + col];
    const T hi = B[(tile_k + row1) * desc.ldb + tile_n + n_offset + col];
    b[reg] = pack_pair(lo, hi);
  }
}

template <typename T>
__device__ void load_b_fragment_smem(const T* __restrict__ b_tile,
                                     int warp_n,
                                     int k_stage,
                                     int n_offset,
                                     int lane_group,
                                     int lane_in_group,
                                     unsigned b[2]) {
#pragma unroll
  for (int reg = 0; reg < 2; ++reg) {
    const int e0 = reg * 2;
    const int e1 = e0 + 1;
    const int row0 = k_stage + mma_b_row(lane_in_group, e0);
    const int row1 = k_stage + mma_b_row(lane_in_group, e1);
    const int col = warp_n * kWarpTileN + n_offset + lane_group;
    b[reg] = pack_pair(b_tile[row0 * kBlockN + col], b_tile[row1 * kBlockN + col]);
  }
}

template <typename T>
__device__ void load_b_fragment_smem_base(const T* __restrict__ b_tile,
                                          int n_base,
                                          int k_stage,
                                          int n_offset,
                                          int lane_group,
                                          int lane_in_group,
                                          unsigned b[2]) {
#pragma unroll
  for (int reg = 0; reg < 2; ++reg) {
    const int e0 = reg * 2;
    const int e1 = e0 + 1;
    const int row0 = k_stage + mma_b_row(lane_in_group, e0);
    const int row1 = k_stage + mma_b_row(lane_in_group, e1);
    const int col = n_base + n_offset + lane_group;
    b[reg] = pack_pair(b_tile[row0 * kBlockN + col], b_tile[row1 * kBlockN + col]);
  }
}

template <typename T>
__device__ void store_accumulator(T* __restrict__ D,
                                  const GemmDesc& desc,
                                  int64_t tile_m,
                                  int64_t tile_n,
                                  int n_offset,
                                  int lane_group,
                                  int lane_in_group,
                                  const float acc[4]) {
#pragma unroll
  for (int element = 0; element < 4; ++element) {
    const int row = mma_c_row(lane_group, element);
    const int col = mma_c_col(lane_in_group, element);
    D[(tile_m + row) * desc.ldd + tile_n + n_offset + col] = from_float<T>(acc[element]);
  }
}

template <typename T>
__global__ __launch_bounds__(kThreadsPerBlock, 1) void sm120_mma_prefill_64x128x64_kernel(
    GemmDesc desc,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ D) {
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int lane_group = lane_id >> 2;
  const int lane_in_group = lane_id & 3;
  const int warp_m = warp_id / kWarpTilesN;
  const int warp_n = warp_id % kWarpTilesN;

  const int64_t tile_m = static_cast<int64_t>(blockIdx.y) * kBlockM + warp_m * kWarpTileM;
  const int64_t tile_n = static_cast<int64_t>(blockIdx.x) * kBlockN + warp_n * kWarpTileN;

  float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int64_t k0 = 0; k0 < desc.k; k0 += 16) {
    unsigned a[4];
    unsigned b0[2];
    unsigned b1[2];
    load_a_fragment(A, desc, tile_m, k0, lane_group, lane_in_group, a);
    load_b_fragment(B, desc, tile_n, k0, 0, lane_group, lane_in_group, b0);
    load_b_fragment(B, desc, tile_n, k0, 8, lane_group, lane_in_group, b1);
    mma_m16n8k16<T>(acc0, a, b0);
    mma_m16n8k16<T>(acc1, a, b1);
  }

  store_accumulator(D, desc, tile_m, tile_n, 0, lane_group, lane_in_group, acc0);
  store_accumulator(D, desc, tile_m, tile_n, 8, lane_group, lane_in_group, acc1);
}

template <typename T>
__global__ __launch_bounds__(kThreadsPerBlock, 1) void sm120_mma_prefill_smem_64x128x64_kernel(
    GemmDesc desc,
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ D) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  T* a_tile = reinterpret_cast<T*>(shared_storage);
  T* b_tile = a_tile + kBlockM * kBlockK;

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int lane_group = lane_id >> 2;
  const int lane_in_group = lane_id & 3;
  const int warp_m = warp_id / kWarpTilesN;
  const int warp_n = warp_id % kWarpTilesN;

  const int64_t block_m = static_cast<int64_t>(blockIdx.y) * kBlockM;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) * kBlockN;
  const int64_t tile_m = block_m + warp_m * kWarpTileM;
  const int64_t tile_n = block_n + warp_n * kWarpTileN;

  float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int64_t k0 = 0; k0 < desc.k; k0 += kBlockK) {
    for (int idx = tid; idx < kBlockM * kBlockK; idx += blockDim.x) {
      const int row = idx / kBlockK;
      const int col = idx % kBlockK;
      a_tile[idx] = A[(block_m + row) * desc.lda + k0 + col];
    }
    for (int idx = tid; idx < kBlockK * kBlockN; idx += blockDim.x) {
      const int row = idx / kBlockN;
      const int col = idx % kBlockN;
      b_tile[idx] = B[(k0 + row) * desc.ldb + block_n + col];
    }
    __syncthreads();

#pragma unroll
    for (int k_stage = 0; k_stage < kBlockK; k_stage += 16) {
      unsigned a[4];
      unsigned b0[2];
      unsigned b1[2];
      load_a_fragment_smem(a_tile, warp_m, k_stage, lane_group, lane_in_group, a);
      load_b_fragment_smem(b_tile, warp_n, k_stage, 0, lane_group, lane_in_group, b0);
      load_b_fragment_smem(b_tile, warp_n, k_stage, 8, lane_group, lane_in_group, b1);
      mma_m16n8k16<T>(acc0, a, b0);
      mma_m16n8k16<T>(acc1, a, b1);
    }
    __syncthreads();
  }

  store_accumulator(D, desc, tile_m, tile_n, 0, lane_group, lane_in_group, acc0);
  store_accumulator(D, desc, tile_m, tile_n, 8, lane_group, lane_in_group, acc1);
}

template <typename T>
__global__ __launch_bounds__(kThreadsPerBlockN32, 1)
void sm120_mma_prefill_smem_warp16x32_64x128x64_kernel(GemmDesc desc,
                                                       const T* __restrict__ A,
                                                       const T* __restrict__ B,
                                                       T* __restrict__ D) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  T* a_tile = reinterpret_cast<T*>(shared_storage);
  T* b_tile = a_tile + kBlockM * kBlockK;

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int lane_group = lane_id >> 2;
  const int lane_in_group = lane_id & 3;
  const int warp_m = warp_id / kWarpTilesN32;
  const int warp_n = warp_id % kWarpTilesN32;
  const int warp_n_base = warp_n * kWarpTileN32;

  const int64_t block_m = static_cast<int64_t>(blockIdx.y) * kBlockM;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) * kBlockN;
  const int64_t tile_m = block_m + warp_m * kWarpTileM;
  const int64_t tile_n = block_n + warp_n_base;

  float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc3[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int64_t k0 = 0; k0 < desc.k; k0 += kBlockK) {
    for (int idx = tid; idx < kBlockM * kBlockK; idx += blockDim.x) {
      const int row = idx / kBlockK;
      const int col = idx % kBlockK;
      a_tile[idx] = A[(block_m + row) * desc.lda + k0 + col];
    }
    for (int idx = tid; idx < kBlockK * kBlockN; idx += blockDim.x) {
      const int row = idx / kBlockN;
      const int col = idx % kBlockN;
      b_tile[idx] = B[(k0 + row) * desc.ldb + block_n + col];
    }
    __syncthreads();

#pragma unroll
    for (int k_stage = 0; k_stage < kBlockK; k_stage += 16) {
      unsigned a[4];
      unsigned b[2];
      load_a_fragment_smem(a_tile, warp_m, k_stage, lane_group, lane_in_group, a);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 0, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc0, a, b);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 8, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc1, a, b);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 16, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc2, a, b);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 24, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc3, a, b);
    }
    __syncthreads();
  }

  store_accumulator(D, desc, tile_m, tile_n, 0, lane_group, lane_in_group, acc0);
  store_accumulator(D, desc, tile_m, tile_n, 8, lane_group, lane_in_group, acc1);
  store_accumulator(D, desc, tile_m, tile_n, 16, lane_group, lane_in_group, acc2);
  store_accumulator(D, desc, tile_m, tile_n, 24, lane_group, lane_in_group, acc3);
}

template <typename T>
__global__ __launch_bounds__(kThreadsPerBlockN64, 1)
void sm120_mma_prefill_smem_warp16x64_64x128x64_kernel(GemmDesc desc,
                                                       const T* __restrict__ A,
                                                       const T* __restrict__ B,
                                                       T* __restrict__ D) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  T* a_tile = reinterpret_cast<T*>(shared_storage);
  T* b_tile = a_tile + kBlockM * kBlockK;

  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const int lane_group = lane_id >> 2;
  const int lane_in_group = lane_id & 3;
  const int warp_m = warp_id / kWarpTilesN64;
  const int warp_n = warp_id % kWarpTilesN64;
  const int warp_n_base = warp_n * kWarpTileN64;

  const int64_t block_m = static_cast<int64_t>(blockIdx.y) * kBlockM;
  const int64_t block_n = static_cast<int64_t>(blockIdx.x) * kBlockN;
  const int64_t tile_m = block_m + warp_m * kWarpTileM;
  const int64_t tile_n = block_n + warp_n_base;

  float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc3[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc5[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc6[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float acc7[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int64_t k0 = 0; k0 < desc.k; k0 += kBlockK) {
    for (int idx = tid; idx < kBlockM * kBlockK; idx += blockDim.x) {
      const int row = idx / kBlockK;
      const int col = idx % kBlockK;
      a_tile[idx] = A[(block_m + row) * desc.lda + k0 + col];
    }
    for (int idx = tid; idx < kBlockK * kBlockN; idx += blockDim.x) {
      const int row = idx / kBlockN;
      const int col = idx % kBlockN;
      b_tile[idx] = B[(k0 + row) * desc.ldb + block_n + col];
    }
    __syncthreads();

#pragma unroll
    for (int k_stage = 0; k_stage < kBlockK; k_stage += 16) {
      unsigned a[4];
      unsigned b[2];
      load_a_fragment_smem(a_tile, warp_m, k_stage, lane_group, lane_in_group, a);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 0, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc0, a, b);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 8, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc1, a, b);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 16, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc2, a, b);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 24, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc3, a, b);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 32, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc4, a, b);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 40, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc5, a, b);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 48, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc6, a, b);
      load_b_fragment_smem_base(b_tile, warp_n_base, k_stage, 56, lane_group, lane_in_group, b);
      mma_m16n8k16<T>(acc7, a, b);
    }
    __syncthreads();
  }

  store_accumulator(D, desc, tile_m, tile_n, 0, lane_group, lane_in_group, acc0);
  store_accumulator(D, desc, tile_m, tile_n, 8, lane_group, lane_in_group, acc1);
  store_accumulator(D, desc, tile_m, tile_n, 16, lane_group, lane_in_group, acc2);
  store_accumulator(D, desc, tile_m, tile_n, 24, lane_group, lane_in_group, acc3);
  store_accumulator(D, desc, tile_m, tile_n, 32, lane_group, lane_in_group, acc4);
  store_accumulator(D, desc, tile_m, tile_n, 40, lane_group, lane_in_group, acc5);
  store_accumulator(D, desc, tile_m, tile_n, 48, lane_group, lane_in_group, acc6);
  store_accumulator(D, desc, tile_m, tile_n, 56, lane_group, lane_in_group, acc7);
}

template <typename T>
Status launch_sm120_mma_prefill(const GemmDesc& desc,
                                const void* A,
                                const void* B,
                                void* D) {
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(desc.n / kBlockN),
                  static_cast<unsigned int>(desc.m / kBlockM));
  sm120_mma_prefill_64x128x64_kernel<T><<<grid, block, 0, desc.stream>>>(
      desc, static_cast<const T*>(A), static_cast<const T*>(B), static_cast<T*>(D));
  return cudaGetLastError() == cudaSuccess ? Status::Success : Status::RuntimeError;
}

template <typename T>
Status launch_sm120_mma_prefill_smem(const GemmDesc& desc,
                                     const void* A,
                                     const void* B,
                                     void* D) {
  const dim3 block(kThreadsPerBlock);
  const dim3 grid(static_cast<unsigned int>(desc.n / kBlockN),
                  static_cast<unsigned int>(desc.m / kBlockM));
  const size_t shared_bytes = (kBlockM * kBlockK + kBlockK * kBlockN) * sizeof(T);
  auto kernel = sm120_mma_prefill_smem_64x128x64_kernel<T>;
  const cudaError_t attr_status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(shared_bytes));
  if (attr_status != cudaSuccess && attr_status != cudaErrorInvalidValue) {
    return Status::RuntimeError;
  }
  kernel<<<grid, block, shared_bytes, desc.stream>>>(
      desc, static_cast<const T*>(A), static_cast<const T*>(B), static_cast<T*>(D));
  return cudaGetLastError() == cudaSuccess ? Status::Success : Status::RuntimeError;
}

template <typename T>
Status launch_sm120_mma_prefill_smem_warp16x32(const GemmDesc& desc,
                                               const void* A,
                                               const void* B,
                                               void* D) {
  const dim3 block(kThreadsPerBlockN32);
  const dim3 grid(static_cast<unsigned int>(desc.n / kBlockN),
                  static_cast<unsigned int>(desc.m / kBlockM));
  const size_t shared_bytes = (kBlockM * kBlockK + kBlockK * kBlockN) * sizeof(T);
  auto kernel = sm120_mma_prefill_smem_warp16x32_64x128x64_kernel<T>;
  const cudaError_t attr_status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(shared_bytes));
  if (attr_status != cudaSuccess && attr_status != cudaErrorInvalidValue) {
    return Status::RuntimeError;
  }
  kernel<<<grid, block, shared_bytes, desc.stream>>>(
      desc, static_cast<const T*>(A), static_cast<const T*>(B), static_cast<T*>(D));
  return cudaGetLastError() == cudaSuccess ? Status::Success : Status::RuntimeError;
}

template <typename T>
Status launch_sm120_mma_prefill_smem_warp16x64(const GemmDesc& desc,
                                               const void* A,
                                               const void* B,
                                               void* D) {
  const dim3 block(kThreadsPerBlockN64);
  const dim3 grid(static_cast<unsigned int>(desc.n / kBlockN),
                  static_cast<unsigned int>(desc.m / kBlockM));
  const size_t shared_bytes = (kBlockM * kBlockK + kBlockK * kBlockN) * sizeof(T);
  auto kernel = sm120_mma_prefill_smem_warp16x64_64x128x64_kernel<T>;
  const cudaError_t attr_status =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(shared_bytes));
  if (attr_status != cudaSuccess && attr_status != cudaErrorInvalidValue) {
    return Status::RuntimeError;
  }
  kernel<<<grid, block, shared_bytes, desc.stream>>>(
      desc, static_cast<const T*>(A), static_cast<const T*>(B), static_cast<T*>(D));
  return cudaGetLastError() == cudaSuccess ? Status::Success : Status::RuntimeError;
}

bool experimental_kernels_enabled() {
  const char* value = std::getenv("ZCUTLASS_EXPERIMENTAL_KERNELS");
  return value != nullptr && std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "FALSE") != 0;
}

bool experimental_kernel_filter_matches(const char* name) {
  const char* filter = std::getenv("ZCUTLASS_EXPERIMENTAL_KERNEL");
  return filter == nullptr || filter[0] == '\0' || std::strstr(name, filter) != nullptr;
}

enum class Sm120MmaPrefillVariant {
  Direct,
  Shared16x16,
  Shared16x32,
  Shared16x64,
};

template <typename T, DType Type, Sm120MmaPrefillVariant Variant>
class Sm120MmaPrefillOperation final : public gemm_api::GemmOperation {
 public:
  constexpr explicit Sm120MmaPrefillOperation(const char* name)
      : description_{name,
                     Type,
                     Type,
                     Type,
                     Type,
                     DType::F32,
                     layout::LayoutKind::RowMajor,
                     layout::LayoutKind::RowMajor,
                     layout::LayoutKind::RowMajor,
                     layout::LayoutKind::RowMajor,
                     arch::ArchKind::Sm120,
                     arch::OpClass::TensorOp,
                     kBlockM,
                     kBlockN,
                     kBlockK,
                     gemm_api::ShapeFamily::Prefill,
                     true,
                     false,
                     false,
                     true,
                     gemm_sm120::Sm120MmaPrefill64x128x64Config::kPipelineStages,
                     gemm_api::EpilogueKind::RegisterLinear} {}

  const gemm_api::GemmOperationDescription& description() const override {
    return description_;
  }

  bool can_implement(const gemm_api::GemmArguments& args,
                     const gemm_api::GemmPreference& pref) const override {
    (void)pref;
    if (!experimental_kernels_enabled() || !experimental_kernel_filter_matches(description_.name)) {
      return false;
    }
    if (args.A.dtype != Type || args.B.dtype != Type || args.C.dtype != Type ||
        args.D.dtype != Type) {
      return false;
    }
    if (args.A.layout != layout::LayoutKind::RowMajor ||
        args.B.layout != layout::LayoutKind::RowMajor ||
        args.D.layout != layout::LayoutKind::RowMajor) {
      return false;
    }
    if (args.problem.m % kBlockM != 0 || args.problem.n % kBlockN != 0 ||
        args.problem.k % kBlockK != 0) {
      return false;
    }
    if (args.problem.m < 32 || args.problem.m > 256 || args.problem.n < 1024 ||
        args.problem.k < 1024) {
      return false;
    }
    if (args.alpha != 1.0f || args.beta != 0.0f || args.bias != nullptr) {
      return false;
    }
    if (args.A.ld != args.problem.k || args.B.ld != args.problem.n ||
        args.D.ld != args.problem.n) {
      return false;
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
    if constexpr (Variant == Sm120MmaPrefillVariant::Shared16x64) {
      return launch_sm120_mma_prefill_smem_warp16x64<T>(
          desc, args.A.data, args.B.data, args.D.data);
    } else if constexpr (Variant == Sm120MmaPrefillVariant::Shared16x32) {
      return launch_sm120_mma_prefill_smem_warp16x32<T>(
          desc, args.A.data, args.B.data, args.D.data);
    } else if constexpr (Variant == Sm120MmaPrefillVariant::Shared16x16) {
      return launch_sm120_mma_prefill_smem<T>(desc, args.A.data, args.B.data, args.D.data);
    } else {
      return launch_sm120_mma_prefill<T>(desc, args.A.data, args.B.data, args.D.data);
    }
  }

 private:
  gemm_api::GemmOperationDescription description_;
};

}  // namespace
}  // namespace zcutlass

namespace zcutlass::gemm_api {

void append_sm120_mma_prefill_operations(Manifest& manifest) {
  static const Sm120MmaPrefillOperation<half, DType::F16, Sm120MmaPrefillVariant::Shared16x64>
      f16_prefill_smem_warp16x64(
          "zcutlass_sm120_mma_f16_64x128x64_prefill_smem_warp16x64_reg_epilogue");
  static const Sm120MmaPrefillOperation<__nv_bfloat16,
                                        DType::BF16,
                                        Sm120MmaPrefillVariant::Shared16x64>
      bf16_prefill_smem_warp16x64(
          "zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_warp16x64_reg_epilogue");
  static const Sm120MmaPrefillOperation<half, DType::F16, Sm120MmaPrefillVariant::Shared16x32>
      f16_prefill_smem_warp16x32(
          "zcutlass_sm120_mma_f16_64x128x64_prefill_smem_warp16x32_reg_epilogue");
  static const Sm120MmaPrefillOperation<__nv_bfloat16,
                                        DType::BF16,
                                        Sm120MmaPrefillVariant::Shared16x32>
      bf16_prefill_smem_warp16x32(
          "zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_warp16x32_reg_epilogue");
  static const Sm120MmaPrefillOperation<half, DType::F16, Sm120MmaPrefillVariant::Shared16x16>
      f16_prefill_smem(
      "zcutlass_sm120_mma_f16_64x128x64_prefill_smem_reg_epilogue");
  static const Sm120MmaPrefillOperation<__nv_bfloat16,
                                        DType::BF16,
                                        Sm120MmaPrefillVariant::Shared16x16>
      bf16_prefill_smem(
      "zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_reg_epilogue");
  static const Sm120MmaPrefillOperation<half, DType::F16, Sm120MmaPrefillVariant::Direct>
      f16_prefill(
      "zcutlass_sm120_mma_f16_64x128x64_prefill_reg_epilogue");
  static const Sm120MmaPrefillOperation<__nv_bfloat16, DType::BF16, Sm120MmaPrefillVariant::Direct>
      bf16_prefill(
      "zcutlass_sm120_mma_bf16_64x128x64_prefill_reg_epilogue");

  manifest.append(&f16_prefill_smem_warp16x64);
  manifest.append(&bf16_prefill_smem_warp16x64);
  manifest.append(&f16_prefill_smem_warp16x32);
  manifest.append(&bf16_prefill_smem_warp16x32);
  manifest.append(&f16_prefill_smem);
  manifest.append(&bf16_prefill_smem);
  manifest.append(&f16_prefill);
  manifest.append(&bf16_prefill);
}

}  // namespace zcutlass::gemm_api
