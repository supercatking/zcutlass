#pragma once

#include <cstdint>
#include <cstddef>

#include <cuda_runtime_api.h>

#include "zcutlass/layout/layout.hpp"
#include "zcutlass/numeric_types.hpp"
#include "zcutlass/status.hpp"

namespace zcutlass::gemm_api {

struct GemmCoord {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
};

struct TensorRef {
  const void* data = nullptr;
  DType dtype = DType::F16;
  layout::LayoutKind layout = layout::LayoutKind::RowMajor;
  int64_t ld = 0;
};

struct TensorRefMutable {
  void* data = nullptr;
  DType dtype = DType::F16;
  layout::LayoutKind layout = layout::LayoutKind::RowMajor;
  int64_t ld = 0;
};

struct GemmArguments {
  GemmCoord problem;
  TensorRef A;
  TensorRef B;
  TensorRef C;
  TensorRefMutable D;
  float alpha = 1.0f;
  float beta = 0.0f;
  const void* bias = nullptr;
  cudaStream_t stream = nullptr;
};

}  // namespace zcutlass::gemm_api

namespace zcutlass {

struct GemmDesc {
  int64_t m;
  int64_t n;
  int64_t k;
  int64_t lda;
  int64_t ldb;
  int64_t ldc;
  int64_t ldd;
  DType a_type;
  DType b_type;
  DType c_type;
  DType d_type;
  float alpha;
  float beta;
  const void* bias;
  cudaStream_t stream;
  layout::LayoutKind a_layout = layout::LayoutKind::RowMajor;
  layout::LayoutKind b_layout = layout::LayoutKind::RowMajor;
  layout::LayoutKind c_layout = layout::LayoutKind::RowMajor;
  layout::LayoutKind d_layout = layout::LayoutKind::RowMajor;
};

Status gemm(const GemmDesc& desc,
            const void* A,
            const void* B,
            const void* C,
            void* D);

Status gemm(const gemm_api::GemmArguments& args);
Status can_implement(const gemm_api::GemmArguments& args);
size_t get_workspace_size(const gemm_api::GemmArguments& args);
const char* selected_kernel_name(const GemmDesc& desc);
int registered_gemm_operation_count();

}  // namespace zcutlass
