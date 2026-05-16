#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

namespace zcutlass {

enum class DType {
  F16,
  BF16,
};

enum class Status {
  Success,
  InvalidArgument,
  NotSupported,
  RuntimeError,
};

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
};

Status gemm(const GemmDesc& desc,
            const void* A,
            const void* B,
            const void* C,
            void* D);

const char* status_to_string(Status status);

int version_major();
int version_minor();
int version_patch();

}  // namespace zcutlass

