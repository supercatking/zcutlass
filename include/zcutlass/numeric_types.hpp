#pragma once

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace zcutlass {

enum class DType {
  F16,
  BF16,
  F32,
  TF32,
  E4M3,
  E5M2,
  E2M1,
};

template <DType>
struct DTypeTraits;

template <>
struct DTypeTraits<DType::F16> {
  using storage_type = half;
  static constexpr int bits = 16;
  static constexpr const char* name = "f16";
};

template <>
struct DTypeTraits<DType::BF16> {
  using storage_type = __nv_bfloat16;
  static constexpr int bits = 16;
  static constexpr const char* name = "bf16";
};

template <>
struct DTypeTraits<DType::F32> {
  using storage_type = float;
  static constexpr int bits = 32;
  static constexpr const char* name = "f32";
};

const char* dtype_name(DType dtype);
int dtype_bits(DType dtype);
bool is_supported_gemm_storage_dtype(DType dtype);

}  // namespace zcutlass

