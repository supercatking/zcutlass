#pragma once

#include <cstdint>

#include "zcutlass/arch/arch.hpp"
#include "zcutlass/gemm/gemm.hpp"
#include "zcutlass/layout/layout.hpp"
#include "zcutlass/numeric_types.hpp"
#include "zcutlass/status.hpp"

namespace zcutlass::gemm_api {

struct GemmPreference {
  int alignment_a = 1;
  int alignment_b = 1;
  int split_k = 1;
};

struct GemmOperationDescription {
  const char* name = "";
  DType a_type = DType::F16;
  DType b_type = DType::F16;
  DType c_type = DType::F16;
  DType d_type = DType::F16;
  DType accum_type = DType::F32;
  layout::LayoutKind a_layout = layout::LayoutKind::RowMajor;
  layout::LayoutKind b_layout = layout::LayoutKind::RowMajor;
  layout::LayoutKind c_layout = layout::LayoutKind::RowMajor;
  layout::LayoutKind d_layout = layout::LayoutKind::RowMajor;
  arch::ArchKind min_arch = arch::ArchKind::Sm80;
  arch::OpClass op_class = arch::OpClass::TensorOp;
  int tile_m = 0;
  int tile_n = 0;
  int tile_k = 0;
};

class GemmOperation {
 public:
  virtual ~GemmOperation() = default;
  virtual const GemmOperationDescription& description() const = 0;
  virtual bool can_implement(const GemmArguments& args,
                             const GemmPreference& pref) const = 0;
  virtual Status run(const GemmArguments& args,
                     const GemmPreference& pref) const = 0;
};

}  // namespace zcutlass::gemm_api

