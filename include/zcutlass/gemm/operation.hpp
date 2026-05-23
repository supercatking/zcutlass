#pragma once

#include <cstdint>

#include "zcutlass/arch/arch.hpp"
#include "zcutlass/gemm/gemm.hpp"
#include "zcutlass/layout/layout.hpp"
#include "zcutlass/numeric_types.hpp"
#include "zcutlass/status.hpp"

namespace zcutlass::gemm_api {

enum class ShapeFamily {
  Decode,
  Prefill,
  Large,
  Fallback,
};

const char* shape_family_name(ShapeFamily family);

enum class EpilogueKind {
  SharedAccumulator,
  RegisterLinear,
};

const char* epilogue_kind_name(EpilogueKind kind);

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
  ShapeFamily family = ShapeFamily::Fallback;
  bool requires_aligned_tiles = false;
  bool supports_beta = true;
  bool supports_bias = true;
  bool experimental = false;
  int pipeline_stages = 1;
  EpilogueKind epilogue = EpilogueKind::SharedAccumulator;
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
