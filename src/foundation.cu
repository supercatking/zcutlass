#include "zcutlass/arch/arch.hpp"
#include "zcutlass/gemm/manifest.hpp"
#include "zcutlass/numeric_types.hpp"
#include "zcutlass/status.hpp"

#include <cuda_runtime_api.h>

namespace zcutlass {

const char* status_to_string(Status status) {
  switch (status) {
    case Status::Success:
      return "Success";
    case Status::InvalidArgument:
      return "InvalidArgument";
    case Status::NotSupported:
      return "NotSupported";
    case Status::RuntimeError:
      return "RuntimeError";
  }
  return "Unknown";
}

const char* dtype_name(DType dtype) {
  switch (dtype) {
    case DType::F16:
      return "f16";
    case DType::BF16:
      return "bf16";
    case DType::F32:
      return "f32";
    case DType::TF32:
      return "tf32";
    case DType::E4M3:
      return "e4m3";
    case DType::E5M2:
      return "e5m2";
    case DType::E2M1:
      return "e2m1";
  }
  return "unknown";
}

int dtype_bits(DType dtype) {
  switch (dtype) {
    case DType::F16:
    case DType::BF16:
      return 16;
    case DType::F32:
    case DType::TF32:
      return 32;
    case DType::E4M3:
    case DType::E5M2:
      return 8;
    case DType::E2M1:
      return 4;
  }
  return 0;
}

bool is_supported_gemm_storage_dtype(DType dtype) {
  return dtype == DType::F16 || dtype == DType::BF16;
}

}  // namespace zcutlass

namespace zcutlass::arch {

const char* arch_name(ArchKind arch) {
  switch (arch) {
    case ArchKind::Sm80:
      return "sm80";
    case ArchKind::Sm90:
      return "sm90";
    case ArchKind::Sm100:
      return "sm100";
    case ArchKind::Sm120:
      return "sm120";
  }
  return "unknown";
}

const char* op_class_name(OpClass op_class) {
  switch (op_class) {
    case OpClass::Simt:
      return "simt";
    case OpClass::TensorOp:
      return "tensorop";
    case OpClass::Wgmma:
      return "wgmma";
    case OpClass::Tcgen05:
      return "tcgen05";
  }
  return "unknown";
}

ComputeCapability current_device_cc() {
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    return {};
  }
  return {prop.major, prop.minor};
}

}  // namespace zcutlass::arch

namespace zcutlass::gemm_api {

const char* shape_family_name(ShapeFamily family) {
  switch (family) {
    case ShapeFamily::Decode:
      return "decode";
    case ShapeFamily::Prefill:
      return "prefill";
    case ShapeFamily::Large:
      return "large";
    case ShapeFamily::Fallback:
      return "fallback";
  }
  return "unknown";
}

const char* epilogue_kind_name(EpilogueKind kind) {
  switch (kind) {
    case EpilogueKind::SharedAccumulator:
      return "shared_accumulator";
    case EpilogueKind::RegisterLinear:
      return "register_linear";
  }
  return "unknown";
}

void Manifest::append(const GemmOperation* operation) {
  if (operation != nullptr && size_ < kMaxOperations) {
    operations_[size_++] = operation;
  }
}

const GemmOperation* Manifest::find_best(const GemmArguments& args,
                                         const GemmPreference& pref) const {
  for (int i = 0; i < size_; ++i) {
    if (operations_[i]->can_implement(args, pref)) {
      return operations_[i];
    }
  }
  return nullptr;
}

int Manifest::size() const {
  return size_;
}

const GemmOperation* Manifest::at(int index) const {
  return index >= 0 && index < size_ ? operations_[index] : nullptr;
}

Manifest& global_manifest() {
  static Manifest manifest;
  return manifest;
}

void initialize_builtin_operations() {}

}  // namespace zcutlass::gemm_api
