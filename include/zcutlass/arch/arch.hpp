#pragma once

namespace zcutlass::arch {

struct Sm80 {};
struct Sm90 {};
struct Sm100 {};
struct Sm120 {};

enum class ArchKind {
  Sm80,
  Sm90,
  Sm100,
  Sm120,
};

enum class OpClass {
  Simt,
  TensorOp,
  Wgmma,
  Tcgen05,
};

struct ComputeCapability {
  int major = 0;
  int minor = 0;
};

constexpr int arch_to_major(ArchKind arch) {
  switch (arch) {
    case ArchKind::Sm80:
      return 8;
    case ArchKind::Sm90:
      return 9;
    case ArchKind::Sm100:
      return 10;
    case ArchKind::Sm120:
      return 12;
  }
  return 0;
}

constexpr int arch_to_minor(ArchKind arch) {
  (void)arch;
  return 0;
}

inline bool supports(ComputeCapability cc, ArchKind arch) {
  const int major = arch_to_major(arch);
  const int minor = arch_to_minor(arch);
  return cc.major > major || (cc.major == major && cc.minor >= minor);
}

const char* arch_name(ArchKind arch);
const char* op_class_name(OpClass op_class);
ComputeCapability current_device_cc();

}  // namespace zcutlass::arch

