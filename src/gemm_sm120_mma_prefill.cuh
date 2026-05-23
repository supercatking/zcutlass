#pragma once

#include "zcutlass/gemm/manifest.hpp"

namespace zcutlass::gemm_sm120 {

struct Sm120MmaPrefill64x128x64Config {
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 128;
  static constexpr int kBlockK = 64;
  static constexpr int kWarpTileM = 16;
  static constexpr int kWarpTileN = 32;
  static constexpr int kSmemSkewA = 8;
  static constexpr int kSmemSkewB = 8;
  static constexpr int kAStride = kBlockK + kSmemSkewA;
  static constexpr int kBStride = kBlockN + kSmemSkewB;
  static constexpr int kPipelineStages = 2;
  static constexpr const char* kEpilogue = "register_linear";
};

}  // namespace zcutlass::gemm_sm120

namespace zcutlass::gemm_api {

void append_sm120_mma_prefill_operations(Manifest& manifest);

}  // namespace zcutlass::gemm_api
