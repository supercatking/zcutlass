#include "gemm_sm120_mma_prefill.cuh"

namespace zcutlass::gemm_api {

void append_sm120_mma_prefill_operations(Manifest& manifest) {
  (void)manifest;
  // The explicit-MMA prefill operation is registered only after the first
  // register-epilogue kernel passes correctness, benchmark, and Nsight gates.
}

}  // namespace zcutlass::gemm_api
