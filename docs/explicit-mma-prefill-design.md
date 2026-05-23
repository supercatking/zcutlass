# Explicit-MMA Prefill Design Notes

## Purpose

The next zcutlass kernel generation must move beyond WMMA. Current WMMA
prefill kernels keep FP32 accumulation, but WMMA accumulator fragments cannot be
directly stored as FP16/BF16 outputs. The implementation therefore spills FP32
accumulators through shared memory before conversion. This is now a structural
performance ceiling.

The explicit-MMA prefill family exists to keep accumulation and epilogue in
registers and expose a CUTLASS-like separation between tile shape, mainloop
schedule, and epilogue policy.

## First Prototype Scope

Target one aligned dense path only:

- Shape family: prefill.
- Primary shape: `M=128,N=4096,K=4096`.
- DTypes: FP16 and BF16 storage, FP32 accumulation.
- Layout: row-major A/B/D.
- Epilogue: `D = A * B`, no beta, no bias.
- Fallback: all unsupported/ragged/beta/bias/padded paths remain on existing
  WMMA kernels.

The first implementation should not try to solve decode, large throughput,
grouped GEMM, FP8, bias fusion, or arbitrary layouts.

## Proposed File Structure

Use a separate implementation slice so explicit-MMA work can be reviewed and
reverted independently from the WMMA fallback:

- `src/gemm_sm120_mma_prefill.cuh`: tile configs, PTX wrappers, fragment mapping,
  and register epilogue helpers.
- `src/gemm_sm120_mma_prefill.cu`: operation classes, launch wrappers, and
  manifest append function.
- `src/gemm.cu`: only calls the explicit-MMA manifest append function before the
  current WMMA prefill entries.
- `CMakeLists.txt`: adds the new CUDA translation unit to the static library.

If helper duplication becomes a problem, introduce an internal dispatch helper
header after the first prototype, not before it.

## Kernel Interface

The manifest should expose the explicit-MMA kernel as a separate operation
family, for example:

- `zcutlass_sm120_mma_f16_64x128x64_prefill_reg_epilogue`
- `zcutlass_sm120_mma_bf16_64x128x64_prefill_reg_epilogue`

Required metadata:

- tile M/N/K.
- pipeline stage count.
- epilogue kind: `register_linear`.
- shape family: `prefill`.
- path: `fast` or `experimental_fast`.
- minimum arch: `sm120`.

The public `zcutlass::gemm` API should not change for M1.

Initial tile config:

```cpp
struct Sm120MmaPrefill64x128x64 {
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 128;
  static constexpr int kBlockK = 64;
  static constexpr int kWarpM = 32;
  static constexpr int kWarpN = 32;
  static constexpr int kWarps = 8;
  static constexpr int kStages = 1;
};
```

The first slice should use PTX `mma.sync.aligned.m16n8k16` forms and explicitly
validate the row-major B staging/`ldmatrix` layout, because incorrect lane
mapping can produce plausible but scrambled tiles.

## Implementation Direction

- Keep existing WMMA code intact as fallback and baseline.
- Add explicit-MMA code in a new CUDA translation unit or internal namespace so
  it can be compiled, benchmarked, and reverted independently.
- Start with a conservative tile that maps cleanly to current benchmark shapes:
  `64x128x64`.
- Use register accumulator fragments and convert directly to FP16/BF16 in the
  epilogue.
- Keep launch/dispatch selectable through the existing manifest.

## Evidence Required

Correctness:

- C++ tests against CPU/cuBLAS reference.
- `compute-sanitizer` on the test binary before promotion.
- Explicit dispatch tests proving unsupported paths stay on WMMA fallback.

Performance:

- `zcutlass_bench --m 128 --n 4096 --k 4096 --dtype f16`.
- `zcutlass_bench --m 128 --n 4096 --k 4096 --dtype bf16`.
- `zcutlass_bench --suite llm-v1.5 --dtype both`.
- Regression gate against the latest zcutlass WMMA baseline.

Profiling:

- Nsight Compute summary for FP16 and BF16 `128x4096x4096`.
- Record SM throughput, tensor active, DRAM throughput, occupancy,
  registers/thread, shared memory/block, and top stalls.
- `nvdisasm` or `cuobjdump` confirmation that the intended MMA path is present.

Promotion:

- M1 accepts a measured and explained prototype.
- M2 promotion requires prefill geomean `>=1.10x` over current zcutlass and
  at least `0.80x` of cuBLAS/CUTLASS.
