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

## Scaffold Status

The M1 scaffold is now represented in code without changing dispatch behavior:

- Manifest descriptions expose `pipeline_stages` and `epilogue_kind`.
- Benchmark JSONL and PyTorch `selected_gemm_config()` report both fields.
- `src/gemm_sm120_mma_prefill.cuh` and `.cu` define the explicit-MMA prefill
  registration boundary.
- The registration function is called before WMMA prefill entries, but currently
  appends only experimental operations. This keeps default correctness and
  benchmark behavior stable until an explicit-MMA kernel passes promotion gates.

Current prototype:

- direct global fragment loads plus a second shared-memory staging variant.
- register epilogue.
- FP16/BF16 aligned prefill.
- gated by `ZCUTLASS_EXPERIMENTAL_KERNELS`.
- correct under tests and compute-sanitizer, but not a promotion candidate.
  Shared-memory staging improves the direct-global prototype, but scalar
  fragment loads and overly small warp output tiles remain too inefficient.
  A `16x32` warp tile improves the result enough to beat current WMMA on one
  FP16 canonical prefill shape, but it remains far below cuBLAS. A `16x64`
  warp tile regresses. The `ldmatrix` 16x32 follow-up is the best explicit-MMA
  point so far, but still remains below the promotion threshold.

Initial tile config:

```cpp
struct Sm120MmaPrefill64x128x64 {
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 128;
  static constexpr int kBlockK = 64;
  static constexpr int kWarpM = 32;
  static constexpr int kWarpN = 32;
  static constexpr int kWarps = 8;
  static constexpr int kStages = 2;
};
```

The first slice should use PTX `mma.sync.aligned.m16n8k16` forms and explicitly
validate the row-major B staging/`ldmatrix` layout, because incorrect lane
mapping can produce plausible but scrambled tiles.

## 2026-05-23 Iteration Evidence

The latest M1 experiments keep the register epilogue and ldmatrix fragment
mapping, then vary shared-memory staging, K tile depth, and warp output tile
shape. All variants remain experimental-only and are selected only through
`ZCUTLASS_EXPERIMENTAL_KERNELS` / `--experimental-kernel`.

Measured on `128x4096x4096`:

- `64x128x64 ldmatrix`, scalar global-to-shared staging: about `0.163 ms`.
- `64x128x64 ldmatrix`, 16-byte vectorized staging: about `0.116 ms`.
- `64x128x128 ldmatrix`, 16-byte vectorized staging: about `0.108-0.110 ms`.
- `64x128x128 ldmatrix`, `warp32x32`: current best, about `0.102 ms` FP16
  and `0.105 ms` BF16.
- `64x64x128`: regresses, so increasing CTA count by shrinking N is not the
  right prefill direction for this shape.
- `launch_bounds(..., 2)`: reduces registers but introduces stack spill and
  regresses.

Current conclusion: vectorized global-to-shared staging, larger K tiles, and
higher per-warp MMA density are effective. The best explicit-MMA path is now
roughly `1.6x` faster than the first ldmatrix prototype, but still only about
`0.52x-0.56x` of cuBLAS for canonical prefill. This is not promotable to
default dispatch. The next kernel direction should be a deeper mainloop change:
`cp.async`/double buffering or a more CUTLASS-like pipelined shared-memory
layout, not smaller CTA tiles or forced register throttling.

## Implementation Direction

- Keep existing WMMA code intact as fallback and baseline.
- Add explicit-MMA code in a new CUDA translation unit or internal namespace so
  it can be compiled, benchmarked, and reverted independently.
- Start with a conservative tile that maps cleanly to current benchmark shapes:
  `64x128x64`.
- Use register accumulator fragments and convert directly to FP16/BF16 in the
  epilogue.
- Keep launch/dispatch selectable through the existing manifest.

## Next `ldmatrix` Iteration

Current measured variants show that `16x32` is the best scalar shared-memory
warp tile, while `16x64` regresses. Use `16x32` as the baseline for the first
`ldmatrix` implementation.

Recommended shape:

- CTA tile: `64x128x64`.
- Warp tile: `16x32`.
- CTA threads: `16 warps = 512 threads`.
- Per warp per K16 step: one A `16x16`, four B `16x8`, four
  `mma.sync.m16n8k16`.
- Variant name:
  `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_ldm_warp16x32_reg_epilogue`
  and BF16 equivalent.

Initial shared layouts:

- A shared layout: row-major `A_smem[M][K]` with `kAStride = 64 + 8`.
- B shared layout: row-major `B_smem[K][N]` with `kBStride = 128 + 8`.
- Approximate shared memory: `(64 * 72 + 64 * 136) * sizeof(T)`, about 26 KiB
  for FP16/BF16.
- Keep scalar global-to-shared copies for the first `ldmatrix` correctness
  slice; vectorized copy or `cp.async` comes after fragment mapping is proven.

Loader plan:

- A: `ldmatrix.sync.aligned.m8n8.x4.shared.b16`, no `.trans`.
- B: `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`, with B staged as
  row-major `[K,N]`.
- Convert shared pointers to 32-bit shared addresses in a small helper before
  inline PTX.

Risks:

- B can pass rough visual checks while being lane-scrambled, so deterministic
  tile tests must compare against the scalar shared loader.
- `ldmatrix.x4` A register order must match the existing `mma.sync` operand
  order.
- BF16 uses the same b16 `ldmatrix` transport but must keep the BF16 MMA opcode.
- Do not move the ldmatrix variant ahead of WMMA in default dispatch until
  correctness, sanitizer, benchmark, and Nsight gates all pass.

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
