# CUTLASS Alignment Roadmap

zcutlass is a clean-room implementation. CUTLASS is a public design reference and
external benchmark target, not a source dependency.

## Current Scope

- Target NVIDIA GeForce RTX 50 series (`sm_120`).
- Keep the v1 API focused on row-major FP16/BF16 GEMM with FP32 accumulation.
- Lock v1.5 on a verifiable LLM inference GEMM acceleration layer for RTX
  5080/SM120. zcutlass should run first only for selected optimized GEMM buckets;
  unsupported or non-profitable work falls back to the original framework,
  cuBLAS, or external CUTLASS path.
- Treat cuBLAS and external `cutlass_profiler` results as baselines.
- Keep CUTLASS source outside this repository.

## Alignment Goals

- Match CUTLASS terminology in docs where it clarifies concepts: tile shape,
  pipeline depth, epilogue, alignment, and problem shape.
- Track comparable problem shapes before adding new kernel variants.
- Build reusable tile families and shape-bucket dispatch instead of many
  full-shape-specific kernels.
- Prove product value with framework-level TTFT, TPOT/decode latency, tokens/s,
  p95/p99, hit-rate, and fallback-rate measurements, not only microbenchmarks.
- Prefer measurable parity targets over broad feature claims.
- Record deviations when zcutlass intentionally chooses a smaller surface.

## Roadmap

### v1.5: LLM GEMM Overlay

1. Stabilize correctness, benchmark JSONL, and smoke/LLM problem coverage.
2. Add CUTLASS-style `arch`, `layout`, `numeric`, `operation`, and `manifest`
   extension points while keeping the simple public GEMM entry point.
3. Define v1.5 LLM shape buckets for RTX 5080/SM120 and map them to a small
   number of reusable tile/pipeline families.
4. Compare selected FP16/BF16 LLM buckets against cuBLAS and external CUTLASS
   profiler for kernel-level evidence.
5. Add PyTorch overlay proof for explicit Linear/GEMM callsites with safe
   fallback to stock PyTorch.
6. Add SGLang serving proof as the first real serving target, measuring stock
   SGLang against SGLang plus zcutlass overlay.
7. Add vLLM OOT CustomOp proof after the overlay semantics are stable.
8. Declare v1.5 product value only when at least one serving engine shows stable
   TTFT or TPOT improvement with correctness, safe fallback, and no material
   p95/p99 regression.

### Post-v1.5: Broader CUTLASS Alignment

After v1.5 is proven in a real inference stack, expand toward a broader
CUTLASS-style platform: grouped GEMM/MoE, richer epilogue fusion, FP8/FP4 and
block-scaled LLM inference, more layouts/dtypes, and a more complete
profiler/autotune flow.

Do not expand v1.5 scope merely to look more like CUTLASS. CUTLASS remains a
design reference and fallback/baseline; zcutlass v1.5 is an overlay accelerator,
not a full replacement.

## Out of Scope for v1

Mixed dtype, non-row-major layouts, grouped GEMM, sparse GEMM, FP8/FP4, and
attention kernels remain product backlog items, not v1 commitments.

Full-shape-specific kernel proliferation is also out of scope. v1.5 should not
grow a separate optimized kernel for every hidden-size/token-count combination;
it should grow tile families and dispatch buckets that preserve correctness
fallback for arbitrary row-major FP16/BF16 GEMM.
