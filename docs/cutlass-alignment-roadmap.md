# CUTLASS Alignment Roadmap

zcutlass is a clean-room implementation. CUTLASS is a public design reference and
external benchmark target, not a source dependency.

## Current Scope

- Target NVIDIA GeForce RTX 50 series (`sm_120`).
- Keep the v1 API focused on row-major FP16/BF16 GEMM with FP32 accumulation.
- Lock v1.5 on optimized LLM GEMM for RTX 5080/SM120. Arbitrary row-major
  FP16/BF16 GEMM remains supported for correctness and fallback, but is not the
  optimized target.
- Treat cuBLAS and external `cutlass_profiler` results as baselines.
- Keep CUTLASS source outside this repository.

## Alignment Goals

- Match CUTLASS terminology in docs where it clarifies concepts: tile shape,
  pipeline depth, epilogue, alignment, and problem shape.
- Track comparable problem shapes before adding new kernel variants.
- Build reusable tile families and shape-bucket dispatch instead of many
  full-shape-specific kernels.
- Prefer measurable parity targets over broad feature claims.
- Record deviations when zcutlass intentionally chooses a smaller surface.

## Roadmap

1. Stabilize correctness, benchmark JSONL, and smoke/LLM problem coverage.
2. Add CUTLASS-style `arch`, `layout`, `numeric`, `operation`, and `manifest`
   extension points while keeping the simple public GEMM entry point.
3. Define v1.5 LLM shape buckets for RTX 5080/SM120 and map them to a small
   number of reusable tile/pipeline families.
4. Compare selected FP16/BF16 LLM buckets against cuBLAS and external CUTLASS
   profiler.
5. Add documented kernel selection notes once multiple internal variants exist.
6. Revisit non-v1 features only after the LLM row-major path is measured well.

## Out of Scope for v1

Mixed dtype, non-row-major layouts, grouped GEMM, sparse GEMM, FP8/FP4, and
attention kernels remain product backlog items, not v1 commitments.

Full-shape-specific kernel proliferation is also out of scope. v1.5 should not
grow a separate optimized kernel for every hidden-size/token-count combination;
it should grow tile families and dispatch buckets that preserve correctness
fallback for arbitrary row-major FP16/BF16 GEMM.
