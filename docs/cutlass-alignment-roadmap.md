# CUTLASS Alignment Roadmap

zcutlass is a clean-room implementation. CUTLASS is a public design reference and
external benchmark target, not a source dependency.

## Current Scope

- Target NVIDIA GeForce RTX 50 series (`sm_120`).
- Keep the v1 API focused on row-major FP16/BF16 GEMM with FP32 accumulation.
- Treat cuBLAS and external `cutlass_profiler` results as baselines.
- Keep CUTLASS source outside this repository.

## Alignment Goals

- Match CUTLASS terminology in docs where it clarifies concepts: tile shape,
  pipeline depth, epilogue, alignment, and problem shape.
- Track comparable problem shapes before adding new kernel variants.
- Prefer measurable parity targets over broad feature claims.
- Record deviations when zcutlass intentionally chooses a smaller surface.

## Roadmap

1. Stabilize correctness, benchmark JSONL, and smoke/LLM problem coverage.
2. Add CUTLASS-style `arch`, `layout`, `numeric`, `operation`, and `manifest`
   extension points while keeping the simple public GEMM entry point.
3. Compare selected FP16/BF16 shapes against cuBLAS and external CUTLASS profiler.
4. Add documented kernel selection notes once multiple internal variants exist.
5. Revisit non-v1 features only after the dense row-major path is measured well.

## Out of Scope for v1

Mixed dtype, non-row-major layouts, grouped GEMM, sparse GEMM, FP8/FP4, and
attention kernels remain product backlog items, not v1 commitments.
