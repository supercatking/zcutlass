# Agent Operating Notes

## Goal Lock

v1.5 is a dedicated LLM GEMM optimization milestone for RTX 5080/SM120. The
optimized path should focus on LLM serving and benchmarking shapes, especially
FP16/BF16 row-major GEMM with FP32 accumulation.

Arbitrary row-major FP16/BF16 GEMM remains supported for correctness, API
coverage, and fallback dispatch. It is not the v1.5 performance target.

## Kernel Policy

- Do not create many full-shape-specific kernels for individual LLM dimensions.
- Prefer CUTLASS-style tile families with explicit tile shape, pipeline depth,
  epilogue, and alignment choices.
- Dispatch through shape buckets that map nearby problem sizes to reusable tile
  families.
- Keep fallback coverage broad enough that unsupported or off-bucket row-major
  FP16/BF16 shapes still produce correct results.

## Coordination

- Multiple agents may be active. Do not revert changes you did not make.
- Keep edits within assigned ownership unless the coordinator expands scope.
- Documentation-only work must not touch code, tests, tools, generated reports,
  or build artifacts.
- Kernel work should update roadmap and measurement notes before widening the
  optimized dispatch surface.
