# Agent Operating Notes

## Goal Lock

v1.5 is a verifiable LLM inference GEMM acceleration milestone for RTX
5080/SM120. The optimized path should focus on LLM serving and benchmarking
shapes, especially FP16/BF16 row-major GEMM with FP32 accumulation.

Arbitrary row-major FP16/BF16 GEMM remains supported for correctness, API
coverage, and fallback dispatch. It is not the v1.5 performance target.

zcutlass is an overlay accelerator, not a full replacement for CUTLASS, cuBLAS,
PyTorch, SGLang, or vLLM. Optimized buckets may route to zcutlass first; every
unsupported or non-profitable path must have an observable fallback.

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

## Product Proof Workstreams

- Framework Integration owns PyTorch/SGLang/vLLM adapter plans, explicit
  callsite routing, fallback behavior, and hit/miss instrumentation.
- Serving Validation owns TTFT, TPOT/decode latency, tokens/s, p50/p95/p99,
  workload definitions, output correctness, and fallback reason summaries.
- Framework integration changes must not use `LD_PRELOAD` or global cuBLAS
  interception as the default path. Prefer explicit custom op or framework
  adapter integration.
