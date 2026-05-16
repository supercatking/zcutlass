# Agent Coordination

zcutlass development uses parallel agents for CPU-heavy implementation and
analysis, but serializes GPU measurements on the single RTX 5080.

## v1.5 Goal Lock

v1.5 optimizes LLM GEMM on RTX 5080/SM120. Arbitrary row-major FP16/BF16 GEMM
remains supported for correctness and fallback dispatch, but it is not the v1.5
optimized target.

Agents must not expand performance work by adding many full-shape-specific
kernels. Use CUTLASS-style tile families and shape-bucket dispatch so optimized
coverage stays measurable, reusable, and reviewable.

## Roles

- Coordinator owns integration, validation order, commits, GitHub pushes, and
  the Windows mirror.
- Profiling Infrastructure owns Nsight Compute automation and profiling
  summaries.
- CUTLASS Baseline owns external CUTLASS profiler workflows and comparison
  records.
- Correctness owns test coverage and sanitizer evidence.
- Kernel agents own one performance experiment at a time in `src/gemm.cu`.

## Write Ownership

- Agents must keep edits inside their assigned files unless the coordinator
  explicitly expands the scope.
- `src/gemm.cu` is single-writer. Only one kernel agent may edit it at a time.
- Shared docs may be edited by multiple agents only when their sections do not
  overlap.
- Generated artifacts belong in `reports/` when they are durable and in
  `build/reports/` when they are temporary.
- Documentation-only assignments must stay out of code, tests, tools, generated
  reports, and build artifacts.

## GPU Queue

Run only one GPU-heavy task at a time:

1. `compute-sanitizer`
2. `zcutlass_bench`
3. official CUTLASS profiler
4. `ncu`

CPU builds, JSON parsing, documentation, and report generation can run in
parallel. On this host, use up to `-j 24` for zcutlass builds and `-j 16..24`
for large external CUTLASS builds.

## Merge Gate

Before merging a performance change:

- `ctest --test-dir build --output-on-failure` passes.
- At least one relevant benchmark JSONL exists.
- Kernel/mainloop changes include Nsight Compute evidence or a documented
  blocker such as `ERR_NVGPUCTRPERM`.
- CUTLASS comparisons record the external CUTLASS commit, profiler path, kernel
  name, and any layout caveat.
- Windows mirror sync completes.
