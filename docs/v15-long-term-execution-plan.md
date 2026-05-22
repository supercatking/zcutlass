# zcutlass v1.5 Long-Term Execution Plan

## Objective

zcutlass v1.5 is an RTX 5080 / SM120 LLM GEMM overlay accelerator.

The product goal is not to replace CUTLASS, cuBLAS, PyTorch, SGLang, or vLLM
globally. The goal is to win selected LLM GEMM buckets, route those buckets to
zcutlass through explicit framework adapters, and safely fall back for everything
else.

The value gate is end-to-end serving evidence: zcutlass must improve TTFT, TPOT,
tokens/s, or tail latency on at least one real LLM serving path without
correctness or fallback regressions.

## Current Baseline

Known completed work:

- C++/CUDA GEMM API and SM120-focused benchmark path exist.
- PyTorch extension path exists through `zcutlass_torch`.
- Explicit PyTorch overlay reports hit/miss/fallback reasons.
- vLLM can discover and load the `zcutlass_overlay` general plugin.
- vLLM can register the experimental `ZCutlassToyForCausalLM` model probe.
- vLLM synthetic adapter probes can route FP16/BF16 prefill-shaped Linear calls
  to zcutlass.

Known gap:

- Current zcutlass GEMM kernels are not yet competitive with PyTorch/cuBLAS on
  the tested prefill adapter path.
- Therefore, the immediate priority is kernel-level evidence and optimization,
  not broader framework integration.

## Long-Term Milestones

### M0: State Freeze And Reproducibility

Purpose: make the current state reproducible before widening scope.

Deliverables:

- Documented validation commands for C++ tests, PyTorch extension checks, vLLM
  plugin checks, and vLLM adapter probes.
- Current known performance gaps recorded in reports.
- Dirty generated artifacts kept out of commits unless they are intentionally
  part of the measurement record.

Acceptance:

- `ctest --test-dir build --output-on-failure` passes.
- `python tools/check_torch_overlay.py --require-extension` passes in the
  relevant PyTorch environment.
- `python tools/check_vllm_plugin.py --require-entry-point --require-vllm`
  passes in `/home/zyz/vllm/.venv`.
- `python tools/check_vllm_linear_adapter.py --require-hit` passes for a
  prefill-shaped FP16 case.

### M1: LLM v1.5 Microbenchmark Evidence Chain

Purpose: establish a reliable performance baseline before kernel changes.

Canonical shapes:

- Decode: `1x4096x4096`, `8x4096x4096`, `16x4096x4096`.
- Prefill: `64x4096x4096`, `128x4096x4096`, `256x4096x4096`.
- MLP up: `128x16384x4096`.
- MLP down: `128x4096x16384`.
- Square sanity: `4096x4096x4096`.

Providers:

- zcutlass C++ benchmark.
- cuBLAS through the existing benchmark provider.
- official CUTLASS profiler when available.
- PyTorch matmul/Linear overlay reports for framework-level context.

Acceptance:

- JSONL result file with commit hash, GPU, CUDA, dtype, shape family, provider,
  kernel name, median latency, TFLOP/s, and route/fallback metadata.
- HTML summary chart generated from JSONL.
- A Markdown report that identifies the first optimization target and explains
  why.

### M2: Nsight Compute Evidence Loop

Purpose: explain why zcutlass is slow before changing kernels.

Priority profiles:

- `f16 128x4096x4096`
- `bf16 128x4096x4096`
- `f16 8x4096x4096`
- `bf16 8x4096x4096`

Metrics:

- Tensor Core utilization.
- SM throughput.
- DRAM/L2 throughput.
- occupancy and waves per SM.
- registers per thread.
- shared memory per CTA.
- top warp stall reasons.
- SASS instruction path.

Acceptance:

- `.ncu-rep` collected when counter permissions allow it.
- CSV, JSON, and Markdown summaries produced by repo tools.
- If counter permissions block profiling, record the failure and preserve the
  latency benchmark results.

### M3: Prefill Kernel First Win

Purpose: win the first high-value LLM GEMM bucket.

Initial target:

- `M=128, K=4096, N=4096`
- `M=128, K=4096, N=16384`
- `M=128, K=16384, N=4096`

Kernel direction:

- Keep tile-family specialization, not full `(M,N,K)` specialization.
- Start with `64x128x16`, `128x128x64`, and `128x256x64` candidates.
- Prioritize double-buffered K mainloop and vectorized load/store.
- Preserve fallback correctness and existing WMMA baseline as a fallback path.

Acceptance:

- At least one FP16 prefill shape approaches or beats PyTorch/cuBLAS.
- BF16 remains a first-class target and is not left unexplained.
- Each promoted variant has correctness, benchmark, and profile evidence.

### M4: Decode Kernel Track

Purpose: target small-M decode latency after the shared mainloop evidence is
understood.

Shapes:

- `M=1,8,16`
- `K=4096`
- `N=4096`, `N=11008`, or `N=16384`

Possible directions:

- small-M warp-specialized GEMM/GEMV.
- split-N or persistent CTA strategy.
- launch overhead analysis.
- decode batching opportunities.

Acceptance:

- Decode-heavy microbenchmark report with wins, losses, and fallback reasons.
- A documented conclusion on whether current bottleneck is kernel throughput or
  launch/scheduling overhead.

### M5: Framework Overlay Integration

Purpose: make framework routing observable and reversible.

Order:

1. PyTorch explicit module overlay.
2. SGLang adapter proof.
3. vLLM out-of-tree/custom model or targeted module replacement.

Rules:

- No global cuBLAS hook as the main path.
- No unbounded monkey patching.
- No attention, KV cache, scheduler, or sampling replacement in v1.5.
- zcutlass must be disabled by default until performance evidence justifies
  promotion.

Acceptance:

- Every framework path records shape, dtype, hit/miss, fallback reason, and
  selected kernel path.
- Environment-variable or config-based kill switch exists.
- Framework correctness passes before performance claims.

### M6: vLLM End-To-End Proof

Purpose: prove product value in a real serving stack.

Precondition:

- At least one zcutlass LLM GEMM bucket is competitive at the microbenchmark
  level.

Workloads:

- Decode-heavy: short prompt, long output.
- Prefill-heavy: long prompt, short output.
- Mixed: realistic concurrent serving.

Metrics:

- TTFT.
- TPOT.
- tokens/s.
- p50/p95/p99 latency.
- zcutlass hit rate.
- fallback reason histogram.
- output correctness.

Acceptance:

- TTFT or TPOT improves in at least one workload.
- p95/p99 does not materially regress.
- Microbenchmark, Nsight, and serving results tell a consistent story.

### M7: Commercial Value Gate

zcutlass v1.5 product value is established only when:

- correctness passes;
- fallback is safe and observable;
- at least one real serving engine shows stable improvement;
- tail latency is not materially worse;
- profile data explains the result;
- zcutlass can be disabled cleanly.

Before this gate, project status should be described as integration-ready but
not commercially validated.

### M8: Post-v1.5 CUTLASS Alignment

After v1.5:

- grouped GEMM and MoE;
- FP8/FP4 and block-scaled inference;
- richer epilogue fusion;
- additional layouts and dtypes;
- fuller profiler and autotuning database;
- CUTLASS-style tile, collective, mainloop, and epilogue abstraction;
- broader GPU architecture coverage.

These are not v1.5 success criteria.

## Multi-Agent Workstreams

Coordinator:

- Own task queue, integration, commits, pushes, and Win11 mirror sync.
- Serialize GPU work.
- Prevent concurrent edits to hot files such as `src/gemm.cu`.

Kernel Agent:

- Own kernel variants and manifest changes.
- Every kernel change must include correctness and benchmark evidence.

Profiling Agent:

- Own Nsight Compute, `nvdisasm`, SASS checks, and summary reports.
- Does not modify kernel code.

Framework Agent:

- Own PyTorch, SGLang, and vLLM adapters.
- Does not modify low-level GEMM kernels.

Validation Agent:

- Own correctness tests, sanitizer, tolerance policy, and regression coverage.
- Does not make performance optimizations.

Reporting Agent:

- Own JSONL, Markdown, HTML charts, and roadmap updates.
- Ensures benchmark records include commit hash and environment metadata.

## Execution Rules

- CPU builds, code reading, documentation, scripts, and report parsing can run
  in parallel.
- GPU benchmark, Nsight Compute, and compute-sanitizer runs are serialized.
- `src/gemm.cu` has single-writer ownership during kernel work.
- Each performance claim must include a correctness result.
- Each commit must include a detailed body explaining purpose, scope, and
  validation.
- Generated reports are committed only when they are part of the evidence chain.
- User-generated or unrelated dirty files are left untouched.

## Immediate Execution Batch

Batch 1 starts with evidence rather than kernel changes:

1. Generate the LLM v1.5 microbenchmark baseline.
2. Generate a visual summary.
3. Attempt Nsight Compute profile collection for the first prefill FP16/BF16
   targets.
4. If profiling is blocked by permissions, record the blocker and continue with
   latency data.
5. Use the evidence to select the first prefill kernel experiment.
