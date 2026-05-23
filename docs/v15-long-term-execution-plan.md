# zcutlass v1.5 Long-Term Execution Plan

## Objective

zcutlass v1.5 targets one product proof: on RTX 5080 / SM120, zcutlass acts as
an explicit vLLM GEMM overlay accelerator and improves at least one real
small-model serving workload versus stock vLLM/NVIDIA baseline paths.

The first model target is a Qwen/Llama-like 1.5B-3B BF16/FP16 model on one RTX
5080. The first technical bet is an explicit-MMA prefill kernel family that
removes the current WMMA FP32-accumulator shared-memory spill.

No commercial value claim is valid until stock vLLM vs vLLM+zcutlass serving
shows a stable TTFT or TPOT win with safe fallback and no material p95/p99
regression.

## Current State And Gaps

Completed:

- C++/CUDA GEMM API, manifest dispatch, tests, benchmark, profiler summaries.
- PyTorch extension and explicit overlay telemetry.
- vLLM plugin discovery, LinearMethod wrapper, and repeatable LinearMethod
  benchmark harness.
- FP16/BF16 prefill WMMA variants with selected-kernel telemetry.
- BF16 prefill `64x128x32` promoted over the previous BF16 WMMA baseline.

Remaining gaps:

- Current kernels are still slower than stock vLLM/cuBLAS/CUTLASS at framework
  level.
- WMMA accumulator fragments cannot directly store FP16/BF16 outputs while
  keeping FP32 accumulation, so the current path spills accumulators through
  shared memory.
- No true vLLM model execution path has been patched yet.
- Decode small-M, MLP up/down, and large throughput paths need separate proof.
- Serving metrics are not yet collected for stock vLLM vs zcutlass overlay.

## Milestones

### M0: Baseline Freeze

Purpose: lock current truth before more kernel work.

Deliverables:

- One-command M0 runner that records commit, dirty files, build/test status,
  kernel microbenchmarks, vLLM plugin checks, and LinearMethod smoke metrics.
- Baseline report under `reports/YYYY-MM-DD-m0-baseline/`.
- Dirty user/manual reports excluded unless explicitly selected.

Acceptance:

- `ctest --test-dir build --output-on-failure` passes.
- `check_torch_overlay.py --require-extension` passes in the active vLLM/PyTorch
  environment.
- `check_vllm_plugin.py --require-entry-point --require-vllm` passes.
- `benchmark_vllm_linear_method.py --suite smoke --dtype both` writes JSONL.

### M1: Explicit-MMA Prefill Prototype

Purpose: replace the WMMA prefill ceiling with a register-epilogue kernel.

Deliverables:

- New SM120 explicit-MMA prefill family, separate from the current WMMA fallback.
- Initial aligned dense scope only: row-major A/B/D, FP16/BF16 storage, FP32
  accumulation, `alpha=1,beta=0,bias=null`, prefill family only.
- Manifest fields that expose tile, stage count, epilogue type, and kernel name.

Initial target:

- `128x4096x4096` for FP16 and BF16.

Acceptance:

- Correctness passes against CPU/cuBLAS reference.
- Nsight shows tensor activity materially above current WMMA prefill path.
- Gap to cuBLAS/CUTLASS is measured and explained, even if not yet closed.

### M2: Prefill Promotion Candidate

Purpose: make explicit-MMA useful beyond one shape.

Target shapes:

- `128x4096x4096`
- `128x16384x4096`
- `128x4096x16384`

Promotion gates:

- No prefill shape slower by more than `3%` versus current zcutlass default.
- Prefill geomean improvement over current zcutlass at least `1.10x`.
- Achieve at least `0.80x` of cuBLAS/CUTLASS before any real-model vLLM
  promotion attempt.
- Achieve at least `1.03x` over cuBLAS/CUTLASS before claiming a kernel-level
  NVIDIA baseline win.

Acceptance:

- FP16/BF16 JSONL, Nsight `.ncu-rep`, CSV, summary JSON/MD.
- fallback/ragged/beta/bias correctness remains intact.
- vLLM LinearMethod benchmark records the selected explicit-MMA kernel.

### M3: vLLM Real Model Overlay

Purpose: move from dummy LinearMethod to real model execution.

Deliverables:

- Explicit vLLM model/module route for selected Linear callsites.
- Environment switches:
  - `ZCUTLASS_VLLM_ENABLE=1`
  - `ZCUTLASS_VLLM_ALLOW_FAMILIES=prefill`
  - `ZCUTLASS_VLLM_LOG_ROUTES=1`
- Route log JSONL with model/layer, shape, dtype, hit/fallback, fallback reason,
  selected kernel, and per-callsite latency.

Acceptance:

- Qwen/Llama-like 1.5B-3B model generates successfully.
- Output correctness/numerical tolerance passes for fixed prompts.
- Unsupported paths safely use stock vLLM.
- Fallback histogram is complete.

### M4: Serving Proof

Purpose: prove product value, not only microbenchmark improvement.

Workloads:

- Decode-heavy: short prompt, long output.
- Prefill-heavy: long prompt, short output.
- Mixed: multiple request lengths and concurrency levels.

Metrics:

- TTFT, TPOT, tokens/s.
- p50/p95/p99 latency.
- zcutlass hit rate.
- selected-kernel distribution.
- fallback reason histogram.
- output correctness.

Acceptance:

- At least one workload improves TTFT or TPOT by `>=3%`.
- p95/p99 does not materially regress.
- Microbenchmark, Nsight, route logs, and serving results tell the same story.
- If no win, the report identifies the bottleneck: kernel, route overhead,
  weight transpose/cache, launch/scheduler, or non-GEMM dominance.

### M5: Decode Track

Purpose: improve TPOT once prefill has a credible path.

Deliverables:

- Small-M kernel family for `M in {1,8,16}`.
- Evaluation of split-N, persistent CTA, and GEMV-like strategies.
- Decode-heavy vLLM workload report.

Acceptance:

- TPOT improvement is measurable or a report proves launch/scheduler overhead
  dominates GEMM-only optimization.

### M6: Product Hardening

Deliverables:

- Autotune/dispatch table versioning.
- Kernel fallback safety policy.
- One-command kernel + vLLM benchmark runner.
- HTML report for microbenchmark and serving comparisons.
- CI smoke: build, ctest, Python import, vLLM plugin check, no-GPU skip.

Acceptance:

- A new machine can reproduce baseline and overlay checks.
- zcutlass overlay can be enabled/disabled cleanly.
- Reports clearly separate stock vLLM, cuBLAS/CUTLASS, and zcutlass results.

### M7: Post-v1.5 CUTLASS Alignment

Only after v1.5 serving proof:

- grouped GEMM / MoE.
- richer epilogue fusion.
- FP8/FP4/block-scaled inference.
- more layouts and dtypes.
- fuller profiler/autotune.
- CUTLASS-style collective/mainloop/epilogue abstraction.

These are not v1.5 success criteria.

## Agent Workstreams

- Coordinator: task queue, GPU serialization, integration, commits, GitHub
  pushes, Win11 mirror sync.
- Kernel-MMA Agent: explicit-MMA mainloop, tile family, SASS/Nsight evidence.
- Epilogue Agent: register epilogue, FP32-to-FP16/BF16 conversion, future fusion.
- Decode Agent: small-M decode GEMM/GEMV family.
- vLLM Agent: real-model overlay, route logging, fallback behavior.
- Serving Agent: TTFT/TPOT/tokens/s/p95/p99 workloads.
- Profiling Agent: ncu summaries, CUTLASS/cuBLAS comparison, regression gates.
- Docs Agent: decision log, roadmap, reports.

Rules:

- GPU experiments are serialized.
- CPU build, scripts, docs, and report parsing can run in parallel.
- Kernel source has single-writer ownership per batch.
- Every promotion needs correctness, benchmark, profiling, and fallback evidence.
- User/manual dirty files are not reverted or mixed into unrelated commits.

## Immediate Batch

1. Add and run the M0 baseline runner.
2. Record current gaps in a dated M0 report.
3. Add explicit-MMA prefill design notes.
4. Use agent research to select the first implementation slice.
5. Start M1 only after M0 report is committed.
