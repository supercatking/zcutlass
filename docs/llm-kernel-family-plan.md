# LLM Kernel Family Plan

This plan turns the current Nsight evidence into the first kernel experiments.
The profile signal is clear enough to choose direction, but not broad enough to
justify shape-specialized kernels: Tensor Core throughput is low and long
scoreboard is the top warp stall. Treat that as a mainloop utilization problem
first, then use epilogue changes only when they remove measurable overhead from
the selected family.

## Guardrails

- Do not add one kernel per exact `m,n,k` shape. Keep variants at the tile family
  level so dispatch remains stable as the LLM suite grows.
- Optimize the mainloop, pipeline depth, load path, tile shape, and epilogue
  policy before expanding the manifest.
- Keep dense aligned fast paths separate from ragged fallback behavior. A faster
  aligned path must not narrow the correctness surface of the generic kernel.
- Promote an experiment only with correctness, benchmark JSONL, and Nsight
  evidence for the target family.

## Common Evidence To Record

For each experiment, capture the previous zcutlass baseline, the modified
zcutlass result, and the best available external baseline. Record:

- Kernel name, dtype, shape, suite family, median latency, and TFLOP/s.
- Nsight SpeedOfLight Tensor Core utilization.
- WarpStateStats long scoreboard, barrier, and math pipe throttling stalls.
- Occupancy, registers per thread, shared memory per block, and waves per SM.
- MemoryWorkloadAnalysis global/shared load efficiency and L2 behavior.
- Framework hit/miss path when the experiment is intended for product
  promotion: model, module, batch size, prompt length, output length, hit rate,
  fallback reason histogram, TTFT, TPOT, tokens/s, p95/p99, and output
  correctness.

The first expected win is lower long scoreboard plus higher Tensor Core issue
rate. If latency improves while Tensor throughput stays low, inspect whether the
change only removed epilogue or launch overhead and keep the conclusion scoped.

## Decode Family

Representative suite: `llm-decode`.

Typical shape pressure is small `M`, large `N/K`, and high sensitivity to
per-block overhead. Existing `32x*` registrations are placeholders because the
current reused WMMA body was slower in spot checks; do not promote them until
the mainloop is actually different.

Model mapping: batch/token decode Linear layers, especially projection and MLP
GEMMs whose effective `M` is the active token count. End-to-end value should
show up in TPOT/decode token latency before TTFT.

First experiment:

1. Build a small-M tile family around the existing `32x64x16` and `32x128x16`
   names, but change the mainloop rather than only dispatch order.
2. Prioritize double-buffered global-to-shared staging for the K loop to cover
   scoreboard latency.
3. Keep the epilogue dense and direct for `alpha=1,beta=0,bias=null`; avoid
   extra predicates on the aligned decode path.

Decision criteria:

- Keep only if decode latency improves on both f16 and bf16 without regressing
  the current `64x*` fallback path for ragged decode shapes.
- Reject or park if long scoreboard remains dominant and Tensor throughput does
  not move; that means the small tile alone did not address the profile.

## Prefill Family

Representative suite: `llm-prefill`.

Prefill has larger `M` than decode while still matching LLM-like large `N/K`
matrices. It should be the first family to validate whether a pipelined WMMA
mainloop is enough before moving to explicit MMA.

Model mapping: prompt/prefill QKV, output projection, and MLP up/down/gate
Linear layers. End-to-end value should show up in TTFT and prefill throughput.

First experiment:

1. Keep the `64x128x16` family as the primary candidate.
2. Add a double-buffered K mainloop and measure whether it reduces long
   scoreboard and barrier stalls.
3. Preserve the aligned fast epilogue for dense prefill, but keep fallback
   epilogue coverage for ragged edges and nonzero beta.

Decision criteria:

- Promote if Tensor Core utilization rises and prefill median latency improves
  across the suite, not just one benchmark shape.
- If occupancy collapses from registers or shared memory, try pipeline depth or
  staging layout adjustments before adding another tile family.

## Large / Throughput Family

Representative suites: `llm` large cases, `square`, and explicit large
throughput shapes such as `m=256,n=4096,k=4096`.

These shapes should expose sustained mainloop throughput. They are the best
place to decide whether the baseline WMMA path has reached its ceiling.

Model mapping: high-concurrency serving, large batch, and throughput-mode
prefill where the engine can accumulate enough work to saturate GEMM.

First experiment:

1. Use `64x128x16` as the default throughput tile and keep `64x64x16` as the
   comparison point for occupancy and scheduling behavior.
2. Measure double buffering first. If long scoreboard falls but Tensor
   throughput is still far below target, plan the explicit MMA/register
   epilogue experiment next.
3. Inspect whether the generic epilogue still consumes meaningful time after
   mainloop changes. If it does, specialize epilogue policy by aligned dense
   family rather than by full problem shape.

Decision criteria:

- Promote only with broad throughput improvement and no correctness regression
  in the smoke and correctness suites.
- Prefer a single stronger throughput tile over multiple close variants unless
  Nsight shows a clear occupancy or memory-system split.

## Fallback Family

Representative suite: `ragged` plus correctness shapes with boundaries, beta,
bias, and non-aligned dimensions.

Fallback kernels protect the public API surface. Their job is not to win peak
throughput; their job is to remain correct and predictable while aligned paths
take the aggressive optimizations.

Model mapping: unsupported dtype/layout, ragged or padded leading dimensions,
non-target shapes, beta/bias combinations, or cases where zcutlass predicts it
will not beat the stock framework path.

First experiment:

1. Keep fallback tile count small and avoid copying dense-only assumptions into
   boundary handling.
2. Reuse mainloop improvements only when predicates and partial tiles remain
   straightforward.
3. Audit epilogue cost separately for beta and bias cases before adding another
   fallback variant.

Decision criteria:

- Promote if correctness stays complete and ragged latency does not regress
  materially.
- If the dense aligned path improves but fallback becomes slower or more complex,
  keep the fallback kernel conservative and document the tradeoff in the
  measurement record.

## Experiment Order

1. Prefill `64x128x16` double-buffered mainloop.
2. Large/throughput validation of the same mainloop on `m=256,n=4096,k=4096`
   and wider `llm` cases.
3. Decode small-M mainloop only after the shared mainloop evidence is reviewed.
4. Fallback audit after aligned changes, focused on correctness and bounded
   regression rather than peak speed.

This order follows the current profile: fix long scoreboard and Tensor Core
underutilization in the common mainloop first, then decide whether the next
experiment should be explicit MMA/register epilogue or a narrower small-M
decode family.

## Framework Integration Milestones

M1: PyTorch Overlay Proof.

- Provide a documented route for `torch.ops.zcutlass.gemm` or `zcutlass_linear`
  that only replaces explicit Linear/GEMM callsites selected by the experiment.
- Fallback to `torch.nn.functional.linear` or the original PyTorch matmul when
  dtype, layout, shape, alignment, correctness mode, or predicted performance is
  not acceptable.
- Acceptance: one offline model path passes numerical correctness and records
  selected GEMM latency plus zcutlass hit/miss counts.

M2: SGLang Serving Proof.

- Treat SGLang as the first serving engine target.
- zcutlass is a GEMM overlay only; do not replace attention, KV cache,
  scheduling, or sampling.
- Acceptance: stock SGLang vs SGLang plus zcutlass overlay on the same model and
  workload, recording TTFT, TPOT, tokens/s, p50/p95/p99, hit rate, and fallback
  reasons.

M3: vLLM OOT CustomOp Proof.

- Use vLLM out-of-tree CustomOp integration instead of `LD_PRELOAD` or global
  cuBLAS interception.
- Acceptance: stock vLLM vs vLLM plus zcutlass overlay on decode-heavy,
  prefill-heavy, and mixed serving workloads.

M4: Commercial Value Gate.

- v1.5 product value is established only after at least one real serving engine
  shows stable TTFT or TPOT improvement, correctness passes, fallback is
  observable and safe, p95/p99 does not materially regress, and microbenchmark,
  Nsight, and end-to-end metrics tell the same story.

M5: Post-v1.5 CUTLASS Alignment.

- After the serving proof, expand toward grouped GEMM/MoE, richer epilogue
  fusion, FP8/FP4/block-scaled inference, more layouts/dtypes, and a fuller
  profiler/autotune system.
- These items are not v1.5 success criteria.
