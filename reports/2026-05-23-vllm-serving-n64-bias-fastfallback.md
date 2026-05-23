# 2026-05-23 vLLM N64 Bias/Fast-Fallback Iteration

## Scope

This iteration targeted the Qwen2.5-1.5B vLLM overlay path on RTX 5080/SM120.
The goal was to increase real-model prefill hit rate without widening dispatch
to shapes that are known to be slower than the NVIDIA baseline.

## Code Changes

- Added optional bias support to the SM120 explicit-MMA prefill register
  epilogue for `alpha=1`, `beta=0`.
- Marked explicit-MMA prefill operations as `supports_bias=true`.
- Kept arbitrary `beta != 0` on fallback.
- Added vLLM serving harness `--no-log-routes` so latency runs can be separated
  from synchronous JSONL route logging runs.
- Added a vLLM Linear adapter fast fallback for unpromoted decode/large/fallback
  families. This avoids extension/config queries before delegating back to the
  stock vLLM Linear path.

## Validation

- `cmake --build build -j 24`: pass.
- `ctest --test-dir build --output-on-failure`: pass.
- `compute-sanitizer --error-exitcode 99 ./build/zcutlass_tests`: pass.
- Added f16/bf16 explicit-MMA N64 bias correctness coverage.
- `check_vllm_plugin.py --require-entry-point --require-vllm`: pass.
- Qwen2.5-1.5B real-model route smoke: pass.

## Real-Model Routing Result

Route log:

`reports/2026-05-23-vllm-real-model-qwen15-n64-bias-cpasync-routes.jsonl`

Observed rows:

- Total route rows: 448.
- zcutlass hits: 224.
- fallbacks: 224.
- Prefill hits: 224.
- Large fallbacks: 224.
- QKV `M=256,N=2048,K=1536,bias=true` now routes to
  `zcutlass_sm120_mma_bf16_64x64x64_prefill_smem_ldm_cpasync_warp32x32_reg_epilogue`.

This confirms the bias epilogue opened the real QKV prefill route that was
previously stuck on WMMA/fallback behavior.

## Serving Results

Bounded 16-request prefill-heavy, no route logging:

`reports/2026-05-23-vllm-serving-prefill-16-n64-bias-fastfallback-nolog/summary-prefill-heavy.jsonl`

| Metric | stock vLLM | overlay | Result |
| --- | ---: | ---: | --- |
| TTFT mean | 44.0875 ms | 44.4754 ms | 0.991x |
| TTFT p95 | 99.6189 ms | 99.0932 ms | 1.005x |
| TPOT mean | 9.0975 ms | 9.4417 ms | 0.964x |
| tokens/s | 389.0259 | 389.2643 | 1.001x |

Bounded 64-request prefill-only, concurrency 4, no route logging:

`reports/2026-05-23-vllm-serving-prefill-64x384x1-c4-n64-bias-fastfallback-nolog/summary-prefill-heavy.jsonl`

| Metric | stock vLLM | overlay | Result |
| --- | ---: | ---: | --- |
| TTFT mean | 31.3114 ms | 32.6684 ms | 0.958x |
| TTFT p95 | 29.0932 ms | 36.4275 ms | 0.799x |
| tokens/s | 1537.5613 | 1537.4192 | 0.9999x |

## Kernel Evidence

Qwen2.5-1.5B M=384 no-bias microbench with N64 explicit-MMA was tested and
rejected for promotion:

- `384x2048x1536`: 0.501x vs cuBLAS.
- `384x1536x1536`: 0.658x vs cuBLAS.
- `384x17920x1536`: 0.605x vs cuBLAS.
- `384x1536x8960`: 0.637x vs cuBLAS.

Variant sweep over existing explicit-MMA kernels:

`reports/2026-05-23-variant-sweep-qwen15/summary.json`

The current N64 cp.async variant is still the fastest existing zcutlass variant
for Qwen M=256 prefill shapes, but it remains slower than stock vLLM/CUDA
baseline. Dispatch tuning alone is therefore insufficient.

## Decision

Do not declare v1.5 serving success yet.

Keep:

- explicit-MMA bias epilogue;
- no-route-log serving measurement mode;
- vLLM fast fallback;
- negative M384 and variant-sweep evidence.

Do not promote:

- M384/M512 prefill routing;
- decode family;
- large family;
- current N64 as a product-winning vLLM path.

## Next Required Work

The next milestone must be a new kernel experiment, not another routing-only
iteration. The current explicit-MMA design needs higher tensor utilization and
lower per-call overhead before it can beat stock vLLM:

- target Qwen `M=256` BF16 shapes first;
- prioritize QKV/O projection where `N,K` are smaller and serving impact is
  easier to isolate;
- keep route logging disabled for latency measurements and use a separate route
  run for hit/fallback evidence.
