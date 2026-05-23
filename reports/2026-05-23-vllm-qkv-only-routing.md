# 2026-05-23 vLLM QKV-Only Routing Experiment

## Purpose

The default vLLM overlay wrapped `qkv_proj,gate_up_proj,down_proj,o_proj`.
Route logs showed many fallback calls from unpromoted decode/large/off-bucket
shapes. This experiment tested whether wrapping fewer layers could expose a
real prefill serving win while preserving observable zcutlass hits.

## Integration Changes

- `ZCUTLASS_VLLM_TRACE_CONFIG=0` skips per-hit `selected_gemm_config()` calls
  in latency runs.
- `--no-log-routes` now sets trace config off.
- The OOT Linear mixin skips `perf_counter_ns()` when no route logger is
  attached.
- The vLLM LinearMethod fast-delegates unpromoted families before transposing
  weights, so decode/large/fallback calls avoid zcutlass weight-cache overhead.

## Route Evidence

Route run:

`reports/2026-05-23-vllm-serving-prefill-qkvonly-routes/summary-prefill-heavy.jsonl`

Route log:

`reports/2026-05-23-vllm-serving-prefill-qkvonly-routes/routes-overlay-prefill-heavy.jsonl`

Summary:

- wrapped layer filter: `qkv_proj`
- route rows: `308`
- zcutlass hits: `28`
- fallbacks: `280`
- zcutlass shape: `M=256,N=2048,K=1536,BF16,bias=true`
- selected kernel:
  `zcutlass_sm120_mma_bf16_64x64x64_prefill_smem_ldm_cpasync_warp32x32_reg_epilogue`

The run confirms QKV-only routing hits zcutlass for the M=256 chunked-prefill
QKV path, while M=384/off-bucket and large/decode paths still fall back.

## Latency Runs

Prefill-only workload: local Qwen2.5-1.5B, BF16, input length 384, output length
1, 64 prompts, request rate 4, max concurrency 4, route logging disabled.

First run:

`reports/2026-05-23-vllm-serving-prefill-64x384x1-c4-qkvonly-traceskip-nolog/summary-prefill-heavy.jsonl`

| Metric | stock vLLM | overlay | Result |
| --- | ---: | ---: | --- |
| TTFT mean | 35.8758 ms | 31.9631 ms | 1.122x |
| TTFT p95 | 38.3009 ms | 29.8858 ms | 1.282x |
| TTFT p99 | 266.3791 ms | 146.5033 ms | 1.818x |
| tokens/s | 1537.4897 | 1537.5053 | 1.000x |

Rerun:

`reports/2026-05-23-vllm-serving-prefill-64x384x1-c4-qkvonly-traceskip-nolog-rerun/summary-prefill-heavy.jsonl`

The rerun showed a large stock-tail outlier, so it cannot be used as a clean
proof by itself.

After additional overhead cleanup:

`reports/2026-05-23-vllm-serving-prefill-64x384x1-c4-qkvonly-overhead2-nolog/summary-prefill-heavy.jsonl`

| Metric | stock vLLM | overlay | Result |
| --- | ---: | ---: | --- |
| TTFT mean | 31.2391 ms | 31.5055 ms | 0.992x |
| TTFT p50 | 26.6253 ms | 26.9581 ms | 0.988x |
| TTFT p95 | 28.7882 ms | 29.0365 ms | 0.991x |
| TTFT p99 | 141.6062 ms | 139.2749 ms | 1.017x |
| tokens/s | 1537.4033 | 1537.3068 | 1.000x |

## Decision

Do not claim v1.5 serving success from QKV-only routing yet.

QKV-only routing is useful as a reduced-overhead experimental mode and route
evidence path, but the latency win is not stable after overhead cleanup. The
kernel is still slower than stock vLLM for the core QKV callsite:

- QKV BF16 `M=256,N=2048,K=1536,bias=true`: zcutlass overlay around
  `0.040-0.041 ms`, stock vLLM around `0.028-0.030 ms`.

## Next Step

The next product-level win requires a faster QKV/O prefill kernel, not more
wrapper tuning. Keep using QKV-only as the first serving proof target once a new
kernel variant clears the single-call gate.
