# 2026-05-23 vLLM Trace Config Skip

## Purpose

Serving latency runs should not pay for per-call kernel metadata queries when
route JSONL logging is disabled. The previous vLLM adapter still called
`selected_gemm_config()` before every zcutlass hit, even under
`--no-log-routes`.

## Change

- Added `ZCUTLASS_VLLM_TRACE_CONFIG`.
- `tools/benchmark_vllm_serving_overlay.py --no-log-routes` now sets
  `ZCUTLASS_VLLM_TRACE_CONFIG=0`.
- `ZCutlassVllmLinearAdapter` skips `selected_gemm_config()` on zcutlass hits
  when trace config is disabled.
- Route logging runs keep `ZCUTLASS_VLLM_TRACE_CONFIG=1`, preserving kernel
  names, tile shape, pipeline, and epilogue fields for evidence collection.

## Single-Call Effect

Command:

```bash
ZCUTLASS_VLLM_TRACE_CONFIG=0 \
ZCUTLASS_EXPERIMENTAL_KERNELS=1 \
ZCUTLASS_EXPERIMENTAL_KERNEL=cpasync \
python tools/check_vllm_linear_method.py \
  --m 256 --n 2048 --k 1536 \
  --dtype bf16 --bias \
  --warmup 5 --iterations 20 --require-hit
```

Result:

- stock vLLM LinearMethod: `0.0280 ms`
- overlay with trace config skipped: `0.0401 ms`
- speedup vs stock: `0.699x`

This removed most metadata-query overhead but did not make the current N64
kernel faster than stock vLLM.

## Serving Effect

16-request prefill-heavy, output length 8, route logging disabled:

`reports/2026-05-23-vllm-serving-prefill-16-n64-traceskip-nolog/summary-prefill-heavy.jsonl`

| Metric | stock vLLM | overlay | Result |
| --- | ---: | ---: | --- |
| TTFT mean | 48.8881 ms | 46.2088 ms | 1.058x |
| TTFT p95 | 117.8652 ms | 103.8890 ms | 1.135x |
| TTFT p99 | 334.9878 ms | 281.5801 ms | 1.190x |
| TPOT mean | 9.2557 ms | 9.4261 ms | 0.982x |
| TPOT p95 | 10.3595 ms | 13.1027 ms | 0.791x |
| tokens/s | 388.8371 | 389.4394 | 1.002x |

64-request prefill-only, output length 1, concurrency 4:

`reports/2026-05-23-vllm-serving-prefill-64x384x1-c4-n64-traceskip-nolog/summary-prefill-heavy.jsonl`

| Metric | stock vLLM | overlay | Result |
| --- | ---: | ---: | --- |
| TTFT mean | 32.8230 ms | 34.5613 ms | 0.950x |
| TTFT p95 | 35.2693 ms | 43.1321 ms | 0.818x |
| tokens/s | 1537.3797 | 1537.5123 | 1.000x |

## Decision

Keep trace skipping because it is low-risk and reduces measurable adapter
overhead. It is not a substitute for a faster kernel. The current N64 cp.async
kernel remains below the stock vLLM/NVIDIA path, so v1.5 serving success is
still not achieved.
