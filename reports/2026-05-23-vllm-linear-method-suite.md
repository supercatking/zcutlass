# 2026-05-23 vLLM LinearMethod benchmark harness

## Goal

Move vLLM validation from one-off probes to a repeatable suite that compares
stock vLLM `UnquantizedLinearMethod` with the explicit zcutlass LinearMethod
wrapper and records kernel-level routing metadata.

## New Tool

`tools/benchmark_vllm_linear_method.py`

- Runs stock vLLM unquantized Linear and zcutlass overlay for each case.
- Uses the vLLM plugin loader and `ZCutlassUnquantizedLinearMethod`.
- Writes schema-v1 JSONL for both providers.
- Records hit/miss counts, fallback reasons, promoted families, and
  `last_trace.selected_config.kernel_name`.

## Validation Command

```bash
cd /home/zyz/zcutlass
source /home/zyz/vllm/.venv/bin/activate

python tools/benchmark_vllm_linear_method.py \
  --suite smoke \
  --dtype both \
  --allow-family prefill \
  --materialize-inputs \
  --warmup 2 \
  --iterations 5 \
  --output reports/2026-05-23-vllm-linear-method-smoke-after-bf16-promotion.jsonl \
  --summary
```

## Result

The harness correctly distinguishes fallback decode cases from promoted prefill
hits:

| Case | DType | Route | Kernel | Speedup vs stock |
| --- | --- | --- | --- | ---: |
| `smoke_decode` | f16 | fallback | `vllm_unquantized_fallback` | 0.362x |
| `smoke_prefill` | f16 | hit | `zcutlass_sm120_tensorop_f16_64x128x32_aligned_prefill` | 0.294x |
| `smoke_prefill_bf16_target` | f16 | hit | `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_n_le_k` | 0.203x |
| `smoke_decode` | bf16 | fallback | `vllm_unquantized_fallback` | 0.544x |
| `smoke_prefill` | bf16 | hit | `zcutlass_sm120_tensorop_bf16_64x128x32_aligned_prefill` | 0.453x |
| `smoke_prefill_bf16_target` | bf16 | hit | `zcutlass_sm120_tensorop_bf16_64x128x32_aligned_prefill` | 0.218x |

This confirms vLLM can route to the promoted BF16 kernel and that fallback
telemetry remains observable. It also confirms that zcutlass is not yet faster
than stock vLLM GEMM at the framework layer.

## Kernel Constraint Found

The obvious WMMA fast-path cleanup, direct global store from accumulator
fragments to FP16/BF16 D, is blocked by the WMMA API: accumulator fragments store
as their accumulator element type. Because zcutlass keeps FP32 accumulation,
the current WMMA path must spill accumulator fragments to shared memory before
output conversion. Removing that cost requires an explicit MMA/register epilogue
kernel family.
