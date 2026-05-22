# vLLM LinearMethod zcutlass Routing Proof - 2026-05-23

## Scope

This proof adds an opt-in vLLM `UnquantizedLinearMethod` wrapper. It does not
patch vLLM globally. A selected vLLM Linear layer can replace its `quant_method`
with `ZCutlassUnquantizedLinearMethod`; eligible shapes route through zcutlass,
and misses delegate back to vLLM's native unquantized GEMM path.

## Validation Commands

```bash
source /home/zyz/vllm/.venv/bin/activate
cd /home/zyz/zcutlass

python tools/check_vllm_linear_method.py \
  --m 128 --n 4096 --k 1024 \
  --dtype f16 \
  --allow-family prefill \
  --require-hit \
  --warmup 1 \
  --iterations 3 \
  --output reports/2026-05-23-vllm-linear-method-prefill-f16.jsonl

python tools/check_vllm_linear_method.py \
  --m 8 --n 512 --k 512 \
  --dtype f16 \
  --allow-family prefill \
  --require-fallback \
  --warmup 1 \
  --iterations 3 \
  --output reports/2026-05-23-vllm-linear-method-fallback-f16.jsonl
```

## Result

| Case | Route | Hit rate | Stock ms | Overlay ms | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `128x4096x1024 f16` | zcutlass | 1.00 | 0.0364 | 0.1158 | Correct, but slower than vLLM native |
| `8x512x512 f16` | vLLM fallback | 0.00 | 0.0229 | 0.0267 | Correct fallback, `shape_not_target_bucket` |

This validates the vLLM Linear abstraction hook, not product performance. The
prefill route is intentionally observable and currently slower, so broader vLLM
serving replacement remains blocked on kernel performance.

## Implementation Notes

- `ZCutlassVllmLinearAdapter.run(..., fallback_fn=...)` preserves framework
  fallback instead of using a generic PyTorch matmul fallback.
- `ZCutlassUnquantizedLinearMethod` delegates weight creation and miss handling
  to vLLM's `UnquantizedLinearMethod`.
- `install_zcutlass_unquantized_linear_method(layer, ...)` is explicit and only
  wraps unquantized Linear layers.
- Transposed `[K, N]` weights are cached on the layer for the zcutlass route and
  invalidated after `process_weights_after_loading`.

## Next Gate

Do not wire this into a full model by default until at least one LLM GEMM bucket
beats vLLM/cuBLAS at the microbenchmark level. The wrapper is now ready for a
targeted callsite experiment once kernel performance justifies promotion.
