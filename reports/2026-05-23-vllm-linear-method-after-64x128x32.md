# vLLM LinearMethod After 64x128x32 Promotion - 2026-05-23

## Scope

Rebuilt the zcutlass Python extension inside `/home/zyz/vllm/.venv` after the
f16 prefill `64x128x32` kernel promotion, then reran the vLLM
`UnquantizedLinearMethod` probe.

The vLLM environment uses PyTorch `2.11.0+cu129`, while the system `nvcc` is
CUDA 13.1. The rebuild required the explicit local-development override:

```bash
ZCUTLASS_ALLOW_CUDA_MISMATCH=1 MAX_JOBS=16 \
  python -m pip install -e ./python --no-build-isolation
```

## Validation Commands

```bash
source /home/zyz/vllm/.venv/bin/activate
cd /home/zyz/zcutlass

python tools/check_vllm_linear_method.py \
  --m 128 --n 4096 --k 1024 \
  --dtype f16 \
  --allow-family prefill \
  --require-hit \
  --warmup 5 \
  --iterations 20 \
  --output reports/2026-05-23-vllm-linear-method-prefill-f16-after-64x128x32.jsonl

python tools/check_vllm_linear_method.py \
  --m 8 --n 512 --k 512 \
  --dtype f16 \
  --allow-family prefill \
  --require-fallback \
  --warmup 2 \
  --iterations 5 \
  --output reports/2026-05-23-vllm-linear-method-fallback-f16-after-64x128x32.jsonl
```

## Result

| Case | Route | Hit rate | Stock ms | Overlay ms | Speedup vs stock |
| --- | --- | ---: | ---: | ---: | ---: |
| `128x4096x1024 f16` | zcutlass | 1.00 | 0.0220 | 0.0756 | 0.2904x |
| `8x512x512 f16` | vLLM fallback | 0.00 | 0.0113 | 0.0380 | 0.2969x |

The promoted kernel reduced the zcutlass overlay latency for the prefill probe
compared with the earlier `0.1158 ms` LinearMethod result, but the route is still
slower than vLLM native GEMM. This keeps full vLLM serving promotion blocked on
more kernel and adapter overhead work.

## Interpretation

- vLLM can now route a real `UnquantizedLinearMethod` callsite through zcutlass.
- Fallback remains observable and delegates to vLLM native unquantized GEMM.
- Product-value proof is not met yet: zcutlass hits are correct but slower than
  stock vLLM on this probe.
