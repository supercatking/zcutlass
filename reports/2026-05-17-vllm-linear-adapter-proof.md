# vLLM Linear Adapter Routing Proof - 2026-05-17

## Result

The vLLM environment can now load the zcutlass plugin and route an explicit
LLM-style Linear callsite through `ZCutlassVllmLinearAdapter`.

This is the first vLLM-side execution proof beyond entry-point discovery:

- vLLM plugin loader calls `zcutlass_vllm.plugin:register`.
- `ZCutlassVllmLinearAdapter` runs inside the vLLM Python environment.
- FP16 and BF16 prefill-shaped Linear calls route to `torch.ops.zcutlass_torch.gemm`.
- Off-bucket shapes safely fall back to PyTorch with observable fallback reason.
- Outputs are checked against `torch.nn.functional.linear`.

## Commands

```bash
source /home/zyz/vllm/.venv/bin/activate
cd /home/zyz/zcutlass

python tools/check_vllm_linear_adapter.py \
  --m 128 --n 4096 --k 1024 \
  --dtype f16 \
  --allow-family prefill \
  --warmup 1 \
  --iterations 3 \
  --require-hit \
  --output reports/2026-05-17-vllm-linear-adapter-prefill-f16.jsonl

python tools/check_vllm_linear_adapter.py \
  --m 128 --n 4096 --k 1024 \
  --dtype bf16 \
  --allow-family prefill \
  --warmup 1 \
  --iterations 3 \
  --require-hit \
  --output reports/2026-05-17-vllm-linear-adapter-prefill-bf16.jsonl

python tools/check_vllm_linear_adapter.py \
  --m 8 --n 512 --k 512 \
  --dtype f16 \
  --allow-family prefill \
  --warmup 1 \
  --iterations 2 \
  --output reports/2026-05-17-vllm-linear-adapter-fallback-f16.jsonl
```

## Observations

| Case | Route | Hit rate | Median adapter time | Median stock time | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `f16 128x4096x1024` | zcutlass prefill | 1.00 | 0.1402 ms | 0.0527 ms | correct, slower |
| `bf16 128x4096x1024` | zcutlass prefill | 1.00 | 0.0885 ms | 0.0406 ms | correct, slower |
| `f16 8x512x512` | PyTorch fallback | 0.00 | 0.0896 ms | 0.0558 ms | correct fallback |

The prefill cases prove that vLLM can use the zcutlass adapter path, but they
also show the current kernel is not yet competitive with PyTorch/cuBLAS for
these callsites. This is expected for the current WMMA baseline and keeps the
next optimization target honest.

## Next Gate

The next milestone is a real vLLM model/worker experiment:

1. Add an explicit experimental callsite that uses `ZCutlassVllmLinearAdapter`
   without global monkey-patching or cuBLAS hooks.
2. Collect per-callsite hit/miss/fallback telemetry during vLLM execution.
3. Run stock vLLM vs vLLM+zcutlass on fixed decode-heavy, prefill-heavy, and
   mixed workloads.
4. Only claim product value after TTFT or TPOT improves without p95/p99
   regression and the microbenchmark/Nsight evidence explains the result.
