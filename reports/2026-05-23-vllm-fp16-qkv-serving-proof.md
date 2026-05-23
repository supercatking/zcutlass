# 2026-05-23 vLLM FP16 QKV Serving Proof

## Result

This is the first scoped v1.5 serving win for zcutlass.

Configuration:

- GPU: RTX 5080 / SM120
- model: local Qwen2.5-1.5B-Instruct
- engine: vLLM stock vs vLLM + zcutlass overlay
- dtype: FP16
- workload: prefill-heavy random workload, input length 384, output length 1,
  64 prompts, request rate 4, max concurrency 4
- overlay policy: `ZCUTLASS_VLLM_PROFIT_POLICY=measured`
- promoted route: QKV only, `M=256,N=2048,K=1536,bias=true`
- selected kernel:
  `zcutlass_sm120_mma_f16_64x64x64_prefill_smem_ldm_cpasync_warp32x32_reg_epilogue`

## Repeated Serving Evidence

Run 1:

`reports/2026-05-23-vllm-serving-fp16-prefill-default-measured-nolog/summary-prefill-heavy.jsonl`

| Metric | stock vLLM | zcutlass overlay | Speedup |
| --- | ---: | ---: | ---: |
| TTFT mean | 31.1211 ms | 29.6221 ms | 1.051x |
| TTFT p50 | 24.9808 ms | 25.0712 ms | 0.996x |
| TTFT p95 | 32.1097 ms | 26.5185 ms | 1.211x |
| TTFT p99 | 178.7618 ms | 138.5647 ms | 1.290x |
| tokens/s | 1537.3772 | 1537.5939 | 1.000x |

Run 2:

`reports/2026-05-23-vllm-serving-fp16-prefill-default-measured-nolog-rerun1/summary-prefill-heavy.jsonl`

| Metric | stock vLLM | zcutlass overlay | Speedup |
| --- | ---: | ---: | ---: |
| TTFT mean | 30.7351 ms | 29.7847 ms | 1.032x |
| TTFT p50 | 25.1337 ms | 25.0891 ms | 1.002x |
| TTFT p95 | 32.4131 ms | 26.5886 ms | 1.219x |
| TTFT p99 | 162.6792 ms | 139.1038 ms | 1.169x |
| tokens/s | 1537.5264 | 1537.5663 | 1.000x |

Both post-fix runs meet the v1.5 serving gate for at least one workload:
TTFT mean improves by at least 3%, and p95/p99 do not regress.

## Route Evidence

Route evidence:

`reports/2026-05-23-vllm-serving-fp16-prefill-default-measured-routes/routes-overlay-prefill-heavy.jsonl`

Summary:

- rows: 532
- zcutlass hits: 28
- fallback calls: 504
- zcutlass route shape: `M=256,N=2048,K=1536,torch.float16,bias=true`
- fallback shapes:
  - `M=384,N=2048,K=1536`: `shape_not_target_bucket`
  - `M=2048,N=2048,K=1536`: `family_not_promoted`
- logged layers: `qkv_proj` only

This proves the measured policy is not globally replacing vLLM GEMM. It routes
only the currently measured profitable FP16 QKV prefill chunk and leaves
unsupported shapes on the stock vLLM path.

## Product-Safety Change

The vLLM overlay now supports `ZCUTLASS_VLLM_PROFIT_POLICY`.

- `measured`: route only locally measured profitable serving callsites.
- `off`: keep previous family-only routing behavior for experiments.
- `all` / `experimental`: bypass measured-profit filtering.

`tools/benchmark_vllm_serving_overlay.py` defaults to `measured` for overlay
serving runs. The OOT vLLM Linear registration skips non-promoted layers under
the measured policy, so BF16, MLP, O projection, large, decode, and off-bucket
paths avoid zcutlass wrapper overhead where possible.

## Negative Evidence Kept

These results are intentionally not generalized:

- BF16 QKV remains slower than stock vLLM for the measured Qwen callsite.
- FP16 MLP up/down remain slower than stock vLLM.
- FP16 O projection without bias is slower than stock vLLM.
- A previous qkv+o measured run regressed before the policy was narrowed to
  QKV-only install.

## Decision

The v1.5 product proof is now true for one scoped serving workload:

FP16 Qwen2.5-1.5B prefill-heavy serving on RTX 5080, with zcutlass routed only
for the measured QKV prefill chunk, beats stock vLLM on TTFT mean and tail
latency across two repeated runs.

The broader project is not complete. Next targets are:

- make the QKV win larger and less sensitive to noise;
- promote BF16 only after single-call and serving evidence is positive;
- build a faster MLP/down-proj path;
- add decode/TPOT-specific kernels.
