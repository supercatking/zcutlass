# zcutlass M0 Baseline Report

## Metadata

- timestamp: `2026-05-23T05:12:40.336442+00:00`
- commit: `77fa3871f137965f69b56868b0f8ed96a074af0d`
- branch: `main`
- dirty_status: `M AGENTS.md;  M README.md;  M docs/measurement-workflow.md;  M docs/v15-long-term-execution-plan.md;  M reports/2026-05-17-vllm-linear-adapter-fallback-f16.jsonl;  M reports/2026-05-17-vllm-linear-adapter-prefill-bf16.jsonl;  M reports/2026-05-17-vllm-linear-adapter-prefill-f16.jsonl; ?? docs/explicit-mma-prefill-design.md; ?? reports/2026-05-23-m0-baseline/; ?? reports/manual-prefill-module-zcutlass.jsonl; ?? tools/run_m0_baseline.py`
- cuda_nvcc: `Build cuda_13.1.r13.1/compiler.37061995_0`
- nvidia_smi: `NVIDIA GeForce RTX 5080, 596.36`
- output_dir: `/home/zyz/zcutlass/reports/2026-05-23-m0-baseline`

## Step Results

- PASS `cmake_build` exit=0 stdout=`reports/2026-05-23-m0-baseline/cmake_build.stdout.txt` stderr=`reports/2026-05-23-m0-baseline/cmake_build.stderr.txt`
- PASS `ctest` exit=0 stdout=`reports/2026-05-23-m0-baseline/ctest.stdout.txt` stderr=`reports/2026-05-23-m0-baseline/ctest.stderr.txt`
- PASS `kernel_llm_v15` exit=0 stdout=`reports/2026-05-23-m0-baseline/kernel_llm_v15.stdout.txt` stderr=`reports/2026-05-23-m0-baseline/kernel_llm_v15.stderr.txt`
- PASS `torch_overlay_check` exit=0 stdout=`reports/2026-05-23-m0-baseline/torch_overlay_check.stdout.txt` stderr=`reports/2026-05-23-m0-baseline/torch_overlay_check.stderr.txt`
- PASS `vllm_plugin_check` exit=0 stdout=`reports/2026-05-23-m0-baseline/vllm_plugin_check.stdout.txt` stderr=`reports/2026-05-23-m0-baseline/vllm_plugin_check.stderr.txt`
- PASS `vllm_linear_method_smoke` exit=0 stdout=`reports/2026-05-23-m0-baseline/vllm_linear_method_smoke.stdout.txt` stderr=`reports/2026-05-23-m0-baseline/vllm_linear_method_smoke.stderr.txt`

## Required Gate

All required M0 checks passed.

## Kernel Baseline Summary

| DType | Shape | Kernel | Path | ms | TFLOP/s |
| --- | --- | --- | --- | ---: | ---: |
| f16 | 8x4096x4096 | `zcutlass_sm120_tensorop_f16_64x128x16` | fallback | 0.3046 | 0.8812 |
| bf16 | 8x4096x4096 | `zcutlass_sm120_tensorop_bf16_64x128x16` | fallback | 0.2938 | 0.9137 |
| f16 | 128x4096x4096 | `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_n_le_k` | fast | 0.2521 | 17.0349 |
| bf16 | 128x4096x4096 | `zcutlass_sm120_tensorop_bf16_64x128x32_aligned_prefill` | fast | 0.2555 | 16.8087 |
| f16 | 128x16384x4096 | `zcutlass_sm120_tensorop_f16_64x128x32_aligned_prefill` | fast | 1.0068 | 17.0646 |
| bf16 | 128x16384x4096 | `zcutlass_sm120_tensorop_bf16_64x128x32_aligned_prefill` | fast | 0.8307 | 20.6815 |
| f16 | 128x4096x16384 | `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_n_le_k` | fast | 1.5911 | 10.7972 |
| bf16 | 128x4096x16384 | `zcutlass_sm120_tensorop_bf16_64x128x32_aligned_prefill` | fast | 1.5845 | 10.8422 |
| f16 | 4096x4096x4096 | `zcutlass_sm120_tensorop_f16_64x128x16_aligned` | fast | 4.6391 | 29.6262 |
| bf16 | 4096x4096x4096 | `zcutlass_sm120_tensorop_bf16_64x128x16_aligned` | fast | 4.1932 | 32.7770 |

## vLLM LinearMethod Summary

| DType | Case | Shape | Hit rate | Kernel | Speedup vs stock |
| --- | --- | --- | ---: | --- | ---: |
| f16 | smoke_decode | 8x512x512 | 0.00 | `vllm_unquantized_fallback` | 0.922x |
| f16 | smoke_prefill | 128x4096x1024 | 1.00 | `zcutlass_sm120_tensorop_f16_64x128x32_aligned_prefill` | 0.301x |
| f16 | smoke_prefill_bf16_target | 128x4096x4096 | 1.00 | `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_n_le_k` | 0.201x |
| bf16 | smoke_decode | 8x512x512 | 0.00 | `vllm_unquantized_fallback` | 0.600x |
| bf16 | smoke_prefill | 128x4096x1024 | 1.00 | `zcutlass_sm120_tensorop_bf16_64x128x32_aligned_prefill` | 0.402x |
| bf16 | smoke_prefill_bf16_target | 128x4096x4096 | 1.00 | `zcutlass_sm120_tensorop_bf16_64x128x32_aligned_prefill` | 0.213x |

## Notes

- This report freezes the current baseline before explicit-MMA work.
- vLLM LinearMethod results prove routing and telemetry only; they are not serving-level value claims.
- Existing unrelated dirty/manual report files are intentionally left untouched.
