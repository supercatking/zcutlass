# RTX 5080 SM120 GEMM vs Official NVIDIA CUTLASS

Date: 2026-05-16

GPU: NVIDIA GeForce RTX 5080, driver 596.36, CUDA 13.1

zcutlass repo: `/home/zyz/zcutlass`

CUTLASS repo: `/home/zyz/cutlass-official`, commit `e406c186f510a15091cce01f782020ceb7ba8eb5`, CUTLASS 4.5.0

Profiler: `/home/zyz/cutlass-official/build-profiler/tools/profiler/cutlass_profiler`

Command:

```bash
python3 tools/visualize_gemm_comparison.py \
  --shape 1x1024x1024 \
  --shape 8x2048x2048 \
  --shape 64x1024x1024 \
  --shape 128x2048x2048 \
  --dtype both \
  --warmup 3 \
  --iterations 10 \
  --cutlass-profiler /home/zyz/cutlass-official/build-profiler/tools/profiler/cutlass_profiler \
  --output build/reports/gemm_vs_cutlass_real.html \
  --save-jsonl build/reports/gemm_vs_cutlass_real.jsonl \
  --title "zcutlass real GEMM vs official NVIDIA CUTLASS"
```

Artifacts:

- `reports/2026-05-16-sm120-gemm-vs-cutlass.html`
- `reports/2026-05-16-sm120-gemm-vs-cutlass.jsonl`

Summary:

| dtype | M | N | K | zcutlass ms | CUTLASS ms | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| f16 | 8 | 2048 | 2048 | 0.1570 | 0.1917 | 1.221x |
| f16 | 64 | 1024 | 1024 | 0.0796 | 0.0997 | 1.253x |
| f16 | 128 | 2048 | 2048 | 0.1581 | 0.1926 | 1.218x |
| bf16 | 8 | 2048 | 2048 | 0.1541 | 0.0992 | 0.643x |
| bf16 | 64 | 1024 | 1024 | 0.0759 | 0.0533 | 0.702x |
| bf16 | 128 | 2048 | 2048 | 0.1595 | 0.1021 | 0.640x |

Notes:

- CUTLASS profiler did not emit a baseline row for `M=1,N=1024,K=1024` with the available f16/bf16 row-row tensorop kernels, so those two zcutlass measurements are present in the JSONL/HTML with missing CUTLASS values.
- The current official CUTLASS profiler build exposes row-major A/B f16/bf16 instances with column-major C/D for these filters. zcutlass measurements remain the v1 row-major A/B/C/D API.
- An attempted larger smoke run including `64x4096x4096` and `256x2048x8192` caused WSL to return `Wsl/Service/E_UNEXPECTED`; WSL recovered after `wsl --shutdown`.
