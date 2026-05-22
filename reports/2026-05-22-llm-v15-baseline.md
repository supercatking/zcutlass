# zcutlass v1.5 LLM GEMM Baseline - 2026-05-22

## Scope

This report starts Batch 1 from `docs/v15-long-term-execution-plan.md`.

Measured target:

- GPU: NVIDIA GeForce RTX 5080 / SM120.
- zcutlass benchmark: `./build/zcutlass_bench`.
- Suite: `llm-v1.5`.
- Dtypes: FP16 and BF16.
- Warmup: 5 iterations.
- Timed iterations: 20 iterations.
- Providers: zcutlass and cuBLAS.
- External comparison: official CUTLASS profiler from
  `/home/zyz/cutlass-official/build-profiler/tools/profiler/cutlass_profiler`.

Validation:

- `ctest --test-dir build --output-on-failure` passed.
- `cmake --build build -j 24` reported no rebuild required.
- Nsight Compute counter collection succeeded for the initial prefill/decode
  profiles.

## Artifacts

- zcutlass/cuBLAS JSONL:
  `reports/2026-05-22-llm-v15-zcutlass-cublas.jsonl`
- CUTLASS profiler JSONL:
  `reports/2026-05-22-llm-v15-cutlass.jsonl`
- Combined visualization JSONL:
  `reports/2026-05-22-llm-v15-baseline-combined.jsonl`
- HTML chart:
  `reports/2026-05-22-llm-v15-baseline.html`
- NCU summaries:
  - `reports/profiles/2026-05-22-f16-128x4096x4096.summary.md`
  - `reports/profiles/2026-05-22-bf16-128x4096x4096.summary.md`
  - `reports/profiles/2026-05-22-f16-8x4096x4096.summary.md`
  - `reports/profiles/2026-05-22-bf16-8x4096x4096.summary.md`

## zcutlass vs cuBLAS

| Shape | Family | zcutlass ms | cuBLAS ms | zcutlass / cuBLAS | zcutlass TFLOP/s | cuBLAS TFLOP/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| f16 8x4096x4096 | decode | 0.3011 | 0.0308 | 0.102x | 0.89 | 8.71 |
| bf16 8x4096x4096 | decode | 0.2944 | 0.0320 | 0.109x | 0.91 | 8.39 |
| f16 128x4096x4096 | prefill | 0.3009 | 0.0537 | 0.179x | 14.28 | 79.94 |
| bf16 128x4096x4096 | prefill | 0.2832 | 0.0559 | 0.197x | 15.16 | 76.87 |
| f16 128x16384x4096 | prefill / MLP up | 1.0175 | 0.2030 | 0.200x | 16.88 | 84.63 |
| bf16 128x16384x4096 | prefill / MLP up | 0.8902 | 0.2179 | 0.245x | 19.30 | 78.84 |
| f16 128x4096x16384 | prefill / MLP down | 1.9459 | 0.1914 | 0.098x | 8.83 | 89.76 |
| bf16 128x4096x16384 | prefill / MLP down | 1.6824 | 0.1936 | 0.115x | 10.21 | 88.75 |
| f16 4096x4096x4096 | large | 4.6473 | 1.2553 | 0.270x | 29.57 | 109.49 |
| bf16 4096x4096x4096 | large | 4.1886 | 1.2518 | 0.299x | 32.81 | 109.79 |

Primary conclusion: zcutlass is not yet competitive with cuBLAS on any LLM
v1.5 canonical shape. The largest immediate gap is the MLP down shape
`128x4096x16384`, where zcutlass is roughly 10 percent of cuBLAS speed.

## zcutlass vs CUTLASS profiler

The CUTLASS profiler comparison is useful for design alignment, but it is not
as strong as the cuBLAS baseline in this run. The current profiler path still
has layout and kernel-enumeration caveats for exact row-major A/B/C/D parity.

Observed CUTLASS profiler winners:

- BF16 CUTLASS profiler is much faster than zcutlass across the measured suite.
- FP16 CUTLASS profiler is faster on MLP up/down and large cases.
- zcutlass is faster than the selected FP16 CUTLASS profiler kernel on the
  smallest decode/prefill cases, but this does not change the main conclusion
  because cuBLAS is much faster for the same shapes.

## Nsight Compute Summary

| Shape | SM % | Tensor % | DRAM % | Achieved occupancy | Regs/thread | Grid | Top stall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| f16 128x4096x4096 | 7.78 | 7.65 | 7.32 | 16.67% | 64 | 64 | long scoreboard 12.4 |
| bf16 128x4096x4096 | 13.82 | 9.16 | 8.44 | 16.66% | 72 | 64 | long scoreboard 8.73 |
| f16 8x4096x4096 | 5.22 | 4.19 | 8.00 | 16.67% | 74 | 32 | long scoreboard 6.17 |
| bf16 8x4096x4096 | 6.38 | 4.14 | 7.71 | 16.67% | 96 | 32 | long scoreboard 6.58 |

Interpretation:

- Tensor Core utilization is very low on both prefill and decode shapes.
- SM throughput is also low, so this is not an epilogue-only issue.
- DRAM throughput is low, which suggests the kernel is not issuing enough work
  to saturate memory or Tensor Cores.
- Long scoreboard is the dominant stall in all four initial profiles.
- Occupancy is low and the decode grid has only 32 CTAs, which makes the
  current `64x128x16` baseline a poor fit for small-M decode.

## Engineering Decision

Do not expand vLLM or SGLang integration before improving the kernel path.
The framework path is now proven, but routing more real callsites to this kernel
would route them to a slower implementation.

The next kernel experiment should target prefill first:

1. Keep the current WMMA baseline as fallback.
2. Add a new prefill candidate instead of replacing the existing kernel.
3. Use a CUTLASS-style reusable tile family, not full-shape specialization.
4. Start with double-buffered K-loop staging for `128x4096x4096`.
5. Record whether long scoreboard drops and Tensor Core utilization rises.
6. Only then test larger MLP up/down shapes.

Recommended first variant:

- family: prefill
- tile candidates: `128x128x64` and `128x256x64`
- target dtype: FP16 first, BF16 immediately after
- target shape: `128x4096x4096`
- fallback: existing `64x128x16` WMMA kernel

Promotion gate:

- correctness passes;
- zcutlass improves over the current baseline;
- NCU shows higher Tensor utilization or a clear reduction in scoreboard stalls;
- no fallback correctness regression.
