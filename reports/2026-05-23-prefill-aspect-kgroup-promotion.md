# Prefill Aspect KGroup Promotion - 2026-05-23

## Scope

Promote an aspect-aware FP16 prefill dispatch rule:

- `N <= K`: use `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_n_le_k`
- `N > K`: keep `zcutlass_sm120_tensorop_f16_64x128x32_aligned_prefill`
- BF16: keep the existing `64x128x16` path

This is a CUTLASS-style tile/schedule bucket rule, not a full `(M,N,K)` shape
specialization.

## Validation

```bash
cmake --build build -j 24
ctest --test-dir build --output-on-failure

./build/zcutlass_bench \
  --suite llm-v1.5 \
  --dtype both \
  --providers zcutlass,cublas \
  --output reports/2026-05-23-llm-v15-aspect-kgroup-promoted-zcutlass-cublas.jsonl \
  --warmup 5 \
  --iterations 20

python3 tools/check_benchmark_regression.py \
  reports/2026-05-23-llm-v15-64x128x32-promoted-final-zcutlass-cublas.jsonl \
  reports/2026-05-23-llm-v15-aspect-kgroup-promoted-zcutlass-cublas.jsonl \
  --provider zcutlass \
  --shape-family prefill \
  --max-slowdown 1.05 \
  --min-geomean-speedup 0.98
```

## Benchmark Result

Prefill gate result versus the prior `64x128x32` promotion: geomean `1.0447x`.

| Shape | DType | Previous ms | New ms | Speedup | New kernel |
| --- | --- | ---: | ---: | ---: | --- |
| 128x4096x4096 | f16 | 0.2806 | 0.2518 | 1.1144x | `64x128x64_n_le_k` |
| 128x16384x4096 | f16 | 0.9823 | 0.9834 | 0.9989x | `64x128x32` |
| 128x4096x16384 | f16 | 1.8460 | 1.5851 | 1.1646x | `64x128x64_n_le_k` |
| 128x4096x4096 | bf16 | 0.2845 | 0.2834 | 1.0039x | unchanged |
| 128x16384x4096 | bf16 | 0.8807 | 0.8817 | 0.9989x | unchanged |
| 128x4096x16384 | bf16 | 1.6814 | 1.6813 | 1.0001x | unchanged |

The rejected all-prefill `64x128x64` experiment regressed `N > K` up-projection
from `0.9823 ms` to `1.5333 ms`. The promoted rule avoids that by only using
`KGroup=4` when `N <= K`.

## Nsight Summary

For `f16 128x4096x4096`, compared with the original 2026-05-22 baseline:

| Metric | Original 64x128x16 | KGroup=2 | KGroup=4 aspect |
| --- | ---: | ---: | ---: |
| SM throughput | 7.78% | 8.18% | 9.34% |
| Tensor active | 7.65% | 8.18% | 9.34% |
| DRAM throughput | 7.32% | 7.76% | 9.24% |
| Long scoreboard | 12.40 inst | 13.16 inst | 11.45 inst |
| Barrier stall | 0.86 inst | 0.73 inst | 0.61 inst |
| Registers / thread | 64 | 64 | 96 |
| Shared memory / block | 39.94 KB | 46.08 KB | 58.37 KB |

The KGroup=4 bucket is the stronger prefill direction for `N <= K`, but its
register footprint is higher. Future work should test whether this schedule can
be adapted to BF16 and whether the `N > K` case needs a different tile_N policy
instead of deeper K grouping.
