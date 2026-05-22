# Prefill 64x128x32 Promotion - 2026-05-23

## Scope

Promote the FP16 prefill `64x128x32` WMMA candidate into default dispatch for
the prefill shape family. This is a family-level specialization, not a full
shape-specific kernel. BF16 and non-prefill shapes remain on their previous
paths.

## Implementation

- `KGroup=2` groups two `16`-wide K slices into one mainloop iteration.
- The promoted operation is:
  `zcutlass_sm120_tensorop_f16_64x128x32_aligned_prefill`.
- The operation is restricted to `ShapeFamily::Prefill`, so decode and large
  shapes do not route to it.
- The original KGroup=1 mainloop is preserved with `if constexpr (KGroup == 1)`
  to avoid BF16/default codegen regressions.

## Validation

```bash
cmake --build build -j 24
ctest --test-dir build --output-on-failure

./build/zcutlass_bench \
  --suite llm-v1.5 \
  --dtype both \
  --providers zcutlass,cublas \
  --output reports/2026-05-23-llm-v15-64x128x32-promoted-final-zcutlass-cublas.jsonl \
  --warmup 5 \
  --iterations 20

python3 tools/check_benchmark_regression.py \
  reports/2026-05-23-m3-fastpath-guard-zcutlass-cublas.jsonl \
  reports/2026-05-23-llm-v15-64x128x32-promoted-final-zcutlass-cublas.jsonl \
  --provider zcutlass \
  --shape-family prefill \
  --max-slowdown 1.05 \
  --min-geomean-speedup 0.98
```

## Benchmark Result

Prefill gate result: geomean `1.0265x` versus the previous default dispatch.

| Shape | DType | Previous ms | New ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| 128x4096x4096 | f16 | 0.3025 | 0.2806 | 1.0780x |
| 128x16384x4096 | f16 | 1.0241 | 0.9823 | 1.0426x |
| 128x4096x16384 | f16 | 1.9484 | 1.8460 | 1.0555x |
| 128x4096x4096 | bf16 | 0.2837 | 0.2845 | 0.9972x |
| 128x16384x4096 | bf16 | 0.8830 | 0.8807 | 1.0026x |
| 128x4096x16384 | bf16 | 1.6589 | 1.6814 | 0.9866x |

The BF16 path stays within the 1.05x per-shape slowdown gate after preserving
the original KGroup=1 code path.

## Nsight Summary

Compared with the 2026-05-22 f16 prefill baseline:

| Metric | Previous 64x128x16 | New 64x128x32 |
| --- | ---: | ---: |
| SM throughput | 7.78% | 8.18% |
| Tensor active | 7.65% | 8.18% |
| DRAM throughput | 7.32% | 7.76% |
| Barrier stall | 0.86 inst | 0.73 inst |
| Shared memory / block | 39.94 KB | 46.08 KB |
| Registers / thread | 64 | 64 |

The result is a real but small first kernel win. Long scoreboard remains the
dominant stall, so the next kernel step should continue toward async/staged
global-to-shared movement or a lower-overhead explicit MMA epilogue.
