# Prefill 128x128 Experimental Candidate - 2026-05-23

## Scope

This run adds an opt-in FP16 `128x128x16` WMMA prefill candidate. It is guarded
by `ZCUTLASS_EXPERIMENTAL_KERNELS=1`, exposed through
`zcutlass_bench --experimental-kernels`, and is not used by default dispatch.

## Validation

```bash
cmake --build build -j 24
ctest --test-dir build --output-on-failure

./build/zcutlass_bench \
  --m 128 --n 4096 --k 4096 \
  --dtype f16 \
  --providers zcutlass,cublas \
  --output reports/2026-05-23-prefill-128x128-default-f16.jsonl \
  --warmup 5 \
  --iterations 20

./build/zcutlass_bench \
  --m 128 --n 4096 --k 4096 \
  --dtype f16 \
  --providers zcutlass,cublas \
  --experimental-kernels \
  --output reports/2026-05-23-prefill-128x128-experimental-f16.jsonl \
  --warmup 5 \
  --iterations 20
```

## Result

| Path | Kernel | zcutlass ms | cuBLAS ms | zcutlass TFLOP/s | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Default | `zcutlass_sm120_tensorop_f16_64x128x16_aligned` | 0.3030 | 0.0550 | 14.17 | Keep default |
| Experimental | `zcutlass_sm120_tensorop_f16_128x128x16_aligned_prefill_experimental` | 0.3474 | 0.0569 | 12.36 | Do not promote |

The regression gate reports a 0.8722x baseline/candidate speed ratio, so the
candidate is slower by 1.1465x. This confirms that increasing CTA tile M to 128
without changing the K mainloop schedule does not address the current bottleneck.

## Next Step

Keep this candidate available only as an explicit experiment. The next useful
kernel step is not another tile-only dispatch change; it is a staged or
double-buffered K mainloop for the prefill family, with the regression gate used
before any default promotion.
