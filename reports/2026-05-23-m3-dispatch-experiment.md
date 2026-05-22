# zcutlass M3 Dispatch Experiment - 2026-05-23

## Scope

This run tested a narrower low-M `32x128x16` prefill dispatch candidate for the
LLM v1.5 canonical suite, then restored the default `64x128x16` route after the
candidate regressed the default path. The retained code change is a dispatch
test guard that requires the Large family to stay on the fast aligned path.

## Validation

- Build: `cmake --build build -j 24`
- Correctness: `ctest --test-dir build --output-on-failure`
- Benchmark, rejected candidate:
  `reports/2026-05-23-m3-rejected-lowm-dispatch-zcutlass-cublas.jsonl`
- Benchmark, retained dispatch:
  `reports/2026-05-23-m3-fastpath-guard-zcutlass-cublas.jsonl`

## Result

The low-M `32x128x16` candidate should not be promoted to default dispatch.

| Shape | DType | Rejected candidate | Retained dispatch | Decision |
| --- | --- | ---: | ---: | --- |
| 8x4096x4096 | f16 | 0.3599 ms | 0.3034 ms | Reject candidate |
| 8x4096x4096 | bf16 | 0.4023 ms | 0.2953 ms | Reject candidate |
| 128x4096x4096 | f16 | 0.3517 ms | 0.3025 ms | Reject candidate |
| 128x4096x4096 | bf16 | 0.4137 ms | 0.2837 ms | Reject candidate |
| 128x4096x16384 | f16 | 3.2522 ms | 1.9484 ms | Reject candidate |
| 128x4096x16384 | bf16 | 3.7181 ms | 1.6589 ms | Reject candidate |

The candidate increased CTA count for some low-M cases, but the extra blocks did
not compensate for poorer per-CTA efficiency. It also misrouted MLP down-proj
shapes into the low-M candidate, which caused the largest regression.

## Additional Finding

An attempted fast-path epilogue simplification was discarded before commit:
CUDA WMMA cannot store a FP32 accumulator fragment directly to FP16/BF16 global
memory. The current epilogue must either keep the FP32 shared-memory staging
path or move to a custom MMA/epilogue implementation where accumulator element
layout is explicitly controlled.

## Follow-Up

- Keep `64x128x16_aligned` as the default fast path for prefill and large
  canonical shapes.
- Do not use manifest order alone to promote new tile candidates; add an
  opt-in experiment mechanism or a benchmark gate before default dispatch.
- Next kernel work should focus on mainloop scheduling and epilogue ownership,
  not merely smaller block-M tiling.
