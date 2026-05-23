# 2026-05-23 N64 cp.async Prefill Experiment

## Scope

This experiment tested a `64x64x64` explicit-MMA prefill variant for SM120. The
goal was to improve Qwen2.5-1.5B shapes where `N` is relatively small or where
more N-direction CTAs may help occupancy without reducing M-tile reuse.

The variant is experimental and selected with:

```bash
ZCUTLASS_EXPERIMENTAL_KERNELS=1
ZCUTLASS_EXPERIMENTAL_KERNEL=64x64x64
```

After measurement it became the preferred operation for the broad
`ZCUTLASS_EXPERIMENTAL_KERNEL=cpasync` filter. Normal dispatch remains
unchanged.

## Verification

- `cmake --build build -j 24`: passed.
- `ctest --test-dir build --output-on-failure`: passed.
- `compute-sanitizer --error-exitcode 99 ./build/zcutlass_tests`: passed with
  `ERROR SUMMARY: 0 errors`.
- `cuobjdump --dump-resource-usage build/libzcutlass.a`: N64 cp.async kernels
  use `REG:128`, `STACK:0`, and dynamic shared memory.

The correctness suite includes exact-filter FP16 and BF16 N64 prefill smoke
cases.

## Microbenchmark Results

All timings are CUDA-event medians on local RTX 5080 / SM120, row-major A/B/D,
`alpha=1`, `beta=0`, no bias.

| Shape | DType | Kernel | zcutlass ms | cuBLAS ms | Speedup |
| --- | --- | --- | ---: | ---: | ---: |
| `256x1536x1536` | BF16 | `64x64x64 cp.async` | `0.0382` | `0.0264` | `0.691x` |
| `256x17920x1536` | BF16 | `64x64x64 cp.async` | `0.2474` | `0.1478` | `0.598x` |
| `256x1536x8960` | BF16 | `64x64x64 cp.async` | `0.1839` | `0.0748` | `0.407x` |
| `128x4096x4096` | BF16 | `64x64x64 cp.async` | `0.0887` | `0.0547` | `0.617x` |
| `128x4096x4096` | FP16 | `64x64x64 cp.async` | `0.0893` | `0.0538` | `0.602x` |
| `128x4096x4096` | FP16 | `64x128x64 cp.async` | `0.0955` | `0.0557` | `0.584x` |

## vLLM Smoke

The Qwen2.5-1.5B real-model route smoke confirmed the broad `cpasync` route now
selects N64 for prefill hits:

- route rows: `448`.
- zcutlass hits: `224`.
- fallbacks: `224`.
- N64 explicit-MMA hits: `84`.
- WMMA bias/small-M hits: `140`.

The bounded serving smoke with input length `384`, output length `8`, four
prompts, request rate `1`, max concurrency `1` did not establish product value:

| Provider | TTFT mean ms | TPOT mean ms | Tokens/s |
| --- | ---: | ---: | ---: |
| stock vLLM | `142.02` | `10.98` | `382.39` |
| vLLM + zcutlass overlay | `137.10` | `14.21` | `377.86` |

## Decision

Keep `64x64x64 cp.async` as the preferred experimental `cpasync` route because
it improves the current zcutlass prefill kernel family. Do not promote it to
normal dispatch or claim vLLM value: TPOT and tokens/s remain worse than stock
vLLM, and the route log is still dominated by decode/large/fallback rows.

Next kernel work should target either:

1. a faster QKV path with bias support, because QKV prefill currently falls back
   to the WMMA tensorop path; or
2. a larger architectural step such as a deeper CUTLASS-style multistage
   mainloop / better fragment scheduling, because tile-shape probes alone are
   not closing the cuBLAS gap.
