# 2026-05-23 M32 cp.async Prefill Experiment

## Scope

This experiment tested a `32x128x64` explicit-MMA prefill variant for the
SM120 cp.async double-buffer path. The goal was to check whether doubling CTA
count along M improves Qwen2.5-1.5B prefill/down-projection shapes.

The variant is experimental-only and selectable with:

```bash
ZCUTLASS_EXPERIMENTAL_KERNELS=1
ZCUTLASS_EXPERIMENTAL_KERNEL=32x128x64
```

## Verification

- `cmake --build build -j 24`: passed.
- `ctest --test-dir build --output-on-failure`: passed.
- `compute-sanitizer --error-exitcode 99 ./build/zcutlass_tests`: passed with
  `ERROR SUMMARY: 0 errors`.
- `cuobjdump --dump-resource-usage build/libzcutlass.a`: M32 cp.async kernels
  use `REG:124`, `STACK:0`, and dynamic shared memory.

The correctness suite now includes exact-filter FP16 and BF16 M32 prefill smoke
cases, so this variant is not only benchmarked but also reference-checked.

## Results

All timings are CUDA-event medians on local RTX 5080 / SM120, BF16, row-major
A/B/D, `alpha=1`, `beta=0`, no bias.

| Shape | Kernel | zcutlass ms | cuBLAS ms | Speedup |
| --- | --- | ---: | ---: | ---: |
| `256x1536x1536` | `32x128x64 cp.async` | `0.0409` | `0.0266` | `0.651x` |
| `256x17920x1536` | `32x128x64 cp.async` | `0.2568` | `0.1476` | `0.575x` |
| `256x1536x8960` | `32x128x64 cp.async` | `0.1926` | `0.0748` | `0.388x` |
| `128x4096x4096` | `32x128x64 cp.async` | `0.0948` | `0.0563` | `0.594x` |
| `256x1536x8960` | `64x128x64 cp.async` | `0.1919` | `0.0778` | `0.406x` |

## Decision

Do not promote `32x128x64` to default dispatch. It is correct and useful as a
controlled experiment, but it does not improve the Qwen2.5-1.5B down-projection
target versus the existing `64x128x64` cp.async path.

The next kernel direction should not rely on CTA-count changes alone. The data
points toward improving per-CTA math density, shared-memory/ldmatrix efficiency,
or moving to a more CUTLASS-like multistage mainloop rather than shrinking M.
