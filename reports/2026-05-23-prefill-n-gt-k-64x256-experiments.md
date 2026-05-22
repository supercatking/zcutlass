# 2026-05-23 f16 prefill N>K 64x256 experiments

## Goal

Test whether a wider `N` tile improves LLM up-projection style GEMMs such as
`128x16384x4096` by reducing N-direction CTAs and repeated A tile loads.

## Candidates

- `zcutlass_sm120_tensorop_f16_64x256x32_aligned_prefill_n_gt_k_experimental`
  - `BlockM=64`, `BlockN=256`, `KGroup=2`, `tile_k=32`.
  - Restricted to `ShapeFamily::Prefill` and `N > K`.
  - Requires `ZCUTLASS_EXPERIMENTAL_KERNELS=1` and a matching filter.
- `zcutlass_sm120_tensorop_f16_64x256x16_aligned_prefill_n_gt_k_experimental`
  - `BlockM=64`, `BlockN=256`, `KGroup=1`, `tile_k=16`.
  - Same opt-in and shape restrictions.

The default path remains
`zcutlass_sm120_tensorop_f16_64x128x32_aligned_prefill` for N>K prefill shapes.

## Commands

```bash
cd /home/zyz/zcutlass

cmake --build build -j 24
ctest --test-dir build --output-on-failure

./build/zcutlass_bench --m 128 --n 16384 --k 4096 \
  --dtype f16 \
  --providers zcutlass,cublas \
  --warmup 20 \
  --iterations 50 \
  --output reports/2026-05-23-up-n-gt-k-default-f16-rerun.jsonl

./build/zcutlass_bench --m 128 --n 16384 --k 4096 \
  --dtype f16 \
  --providers zcutlass,cublas \
  --experimental-kernel 64x256x32 \
  --warmup 20 \
  --iterations 50 \
  --output reports/2026-05-23-up-n-gt-k-64x256x32-experimental-f16-rerun.jsonl

./build/zcutlass_bench --m 128 --n 16384 --k 4096 \
  --dtype f16 \
  --providers zcutlass,cublas \
  --experimental-kernel 64x256x16 \
  --warmup 20 \
  --iterations 50 \
  --output reports/2026-05-23-up-n-gt-k-64x256x16-experimental-f16.jsonl
```

## Results

| Candidate | Target shape zcutlass ms | Default ms | Speedup vs default | Decision |
| --- | ---: | ---: | ---: | --- |
| `64x256x32` | 0.9408 | 0.9857 | 1.0477x | Keep experimental, do not promote |
| `64x256x16` | 1.1724 | 0.9857 | 0.8408x | Reject for promotion |

The full f16 `llm-v1.5` prefill regression check for `64x256x32` showed
`1.0136x` geomean speedup across matching prefill records. The target
N>K shape improved `1.0473x` in the suite run, but this is below the `1.05x`
single-shape promotion gate and is not enough to justify changing default
dispatch.

The `64x256x16` variant regressed the target N>K shape by `1.1917x` and failed
the geomean gate.

## Notes

One first-run single-shape measurement for `64x256x32` reported `6.6848 ms`,
but the rerun with longer warmup and the suite run both landed near `0.94 ms`.
Promotion decisions use the stable rerun and suite data.

No Nsight promotion profile was taken because neither candidate met the
promotion threshold. The next N>K investigation should focus on reducing the
shared-memory accumulator spill and the high overhead of 16-warps/block WMMA
tiles rather than simply widening `BlockN`.
