# zcutlass v1.5 LLM GEMM Goal Lock

Date: 2026-05-17

## Scope

v1.5 is now defined as a dedicated LLM GEMM optimization milestone for RTX
5080 / SM120. Arbitrary row-major FP16/BF16 GEMM remains supported for
correctness and fallback dispatch, but is not a performance target.

The optimization strategy is CUTLASS-style tile-family dispatch:

- `decode`: `M <= 16`, large `N/K`, LLM token decode.
- `prefill`: `32 <= M <= 256`, large `N/K`, prompt/prefill and medium batch.
- `large`: large `M/N/K`, square or high-throughput batch GEMM.
- `fallback`: ragged, padded, beta/bias, small non-target, or unsupported fast
  path.

Do not add many full `(M,N,K)` shape-specific kernels. Add reusable tile,
mainloop, schedule, and epilogue families, then route shapes through buckets.

## Implemented

- Added public selection observability:
  - `selected_kernel_family`
  - `selected_kernel_path`
  - `selected_kernel_tile_m/n/k`
- Added manifest `ShapeFamily` metadata and dynamic problem-family
  classification.
- Added `llm-v1.5` benchmark suite and `llm-canonical` alias.
- Added JSONL tags for `shape_family`, `kernel_path`, and tile shape.
- Updated CUTLASS compare and visualization tooling to accept `llm-v1.5`.
- Updated docs and agent policy:
  - `README.md`
  - `AGENTS.md`
  - `docs/cutlass-alignment-roadmap.md`
  - `docs/agent-coordination.md`
  - `docs/llm-kernel-family-plan.md`

## Validation

Commands passed:

```bash
cmake --build build -j 24
ctest --test-dir build --output-on-failure
compute-sanitizer --error-exitcode 99 ./build/zcutlass_tests
python3 -m py_compile tools/compare_cutlass.py tools/visualize_gemm_comparison.py tools/profile_gemm.py tools/ncu_gemm_summary.py
```

`compute-sanitizer` reported `ERROR SUMMARY: 0 errors`.

## Canonical zcutlass Measurements

Command:

```bash
./build/zcutlass_bench --suite llm-v1.5 --dtype both --providers zcutlass \
  --warmup 1 --iterations 1 \
  --output reports/2026-05-17-v15-llm-canonical-zcutlass.jsonl
```

Key results:

| Shape | Family | Path | Kernel | Median ms | TFLOP/s |
| --- | --- | --- | --- | ---: | ---: |
| f16 8x4096x4096 | decode | fallback | `f16_64x128x16` | 0.3042 | 0.8824 |
| bf16 8x4096x4096 | decode | fallback | `bf16_64x128x16` | 0.2970 | 0.9039 |
| f16 128x4096x4096 | prefill | fast | `f16_64x128x16_aligned` | 0.3016 | 14.2421 |
| bf16 128x4096x4096 | prefill | fast | `bf16_64x128x16_aligned` | 0.2855 | 15.0435 |
| f16 128x16384x4096 | prefill | fast | `f16_64x128x16_aligned` | 35.7017 | 0.4812 |
| bf16 128x16384x4096 | prefill | fast | `bf16_64x128x16_aligned` | 30.6483 | 0.5605 |
| f16 128x4096x16384 | prefill | fast | `f16_64x128x16_aligned` | 37.2877 | 0.4607 |
| bf16 128x4096x16384 | prefill | fast | `bf16_64x128x16_aligned` | 32.5383 | 0.5280 |
| f16 4096x4096x4096 | large | fast | `f16_64x128x16_aligned` | 19.4435 | 7.0686 |
| bf16 4096x4096x4096 | large | fast | `bf16_64x128x16_aligned` | 5.2450 | 26.2040 |

## Representative CUTLASS Comparison

Command:

```bash
python3 tools/compare_cutlass.py --providers zcutlass,cutlass \
  --shape 128x4096x4096 --dtype both --warmup 1 --iterations 1 \
  --output reports/2026-05-17-v15-representative-cutlass.jsonl --summary
```

| Shape | zcutlass ms | CUTLASS ms | zcutlass / CUTLASS |
| --- | ---: | ---: | ---: |
| f16 128x4096x4096 | 0.3049 | 0.3871 | 1.27x faster |
| bf16 128x4096x4096 | 0.2833 | 0.2027 | 0.72x slower |

The CUTLASS profiler baseline still uses row-major A/B and column-major C/D
instances for this external comparison because the local official profiler
build does not enumerate matching row-row-row-row f16/bf16 tensor-op instances.

HTML report:
`reports/2026-05-17-v15-representative-cutlass.html`

## Nsight Compute Summary

Commands:

```bash
python3 tools/profile_gemm.py --m 128 --n 4096 --k 4096 --dtype f16 \
  --warmup 1 --iterations 1 --output-dir reports/profiles --launch-count 1
python3 tools/profile_gemm.py --m 128 --n 4096 --k 4096 --dtype bf16 \
  --warmup 1 --iterations 1 --output-dir reports/profiles --launch-count 1
```

| Shape | SM % | Tensor % | DRAM % | Achieved occupancy | Top stall |
| --- | ---: | ---: | ---: | ---: | --- |
| f16 128x4096x4096 | 7.75 | 7.63 | 7.34 | 16.67% | long scoreboard |
| bf16 128x4096x4096 | 13.82 | 9.16 | 8.41 | 16.66% | long scoreboard |

Nsight reports:

- `reports/profiles/gemm_m128_n4096_k4096_f16.ncu-rep`
- `reports/profiles/gemm_m128_n4096_k4096_bf16.ncu-rep`

## Next Optimization Direction

The current WMMA baseline is sufficient as a correctness/fallback anchor, but
not enough for the v1.5 LLM objective. The immediate kernel work should focus on:

- Decode family: avoid under-filled block scheduling and improve small-M
  occupancy.
- Prefill family: implement staged/double-buffered K mainloop to reduce long
  scoreboard stalls.
- BF16 family: close the gap to CUTLASS's `256x128_32x3` style pipeline.
- Large family: stop reusing the same prefill kernel blindly; add a throughput
  family with a larger tile and better K pipeline.
