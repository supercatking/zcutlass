# Batch 1 Evidence Chain

Date: 2026-05-16

GPU: NVIDIA GeForce RTX 5080, CUDA 13.1, `sm_120`

## What Landed

- Nsight Compute automation now exports `.ncu-rep`, imports raw CSV, and writes
  shape-level Markdown/JSON summaries.
- CUTLASS comparison now emits schema-v1 JSONL with official profiler command,
  CUTLASS commit, kernel name, and layout caveat.
- Correctness coverage now includes zero-size, richer invalid arguments, ragged
  FP16/BF16, alpha/beta, bias/no-bias, padded leading dimensions, and output
  padding sentinel checks.
- Agent coordination rules are documented for write ownership and single-GPU
  measurement serialization.

## Validation

```bash
cmake --build build -j 24
ctest --test-dir build --output-on-failure
compute-sanitizer --error-exitcode 99 ./build/zcutlass_tests
python3 -m py_compile tools/profile_gemm.py tools/ncu_gemm_summary.py tools/compare_cutlass.py tools/visualize_gemm_comparison.py tools/summarize_measurements.py
python3 tools/compare_cutlass.py --providers zcutlass,cutlass --shape 64x1024x1024 --dtype both --warmup 1 --iterations 3 --output build/reports/batch1_compare_cutlass.jsonl --summary
python3 tools/profile_gemm.py --m 64 --n 1024 --k 1024 --dtype f16 --warmup 1 --iterations 1 --output-dir build/reports/profiles --launch-count 1
```

All commands above passed. `compute-sanitizer` reported `ERROR SUMMARY: 0 errors`.

## Smoke Comparison

| dtype | shape | zcutlass ms | CUTLASS ms | speedup |
| --- | --- | ---: | ---: | ---: |
| f16 | 64x1024x1024 | 0.0824 | 0.1044 | 1.267x |
| bf16 | 64x1024x1024 | 0.0779 | 0.0551 | 0.707x |

CUTLASS baseline used `/home/zyz/cutlass-official` at commit
`e406c186f510a15091cce01f782020ceb7ba8eb5`.

## Nsight Compute Snapshot

Profiled shape: `f16 64x1024x1024`

- SM throughput: `0.97%`
- Tensor throughput: `0.91%`
- DRAM throughput: `1.90%`
- Achieved occupancy: `16.66%`
- Registers per thread: `64`
- Allocated shared memory per block: `39.94 KB`
- Top stall: `long scoreboard`

This confirms the current WMMA baseline is not close to saturating Tensor Cores.
The next kernel work should focus on memory dependency and staging before making
claims about peak SM120 performance.

## Artifacts

- `reports/2026-05-16-batch1-compare-cutlass.html`
- `reports/2026-05-16-batch1-compare-cutlass.jsonl`
- `reports/2026-05-16-batch1-ncu-f16-64x1024x1024.md`
- `reports/2026-05-16-batch1-sanitizer.txt`

