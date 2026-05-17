# zcutlass

zcutlass is a clean-room CUDA GEMM library aimed first at NVIDIA GeForce RTX 50
series GPUs (`sm_120`). The v1 surface is deliberately small: row-major dense
FP16/BF16 GEMM with FP32 accumulation and an optional N-broadcast bias.

The v1.5 goal is a verifiable LLM inference GEMM acceleration layer for RTX
5080/SM120. zcutlass should win selected LLM GEMM buckets through explicit
framework integration while unsupported or non-profitable paths fall back to the
original PyTorch/SGLang/vLLM, cuBLAS, or CUTLASS route. Arbitrary row-major
FP16/BF16 GEMM remains supported for correctness coverage and fallback dispatch,
but it is not the optimized target for v1.5.

The project studies public CUTLASS design ideas and uses CUTLASS/cuBLAS only as
external baselines. CUTLASS source is not vendored into this repository.

## Docs

- [CUTLASS alignment roadmap](docs/cutlass-alignment-roadmap.md)
- [Measurement workflow](docs/measurement-workflow.md)
- [Development workflow](docs/development-workflow.md)
- [Agent operating notes](AGENTS.md)
- [LLM kernel family plan](docs/llm-kernel-family-plan.md)
- [Kernel optimization notes](docs/kernel-optimization-notes.md)
- [Agent coordination](docs/agent-coordination.md)
- [2026-05-17 v1.5 LLM GEMM goal lock](reports/2026-05-17-v15-llm-goal-lock.md)
- [2026-05-17 M1 PyTorch overlay proof](reports/2026-05-17-m1-pytorch-overlay-proof.md)
- [2026-05-16 RTX 5080 CUTLASS comparison](reports/2026-05-16-sm120-gemm-vs-cutlass.md)
- [2026-05-16 Batch 1 evidence chain](reports/2026-05-16-batch1-evidence-chain.md)

## Build

```bash
cd /home/zyz/zcutlass
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build
```

## Test

```bash
ctest --test-dir build --output-on-failure
compute-sanitizer ./build/zcutlass_tests
```

## Benchmark

```bash
./build/zcutlass_bench --suite smoke --dtype f16
./build/zcutlass_bench --m 256 --n 4096 --k 4096 --dtype bf16 --json
./build/zcutlass_bench --suite correctness --dtype both --providers zcutlass,cublas --output build/measurement.jsonl
python3 tools/summarize_measurements.py build/measurement.jsonl
python3 tools/visualize_gemm_comparison.py --suite smoke --dtype f16 --cutlass-jsonl build/cutlass_baseline.jsonl --output build/gemm_comparison.html
python3 tools/tune_gemm.py --suite smoke --dtype both --output build/tuning_results.json
python3 tools/profile_gemm.py --m 256 --n 4096 --k 4096 --dtype f16
python3 tools/compare_cutlass.py --m 256 --n 4096 --k 4096 --dtype f16 --cutlass-dir /path/to/cutlass
```

The benchmark compares zcutlass against cuBLAS with CUDA events and reports
median latency plus TFLOP/s. The `llm` suite covers common hidden sizes
`1024..8192` and token/batch sizes `1..1024`; it can allocate large matrices, so
start with `smoke` while iterating. CUTLASS comparison is intentionally routed
through an external `cutlass_profiler`; no CUTLASS source is copied into this
repository.

Microbenchmarks are necessary but not sufficient for v1.5. Commercial value must
be proven in an LLM inference stack with fixed model/workload settings and
end-to-end metrics such as TTFT, TPOT/decode token latency, tokens/s, p95/p99,
zcutlass hit rate, fallback reason histogram, and numerical correctness.

Kernel growth for v1.5 must follow CUTLASS-style tile families and
shape-bucket dispatch. Do not add many full-shape-specific kernels for individual
LLM dimensions; add reusable tile/pipeline/epilogue families and route nearby
problem shapes into buckets that can be measured and maintained.

## v1.5 Product Validation Path

1. PyTorch overlay proof: expose a `torch.ops.zcutlass.gemm` or
   `zcutlass_linear` path for explicitly selected Linear/GEMM callsites, with
   fallback to the original PyTorch operation. The first overlay package lives
   under `python/` and is intentionally opt-in. Routing is policy-gated:
   unpromoted or currently non-profitable shapes must fall back with observable
   hit/miss and fallback-reason counters instead of forcing zcutlass.
2. SGLang serving proof: integrate zcutlass as a GEMM overlay only, leaving
   attention, KV cache, scheduling, and sampling on the stock path.
3. vLLM OOT CustomOp proof: validate the same overlay model through a vLLM
   out-of-tree custom op rather than global cuBLAS interception.

v1.5 succeeds only when at least one real inference engine shows a stable TTFT or
TPOT improvement without correctness failures or p95/p99 regressions. External
CUTLASS/cuBLAS results explain kernel-level behavior; they do not replace
serving-level proof.

## Roadmap Docs

- [CUTLASS alignment roadmap](docs/cutlass-alignment-roadmap.md)
- [Measurement workflow](docs/measurement-workflow.md)
- [Development workflow](docs/development-workflow.md)

## Public API

```cpp
#include <zcutlass/gemm.hpp>

zcutlass::GemmDesc desc{/* m/n/k/ld..., dtype fields, alpha, beta, bias, stream */};
zcutlass::Status status = zcutlass::gemm(desc, A, B, C, D);
```

v1 supports only matching input/output dtypes: all FP16 or all BF16. Mixed dtype,
non-row-major layouts, grouped GEMM, sparse GEMM, FP8/FP4, and attention kernels
are intentionally out of scope for the first milestone.
