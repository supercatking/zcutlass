# Measurement Workflow

Use measurement to decide whether a kernel change is worth keeping. Start small,
then broaden only after correctness and smoke performance are stable.

## Baseline Run

```bash
cd /home/zyz/zcutlass
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build
ctest --test-dir build --output-on-failure
./build/zcutlass_bench --suite smoke --dtype both --json
./build/zcutlass_bench --suite llm-v1.5 --dtype both --providers zcutlass --output build/llm_v15.jsonl
./build/zcutlass_bench --suite correctness --dtype both --providers zcutlass,cublas --output build/measurement.jsonl
python3 tools/summarize_measurements.py build/measurement.jsonl
python3 tools/visualize_gemm_comparison.py --suite smoke --dtype f16 --cutlass-jsonl build/cutlass_baseline.jsonl --output build/gemm_comparison.html
```

To run against a locally built official NVIDIA CUTLASS profiler:

```bash
python3 tools/visualize_gemm_comparison.py \
  --shape 1x1024x1024 \
  --shape 8x2048x2048 \
  --shape 64x1024x1024 \
  --shape 128x2048x2048 \
  --dtype both \
  --warmup 3 \
  --iterations 10 \
  --cutlass-profiler /home/zyz/cutlass-official/build-profiler/tools/profiler/cutlass_profiler \
  --output build/reports/gemm_vs_cutlass_real.html \
  --save-jsonl build/reports/gemm_vs_cutlass_real.jsonl
```

The current CUTLASS 4.5.0 profiler build does not emit f16/bf16 row-major D
instances for the tested row-row shapes, so the script uses row-major A/B and
C/D column-major CUTLASS profiler instances as the external baseline. zcutlass
still measures its v1 row-major A/B/C/D runtime API.

## Compare

```bash
./build/zcutlass_bench --m 256 --n 4096 --k 4096 --dtype f16 --json
python3 tools/tune_gemm.py --suite smoke --dtype both --output build/tuning_results.json
python3 tools/profile_gemm.py --m 256 --n 4096 --k 4096 --dtype f16
python3 tools/compare_cutlass.py --m 256 --n 4096 --k 4096 --dtype f16 --cutlass-dir /path/to/cutlass
python3 tools/compare_cutlass.py --suite llm-v1.5 --dtype both --warmup 3 --iterations 10 --summary
```

Before promoting a new default dispatch or kernel variant, compare it against a
known-good JSONL baseline. The gate compares matching shapes for one provider
and exits non-zero if any shape exceeds the slowdown threshold or the geomean
speedup falls below the configured floor:

```bash
python3 tools/check_benchmark_regression.py \
  reports/2026-05-22-llm-v15-zcutlass-cublas.jsonl \
  build/reports/candidate-llm-v15.jsonl \
  --provider zcutlass \
  --max-slowdown 1.05 \
  --min-geomean-speedup 0.98 \
  --markdown build/reports/candidate-regression-check.md
```

Treat a failed gate as a rejected experiment unless the report includes a
specific, profile-backed reason to keep the variant disabled or opt-in.

For schema-v1 JSONL that can feed the visualization tool directly:

```bash
python3 tools/compare_cutlass.py \
  --providers zcutlass,cutlass \
  --shape 64x1024x1024 \
  --dtype both \
  --warmup 3 \
  --iterations 10 \
  --cutlass-profiler /home/zyz/cutlass-official/build-profiler/tools/profiler/cutlass_profiler \
  --output build/reports/compare_cutlass.jsonl \
  --summary

python3 tools/visualize_gemm_comparison.py \
  --zcutlass-jsonl build/reports/compare_cutlass.jsonl \
  --cutlass-jsonl build/reports/compare_cutlass.jsonl \
  --output build/reports/compare_cutlass.html
```

`tools/profile_gemm.py` exports one Nsight Compute report and, by default,
imports the raw CLI CSV back into small JSON and Markdown summaries:

```bash
python3 tools/profile_gemm.py \
  --m 256 --n 4096 --k 4096 --dtype f16 \
  --warmup 3 --iterations 5 \
  --output-dir build/profiles
```

The default sections cover SpeedOfLight, occupancy, launch resources, memory
workload, scheduler, and warp-state signals. Outputs are named by shape, for
example `build/profiles/gemm_m256_n4096_k4096_f16.ncu-rep`,
`build/profiles/gemm_m256_n4096_k4096_f16.csv`,
`build/profiles/gemm_m256_n4096_k4096_f16.summary.json`, and
`build/profiles/gemm_m256_n4096_k4096_f16.summary.md`.

Useful variants:

```bash
# Collect only the report.
python3 tools/profile_gemm.py --m 256 --n 4096 --k 4096 --dtype f16 --no-import

# Use a wider Nsight section set.
python3 tools/profile_gemm.py --m 256 --n 4096 --k 4096 --dtype f16 --set full

# Re-summarize an already exported raw CSV.
python3 tools/ncu_gemm_summary.py build/profiles/gemm_m256_n4096_k4096_f16.csv \
  --m 256 --n 4096 --k 4096 --dtype f16 \
  --summary-json build/profiles/gemm_m256_n4096_k4096_f16.summary.json \
  --summary-md build/profiles/gemm_m256_n4096_k4096_f16.summary.md
```

Named suites currently include `single`, `correctness`, `smoke`, `llm-v1.5`,
`llm-canonical` (alias), `llm`, `llm-decode`, `llm-prefill`, `square`, and
`ragged`. The `llm-v1.5` suite is the canonical product gate:
`8x4096x4096`, `128x4096x4096`, `128x16384x4096`,
`128x4096x16384`, and `4096x4096x4096`, for both FP16 and BF16.

Nsight Compute may fail with `ERR_NVGPUCTRPERM` until NVIDIA GPU performance
counter permissions are enabled on the host/driver. `tools/profile_gemm.py`
detects this failure and prints a focused hint. Benchmark timing still works
without those counters; full stall and throughput analysis needs the permission.

## Visual Report

Use `tools/visualize_gemm_comparison.py` to generate a self-contained HTML
report with latency, TFLOP/s, and speedup charts. The CUTLASS baseline can come
from schema-v1 JSONL, a CUTLASS profiler CSV, or an external `cutlass_profiler`
binary:

```bash
python3 tools/visualize_gemm_comparison.py \
  --suite smoke \
  --dtype f16 \
  --cutlass-jsonl build/cutlass_baseline.jsonl \
  --output build/gemm_comparison.html
```

## PyTorch Overlay Check

The M1 overlay proof lives under `python/`. It is opt-in and must never hook
PyTorch or cuBLAS globally.

```bash
cd /home/zyz/zcutlass
python3 -m pip install -e ./python --no-build-isolation
python3 tools/check_torch_overlay.py --require-extension
```

In environments without PyTorch, the default check skips cleanly:

```bash
python3 tools/check_torch_overlay.py
```

Record overlay experiments with the same care as kernel benchmarks: model,
callsite, dtype, shape, hit/miss counts, fallback reasons, and PyTorch stock
latency for the same callsite.

The default PyTorch overlay is policy-gated. It accepts only explicitly
promoted shape families; target buckets that are not promoted yet fall back with
`family_not_promoted`, while off-bucket shapes fall back with
`shape_not_target_bucket`. Use `--allow-family` only when a family has enough
correctness and performance evidence for the experiment. Use `--force-zcutlass`
for kernel debugging, not product-value claims.

For synthetic stock-vs-overlay measurements:

```bash
python3 tools/benchmark_torch_overlay.py \
  --suite smoke \
  --dtype both \
  --require-extension \
  --output build/reports/torch_overlay_smoke.jsonl \
  --summary

python3 tools/benchmark_torch_overlay.py \
  --suite llm-v1.5 \
  --dtype both \
  --require-extension \
  --allow-family prefill \
  --output build/reports/torch_overlay_prefill_policy.jsonl \
  --summary
```

Overlay JSONL records include `shape_family`, `route_family`, `kernel_path`,
`fallback_reason`, `hit_count`, `miss_count`, `hit_rate`,
`fallback_reasons`, `routing_policy_enabled`, and `promoted_families`.

For a closer Linear-callsite proof, use the pre-transposed-weight harness. Stock
PyTorch still receives `[N, K]` weights, while zcutlass receives the explicit
`[K, N]` contiguous fast-path weight:

```bash
python3 tools/benchmark_torch_linear_overlay.py \
  --suite smoke \
  --dtype both \
  --require-extension \
  --output build/reports/torch_linear_overlay_smoke.jsonl \
  --summary
```

To model one GEMM-heavy decoder layer before wiring a real serving engine, run
the synthetic LLM layer harness. It measures QKV, output projection, MLP
up/gate, and MLP down Linear callsites, then appends aggregate layer records:

```bash
python3 tools/benchmark_torch_llm_overlay.py \
  --suite smoke \
  --dtype f16 \
  --require-extension \
  --output build/reports/torch_llm_layer_overlay_smoke.jsonl \
  --summary

python3 tools/benchmark_torch_llm_overlay.py \
  --suite smoke \
  --dtype f16 \
  --require-extension \
  --allow-family prefill \
  --output build/reports/torch_llm_layer_overlay_prefill_promoted.jsonl \
  --summary

python3 tools/summarize_overlay_report.py \
  build/reports/torch_llm_layer_overlay_prefill_promoted.jsonl \
  --markdown build/reports/torch_llm_layer_overlay_prefill_promoted.md
```

For the closest M1 proof before SGLang/vLLM integration, use the
`torch.nn.Module` mini decoder harness:

```bash
python3 tools/benchmark_torch_module_overlay.py \
  --suite smoke \
  --dtype both \
  --require-extension \
  --output build/reports/torch_module_overlay_smoke.jsonl \
  --summary

python3 tools/benchmark_torch_module_overlay.py \
  --suite smoke \
  --dtype f16 \
  --require-extension \
  --allow-family prefill \
  --materialize-overlay-inputs \
  --output build/reports/torch_module_overlay_prefill_materialized.jsonl \
  --summary
```

The materialized variant records whether non-contiguous view inputs were copied
before overlay routing. Treat that copy as part of the overlay cost.

For vLLM discovery, first verify that the zcutlass package is visible as a vLLM
general plugin:

```bash
python3 tools/check_vllm_plugin.py
python3 tools/check_vllm_plugin.py --require-entry-point
```

This check does not claim vLLM end-to-end acceleration. It only verifies the
plugin discovery and adapter import layer needed before a custom vLLM model or
worker path routes selected Linear callsites through zcutlass.

To validate the next integration layer, use the vLLM `UnquantizedLinearMethod`
probe. It replaces one dummy vLLM-style Linear layer's `quant_method` with the
explicit zcutlass wrapper, so hits route through zcutlass while misses delegate
back to vLLM's native unquantized GEMM path:

```bash
source /home/zyz/vllm/.venv/bin/activate
cd /home/zyz/zcutlass

python tools/check_vllm_linear_method.py \
  --m 128 --n 4096 --k 1024 \
  --dtype f16 \
  --allow-family prefill \
  --require-hit

python tools/check_vllm_linear_method.py \
  --m 8 --n 512 --k 512 \
  --dtype f16 \
  --allow-family prefill \
  --require-fallback
```

## Record

Capture enough context to make a result repeatable:

- Git commit or working-tree note.
- GPU model, driver, CUDA version, and architecture flag.
- Command line, dtype, shape, suite, median latency, and TFLOP/s.
- Baseline used: previous zcutlass, cuBLAS, or external CUTLASS profiler.
- Any correctness, sanitizer, thermal, or clocking caveats.

## Promotion Bar

A result is ready to cite when it passes tests, has a stable smoke measurement,
and includes a baseline comparison. Wider suites such as `llm` should be used
before claiming broad product improvement.
