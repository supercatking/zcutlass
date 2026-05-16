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
./build/zcutlass_bench --suite correctness --dtype both --providers zcutlass,cublas --output build/measurement.jsonl
python3 tools/summarize_measurements.py build/measurement.jsonl
```

## Compare

```bash
./build/zcutlass_bench --m 256 --n 4096 --k 4096 --dtype f16 --json
python3 tools/tune_gemm.py --suite smoke --dtype both --output build/tuning_results.json
python3 tools/profile_gemm.py --m 256 --n 4096 --k 4096 --dtype f16
python3 tools/compare_cutlass.py --m 256 --n 4096 --k 4096 --dtype f16 --cutlass-dir /path/to/cutlass
```

Named suites currently include `single`, `correctness`, `smoke`, `llm`,
`llm-decode`, `llm-prefill`, `square`, and `ragged`.

Nsight Compute may fail with `ERR_NVGPUCTRPERM` until NVIDIA GPU performance
counter permissions are enabled on the host/driver. Benchmark timing still works
without those counters; full stall and throughput analysis needs the permission.

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
