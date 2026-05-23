# M1 Explicit-MMA Prefill Scaffold

## Scope

This change prepares the codebase for the next prefill kernel generation without
changing selected kernels or runtime behavior.

## Delivered

- Added manifest metadata for `pipeline_stages` and `epilogue_kind`.
- Added public selection helpers for pipeline and epilogue reporting.
- Added benchmark JSONL and `--json` fields for the new metadata.
- Added PyTorch extension `selected_gemm_config()` fields for the new metadata.
- Added `src/gemm_sm120_mma_prefill.cuh` and `.cu` as the explicit-MMA
  registration boundary.
- Hooked the explicit-MMA manifest append point before existing WMMA prefill
  entries. The append function intentionally registers no operation yet.

## Expected Behavior

- Current WMMA kernels remain the selected path.
- Existing fallback behavior is unchanged.
- `128x4096x4096` FP16 should still report the current
  `64x128x64_aligned_prefill_n_le_k` WMMA kernel with
  `pipeline_stages=4` and `epilogue_kind=shared_accumulator`.
- BF16 prefill should still report the current `64x128x32` WMMA kernel with
  `pipeline_stages=2` and `epilogue_kind=shared_accumulator`.

## Next Step

Implement the first experimental operation in `gemm_sm120_mma_prefill.cu`:

- kernel name: `zcutlass_sm120_mma_f16_64x128x64_prefill_reg_epilogue`
- scope: aligned prefill, FP16, `alpha=1`, `beta=0`, `bias=null`
- epilogue: register linear conversion to FP16
- promotion gate: correctness, JSONL benchmark, Nsight summary, and explicit
  confirmation that non-target shapes remain on WMMA fallback.

## Validation

- `cmake --build build -j 24`: pass.
- `ctest --test-dir build --output-on-failure`: pass.
- `zcutlass_bench --m 128 --n 4096 --k 4096 --dtype f16 --providers zcutlass --warmup 1 --iterations 3 --json`: pass, reports `pipeline_stages=4` and `epilogue_kind=shared_accumulator`.
- `tools/check_torch_overlay.py --require-extension`: pass after rebuilding the extension in place.
- `tools/check_vllm_plugin.py --require-entry-point --require-vllm` in `/home/zyz/vllm/.venv`: pass.
