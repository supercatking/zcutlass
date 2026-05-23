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

## Prototype Result

The first experimental explicit-MMA operation is available behind the existing
experimental gate:

- `ZCUTLASS_EXPERIMENTAL_KERNELS=1`
- `ZCUTLASS_EXPERIMENTAL_KERNEL=sm120_mma_f16` or `sm120_mma_bf16`

Implemented kernels:

- `zcutlass_sm120_mma_f16_64x128x64_prefill_reg_epilogue`
- `zcutlass_sm120_mma_bf16_64x128x64_prefill_reg_epilogue`

The prototype uses direct global fragment loads and a register epilogue. This
proves the `mma.sync` fragment mapping and fallback routing, but it is not a
promotion candidate.

Measured `128x4096x4096` results:

| dtype | zcutlass ms | zcutlass TFLOP/s | cuBLAS ms | cuBLAS TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| f16 | 0.2922 | 14.6991 | 0.0574 | 74.8148 |
| bf16 | 0.2929 | 14.6654 | 0.0575 | 74.6483 |

vLLM LinearMethod smoke with the FP16 experimental route confirms that framework
routing can select the explicit-MMA kernel, but performance is not yet usable:

| case | stock ms | overlay ms | speedup | hit rate | kernel |
| --- | ---: | ---: | ---: | ---: | --- |
| smoke_prefill | 0.0289 | 0.1022 | 0.283x | 1.00 | `zcutlass_sm120_mma_f16_64x128x64_prefill_reg_epilogue` |
| smoke_prefill_bf16_target | 0.0590 | 0.3356 | 0.176x | 1.00 | `zcutlass_sm120_mma_f16_64x128x64_prefill_reg_epilogue` |

Conclusion: the next kernel step is not more direct-global tuning. It must add
shared-memory staging plus `ldmatrix`/lane-swizzled loading so B loads are not
serialized by strided global access.

## Next Step

Upgrade the experimental operation in `gemm_sm120_mma_prefill.cu`:

- scope: aligned prefill, FP16/BF16, `alpha=1`, `beta=0`, `bias=null`
- mainloop: shared-memory staged A/B tiles with `ldmatrix` loads
- epilogue: keep register linear conversion to FP16/BF16
- promotion gate: correctness, JSONL benchmark, Nsight summary, and explicit
  confirmation that non-target shapes remain on WMMA fallback.

## Validation

- `cmake --build build -j 24`: pass.
- `ctest --test-dir build --output-on-failure`: pass.
- `zcutlass_bench --m 128 --n 4096 --k 4096 --dtype f16 --providers zcutlass --warmup 1 --iterations 3 --json`: pass, reports `pipeline_stages=4` and `epilogue_kind=shared_accumulator`.
- `tools/check_torch_overlay.py --require-extension`: pass after rebuilding the extension in place.
- `tools/check_vllm_plugin.py --require-entry-point --require-vllm` in `/home/zyz/vllm/.venv`: pass.
- `compute-sanitizer --error-exitcode 99 ./build/zcutlass_tests`: pass, 0 errors.
- `cuobjdump --dump-sass build/zcutlass_bench`: confirms the explicit kernels contain `HMMA.16816.F32` / `HMMA.16816.F32.BF16` instructions.
- `benchmark_vllm_linear_method.py --suite smoke --dtype f16` with experimental route: pass, selected the explicit-MMA kernel for prefill callsites.
