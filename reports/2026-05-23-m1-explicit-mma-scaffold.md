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

## Shared-Memory Staging Follow-Up

A second experimental variant stages each `64x128x64` CTA tile through shared
memory before feeding the same register-level `mma.sync` epilogue path:

- `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_reg_epilogue`
- `zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_reg_epilogue`

Measured `128x4096x4096` results:

| dtype | zcutlass ms | zcutlass TFLOP/s | cuBLAS ms | cuBLAS TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| f16 | 0.2712 | 15.8350 | 0.0544 | 78.9516 |
| bf16 | 0.2713 | 15.8332 | 0.0555 | 77.3589 |

This is a real improvement over direct-global fragment loads, but still slower
than the current WMMA fallback for the same canonical shape. It confirms that
data staging matters, while also showing that scalar shared-memory fragment
loads are not enough. The next iteration must use `ldmatrix`-compatible shared
layout and reduce the 1024-thread CTA shape.

vLLM LinearMethod smoke with the shared-memory FP16 route:

| case | stock ms | overlay ms | speedup | hit rate | kernel |
| --- | ---: | ---: | ---: | ---: | --- |
| smoke_prefill | 0.0246 | 0.0945 | 0.260x | 1.00 | `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_reg_epilogue` |
| smoke_prefill_bf16_target | 0.0586 | 0.3131 | 0.187x | 1.00 | `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_reg_epilogue` |

## Warp 16x32 Follow-Up

A third experimental variant keeps the shared-memory staging and register
epilogue, but increases each warp's output tile from `16x16` to `16x32`.
This reduces CTA size from 1024 threads to 512 threads and doubles the MMA work
per warp:

- `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_warp16x32_reg_epilogue`
- `zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_warp16x32_reg_epilogue`

Measured `128x4096x4096` results:

| dtype | zcutlass ms | zcutlass TFLOP/s | cuBLAS ms | cuBLAS TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| f16 | 0.2455 | 17.4945 | 0.0575 | 74.6899 |
| bf16 | 0.2459 | 17.4649 | 0.0559 | 76.8275 |

This is the first explicit-MMA variant to edge past the current default WMMA
kernel on the FP16 canonical prefill shape, but it is still far from the
`>=0.80x` cuBLAS/CUTLASS M2 promotion threshold. It should remain experimental.

## Warp 16x64 Follow-Up

A fourth experimental variant increases each warp output tile again, from
`16x32` to `16x64`, reducing the CTA to 256 threads:

- `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_warp16x64_reg_epilogue`
- `zcutlass_sm120_mma_bf16_64x128x64_prefill_smem_warp16x64_reg_epilogue`

Measured `128x4096x4096` results:

| dtype | zcutlass ms | zcutlass TFLOP/s | cuBLAS ms | cuBLAS TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| f16 | 0.2690 | 15.9669 | 0.0563 | 76.3468 |
| bf16 | 0.2678 | 16.0356 | 0.0555 | 77.4035 |

This regresses versus `16x32`, so blindly expanding per-warp output is not the
right next step. The current best experimental point is `16x32`.

## ldmatrix 16x32 Follow-Up

A fifth experimental variant keeps the `16x32` warp tile and replaces scalar
shared-memory fragment loads with `ldmatrix`:

- A: `ldmatrix.sync.aligned.m8n8.x4.shared.b16`
- B: `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16`
- Shared layout: padded row-major A/B tiles.

Measured `128x4096x4096` results:

| dtype | zcutlass ms | zcutlass TFLOP/s | cuBLAS ms | cuBLAS TFLOP/s |
| --- | ---: | ---: | ---: | ---: |
| f16 | 0.1633 | 26.3069 | 0.0549 | 78.2611 |
| bf16 | 0.1627 | 26.3948 | 0.0549 | 78.2611 |

This is the best explicit-MMA result so far and is about `1.50x` faster than
the scalar shared-memory `16x32` variant. It is still only about `0.34x` of
cuBLAS on the canonical prefill shape, so it remains experimental and cannot be
used for vLLM promotion.

vLLM LinearMethod smoke with the ldmatrix FP16 route:

| case | stock ms | overlay ms | speedup | hit rate | kernel |
| --- | ---: | ---: | ---: | ---: | --- |
| smoke_prefill | 0.0229 | 0.0658 | 0.348x | 1.00 | `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_ldm_warp16x32_reg_epilogue` |
| smoke_prefill_bf16_target | 0.0810 | 0.2096 | 0.386x | 1.00 | `zcutlass_sm120_mma_f16_64x128x64_prefill_smem_ldm_warp16x32_reg_epilogue` |

## Next Step

Upgrade the experimental operation in `gemm_sm120_mma_prefill.cu`:

- scope: aligned prefill, FP16/BF16, `alpha=1`, `beta=0`, `bias=null`
- mainloop: keep the `ldmatrix` 16x32 path and improve global-to-shared copy,
  double buffering, shared-memory swizzle, and occupancy
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
