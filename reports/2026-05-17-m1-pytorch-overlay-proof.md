# M1 PyTorch Overlay Proof

Date: 2026-05-17

## Goal

M1 proves that zcutlass can be called as an explicit PyTorch GEMM overlay without
globally replacing PyTorch, cuBLAS, or CUTLASS. Unsupported paths remain
fallback-capable at the Python wrapper layer.

## Implemented

- `python/zcutlass_torch` package.
- `torch.ops.zcutlass_torch.gemm` CUDA custom op.
- `ZCutlassGemmOverlay` Python wrapper with hit/miss and fallback reason stats.
- `RoutingPolicy` gate for observable fallback before a family is promoted.
- `tools/check_torch_overlay.py` skip-friendly extension smoke check.
- `tools/benchmark_torch_overlay.py` stock PyTorch vs zcutlass overlay JSONL
  benchmark.

## Build Environment

- Host GPU: NVIDIA GeForce RTX 5080.
- System CUDA toolkit: 13.1.
- Working PyTorch validation environment: `build/torch-cu130-venv`.
- PyTorch: `2.11.0+cu130`.
- PyTorch CUDA: `13.0`.

The first validation attempt with PyTorch `2.11.0+cu128` failed because PyTorch
extension builds reject a major CUDA mismatch between local CUDA 13.1 and
PyTorch CUDA 12.8. The cu130 wheel accepts the local CUDA 13.1 toolkit as a
minor-version mismatch and successfully builds the extension.

## Commands

```bash
cd /home/zyz/zcutlass
python3 -m venv build/torch-cu130-venv
source build/torch-cu130-venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ./python --no-build-isolation --force-reinstall
python3 tools/check_torch_overlay.py --require-extension
python3 tools/benchmark_torch_overlay.py --suite smoke --dtype both \
  --warmup 1 --iterations 3 --require-extension --summary \
  --output reports/2026-05-17-m1-torch-overlay-smoke-both.jsonl
```

## Results

Updated extension smoke check:

```text
PASS: forced_hits=1 policy_misses=1 policy_reasons={'shape_not_target_bucket': 1}
```

Synthetic smoke benchmark:

| Shape | Stock PyTorch ms | zcutlass overlay ms | Speedup | Hit rate |
| --- | ---: | ---: | ---: | ---: |
| f16 8x256x256 | 0.0176 | 0.0449 | 0.391x | 1.00 |
| f16 32x512x512 | 0.0087 | 0.0477 | 0.182x | 1.00 |
| f16 128x1024x1024 | 0.0227 | 0.0818 | 0.278x | 1.00 |
| bf16 8x256x256 | 0.0127 | 0.0466 | 0.273x | 1.00 |
| bf16 32x512x512 | 0.0099 | 0.0475 | 0.208x | 1.00 |
| bf16 128x1024x1024 | 0.0165 | 0.0772 | 0.213x | 1.00 |

These smoke shapes validate integration and correctness, not performance. The
current WMMA baseline is slower than stock PyTorch on these small synthetic
cases. That is acceptable for M1 because the milestone is callsite integration,
fallback observability, and correctness. Performance promotion still requires
the v1.5 kernel family work.

The overlay now defaults to policy-gated routing. Target LLM families are
classified for measurement, but no family is sent to zcutlass unless it is
explicitly promoted with benchmark evidence. Unpromoted target buckets fall back
with `family_not_promoted`; off-bucket shapes fall back with
`shape_not_target_bucket`. Kernel debugging can still force the zcutlass path,
but forced runs are not product-value evidence.

## Next Steps

- Add a PyTorch model-level offline Linear replacement harness with explicit
  pre-transposed weights.
- Promote a first family only after stock-vs-overlay measurement shows a
  product-relevant win or after a kernel change creates one.
- Use the same JSONL schema for model callsite records before moving to SGLang
  serving proof.
