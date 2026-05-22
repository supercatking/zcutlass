# 2026-05-23 vLLM dispatch telemetry proof

## Goal

Prove that the PyTorch/vLLM overlay records the actual zcutlass C++ manifest
selection for hit paths. This closes the framework evidence gap where previous
reports could show `kernel_path=zcutlass` but could not name the selected
kernel or tile.

## Changes validated

- `zcutlass_torch.selected_gemm_config()` now calls the C++ manifest selector.
- `ZCutlassGemmOverlay` records `last_kernel_name`, `last_tile`, and
  `last_config` for hit paths.
- `ZCutlassVllmLinearAdapter` and `ZCutlassUnquantizedLinearMethod` propagate
  the same metadata into `last_trace`.
- Overlay benchmark JSONL records include `kernel_name` and `selected_config`.

## Commands

```bash
cd /home/zyz/zcutlass
source /home/zyz/vllm/.venv/bin/activate

ZCUTLASS_ALLOW_CUDA_MISMATCH=1 MAX_JOBS=16 \
  python -m pip install -e ./python --no-build-isolation

python tools/check_torch_overlay.py --require-extension

python tools/check_vllm_linear_method.py \
  --m 128 --n 1024 --k 4096 \
  --dtype f16 \
  --allow-family prefill \
  --require-hit \
  --warmup 5 \
  --iterations 20 \
  --output reports/2026-05-23-vllm-linear-method-telemetry-f16-n-le-k.jsonl

python tools/check_vllm_linear_method.py \
  --m 128 --n 4096 --k 1024 \
  --dtype f16 \
  --allow-family prefill \
  --require-hit \
  --warmup 5 \
  --iterations 20 \
  --output reports/2026-05-23-vllm-linear-method-telemetry-f16-n-gt-k.jsonl
```

## Evidence

- `128x1024x4096 f16` routed through
  `zcutlass_sm120_tensorop_f16_64x128x64_aligned_prefill_n_le_k`, tile
  `64x128x64`, `hit_rate=1.0`.
- `128x4096x1024 f16` routed through
  `zcutlass_sm120_tensorop_f16_64x128x32_aligned_prefill`, tile `64x128x32`,
  `hit_rate=1.0`.
- Both checks loaded the vLLM plugin and used the rebuilt extension in
  `/home/zyz/vllm/.venv`.

## Performance note

This is an observability improvement, not a performance promotion. The measured
overlay path remains slower than vLLM's native unquantized GEMM for these probe
shapes, so the v1.5 product-value gate is still open.
