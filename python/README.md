# zcutlass PyTorch Overlay

This package is the M1 proof path for using zcutlass as an explicit PyTorch
GEMM overlay. It is not a global PyTorch, cuBLAS, or CUTLASS replacement.
Callsites opt in, unsupported inputs fall back in Python, and experiments record
hit/miss counts.

## Install

From the repository root:

```bash
cd /home/zyz/zcutlass
python3 -m pip install -e ./python --no-build-isolation
```

The extension builds the current zcutlass sources into `zcutlass_torch._C`.
PyTorch with CUDA support is required for the compiled extension.

## Use

```python
import torch
from zcutlass_torch import ZCutlassGemmOverlay

overlay = ZCutlassGemmOverlay()
A = torch.randn((128, 4096), device="cuda", dtype=torch.float16)
B = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
D = overlay.gemm(A, B)
print(overlay.stats.hits, overlay.stats.fallback_reasons)
```

For Linear layers, PyTorch stores weight as `[out_features, in_features]`, while
zcutlass v1 expects row-major GEMM `B` as `[K, N]`. The zcutlass fast path
therefore requires pre-transposed contiguous weight:

```python
weight_k_n = module.weight.detach().t().contiguous()
out = overlay.linear(x, weight_k_n, module.bias, weight_is_transposed=True)
```

Without `weight_is_transposed=True`, the overlay preserves stock PyTorch Linear
semantics and records a `weight_not_pretransposed` fallback.

## Check

```bash
python3 tools/check_torch_overlay.py
python3 tools/check_torch_overlay.py --require-extension
```

The default check skips cleanly when PyTorch is not installed. Use
`--require-extension` in an environment where the extension should be built.

## Benchmark

```bash
python3 tools/benchmark_torch_overlay.py \
  --suite smoke \
  --dtype both \
  --require-extension \
  --output build/reports/torch_overlay_smoke.jsonl \
  --summary
```

This benchmark records stock `torch.matmul` and explicit zcutlass overlay
measurements as schema-v1 JSONL. It also records hit rate and fallback reasons
for the overlay path.

For a closer `torch.nn.Module` proof before wiring a serving engine:

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

The module harness builds a tiny decoder block with QKV, output projection, MLP
up/gate, and MLP down Linear modules. Stock and overlay paths share the same
weights; only the Linear execution route changes.
`--materialize-overlay-inputs` makes non-contiguous view inputs explicit in the
measurement, which is important because zcutlass v1 only accepts contiguous
row-major tensors.
