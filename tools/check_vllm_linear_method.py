#!/usr/bin/env python3
"""Validate the vLLM LinearMethod wrapper for zcutlass routing.

This is still a focused integration probe, not an end-to-end serving benchmark.
It proves that a vLLM-style unquantized Linear layer can replace its
`quant_method` with an explicit zcutlass wrapper, route eligible shapes through
zcutlass, and delegate misses back to vLLM's native unquantized GEMM path.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def torch_dtype(torch, dtype: str):
    return torch.float16 if dtype == "f16" else torch.bfloat16


def cuda_time_ms(torch, fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    samples: list[float] = []
    for _ in range(iterations):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
    samples.sort()
    return samples[len(samples) // 2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--dtype", choices=("f16", "bf16"), default="f16")
    parser.add_argument("--allow-family", action="append", default=[])
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--rtol", type=float, default=5.0e-2)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--require-hit", action="store_true")
    parser.add_argument("--require-fallback", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root() / "python"))

    import torch
    import vllm.plugins
    import zcutlass_vllm
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from zcutlass_torch import extension_available
    from zcutlass_vllm import install_zcutlass_unquantized_linear_method

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA is unavailable")
    if args.require_hit and not extension_available():
        raise SystemExit("FAIL: zcutlass_torch extension is unavailable")

    vllm.plugins.load_general_plugins()
    if not zcutlass_vllm.is_registered():
        raise SystemExit("FAIL: vLLM plugin loader did not register zcutlass")

    dtype = torch_dtype(torch, args.dtype)
    x = torch.randn((args.m, args.k), device="cuda", dtype=dtype) * 0.25
    weight = torch.randn((args.n, args.k), device="cuda", dtype=dtype) * 0.10
    bias = torch.randn((args.n,), device="cuda", dtype=dtype) * 0.01 if args.bias else None

    class DummyVllmLinear(torch.nn.Module):
        def __init__(self, linear_weight) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(linear_weight)
            self.quant_method = UnquantizedLinearMethod()

        def forward(self, value, linear_bias=None):
            return self.quant_method.apply(self, value, linear_bias)

    layer = DummyVllmLinear(weight)
    expected = layer(x, bias)
    stock_ms = cuda_time_ms(torch, lambda: layer(x, bias), args.warmup, args.iterations)

    promoted_families = tuple(args.allow_family or ["prefill"])
    method = install_zcutlass_unquantized_linear_method(
        layer,
        promoted_families=promoted_families,
        materialize_inputs=True,
    )
    actual = layer(x, bias)
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
    overlay_ms = cuda_time_ms(torch, lambda: layer(x, bias), args.warmup, args.iterations)

    trace: dict[str, Any] = dict(method.last_trace or {})
    result = {
        "schema_version": 1,
        "operation": "vllm_unquantized_linear_method",
        "problem": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "dtype": args.dtype,
            "bias": bool(args.bias),
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm_plugin_registered": zcutlass_vllm.is_registered(),
            "extension_available": extension_available(),
            "gpu": torch.cuda.get_device_name(0),
        },
        "routing": {
            "hit_rate": method.stats.hit_rate,
            "hits": method.stats.hits,
            "misses": method.stats.misses,
            "fallback_reasons": dict(method.stats.fallback_reasons),
            "last_trace": trace,
            "promoted_families": promoted_families,
        },
        "performance": {
            "warmup_iterations": args.warmup,
            "profiling_iterations": args.iterations,
            "stock_ms": stock_ms,
            "overlay_ms": overlay_ms,
            "speedup_vs_stock": stock_ms / overlay_ms if overlay_ms else 0.0,
        },
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")

    if args.require_hit and method.stats.hits == 0:
        raise SystemExit("FAIL: LinearMethod wrapper did not route any call to zcutlass")
    if args.require_fallback and method.stats.misses == 0:
        raise SystemExit("FAIL: LinearMethod wrapper did not record a fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
