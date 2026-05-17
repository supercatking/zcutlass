#!/usr/bin/env python3
"""Validate the explicit vLLM Linear adapter path.

This is not an end-to-end vLLM serving benchmark. It proves the next integration
step: a vLLM process can load the zcutlass plugin and an explicit Linear callsite
can route through ZCutlassVllmLinearAdapter with observable hit/fallback stats.
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
    parser.add_argument("--rtol", type=float, default=5.0e-2)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--require-hit", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root() / "python"))

    import torch
    import torch.nn.functional as F
    import vllm.plugins
    import zcutlass_vllm
    from zcutlass_torch import extension_available
    from zcutlass_vllm import ZCutlassVllmLinearAdapter

    if not torch.cuda.is_available():
        raise SystemExit("FAIL: CUDA is unavailable")
    if not extension_available():
        raise SystemExit("FAIL: zcutlass_torch extension is unavailable")

    plugin_registered_before = zcutlass_vllm.is_registered()
    vllm.plugins.load_general_plugins()
    plugin_registered_after = zcutlass_vllm.is_registered()
    if not plugin_registered_after:
        raise SystemExit("FAIL: vLLM plugin loader did not register zcutlass")

    dtype = torch_dtype(torch, args.dtype)
    x = torch.randn((args.m, args.k), device="cuda", dtype=dtype) * 0.25
    weight = torch.randn((args.n, args.k), device="cuda", dtype=dtype) * 0.10
    bias = torch.randn((args.n,), device="cuda", dtype=dtype) * 0.01
    weight_t = weight.t().contiguous()

    promoted_families = tuple(args.allow_family or ["prefill"])
    adapter = ZCutlassVllmLinearAdapter(
        promoted_families=promoted_families,
        materialize_inputs=True,
    )

    expected = F.linear(x, weight, bias)
    actual = adapter(x, weight_t, bias, weight_is_transposed=True)
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)

    def stock_fn():
        return F.linear(x, weight, bias)

    def adapter_fn():
        return adapter(x, weight_t, bias, weight_is_transposed=True)

    stock_ms = cuda_time_ms(torch, stock_fn, args.warmup, args.iterations)
    adapter_ms = cuda_time_ms(torch, adapter_fn, args.warmup, args.iterations)
    flops = 2.0 * args.m * args.n * args.k
    trace: dict[str, Any] = dict(adapter.last_trace or {})
    result = {
        "schema_version": 1,
        "operation": "vllm_explicit_linear_adapter",
        "problem": {
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "dtype": args.dtype,
            "bias": True,
        },
        "environment": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "plugin_registered_before_load": plugin_registered_before,
            "plugin_registered_after_load": plugin_registered_after,
            "extension_available": extension_available(),
        },
        "routing": {
            "hit_rate": adapter.stats.hit_rate,
            "hits": adapter.stats.hits,
            "misses": adapter.stats.misses,
            "fallback_reasons": dict(adapter.stats.fallback_reasons),
            "last_trace": trace,
        },
        "performance": {
            "warmup_iterations": args.warmup,
            "profiling_iterations": args.iterations,
            "stock_ms": stock_ms,
            "adapter_ms": adapter_ms,
            "speedup_vs_stock": stock_ms / adapter_ms if adapter_ms else 0.0,
            "stock_tflops": flops / (stock_ms * 1.0e-3) / 1.0e12,
            "adapter_tflops": flops / (adapter_ms * 1.0e-3) / 1.0e12,
        },
    }

    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")

    if args.require_hit and adapter.stats.hits == 0:
        raise SystemExit("FAIL: adapter did not route any call to zcutlass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
