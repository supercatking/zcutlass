#!/usr/bin/env python3
"""Benchmark vLLM UnquantizedLinearMethod with optional zcutlass routing."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LinearCase:
    name: str
    m: int
    n: int
    k: int
    dtype: str


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def torch_dtype(torch, dtype: str):
    return torch.float16 if dtype == "f16" else torch.bfloat16


def shape_family(m: int, n: int, k: int) -> str:
    if m <= 16 and n >= 1024 and k >= 1024:
        return "decode"
    if 32 <= m <= 256 and n >= 1024 and k >= 1024:
        return "prefill"
    if m >= 512 and n >= 1024 and k >= 1024:
        return "large"
    return "fallback"


def suite_cases(suite: str, dtypes: list[str]) -> list[LinearCase]:
    if suite == "smoke":
        base = [
            ("smoke_decode", 8, 512, 512),
            ("smoke_prefill", 128, 4096, 1024),
            ("smoke_prefill_bf16_target", 128, 4096, 4096),
        ]
    elif suite == "llm-v1.5":
        base = [
            ("decode", 8, 4096, 4096),
            ("prefill_square", 128, 4096, 4096),
            ("mlp_up", 128, 16384, 4096),
            ("mlp_down", 128, 4096, 16384),
            ("large_square", 4096, 4096, 4096),
        ]
    else:
        raise SystemExit(f"unknown suite: {suite}")
    return [
        LinearCase(name, m, n, k, dtype)
        for dtype in dtypes
        for name, m, n, k in base
    ]


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


def tflops(case: LinearCase, ms: float) -> float:
    flops = 2.0 * case.m * case.n * case.k
    return flops / (ms * 1.0e-3) / 1.0e12 if ms else 0.0


class DummyVllmLinear:
    pass


def make_layer(torch, weight, unquantized_method):
    layer = DummyVllmLinear()
    layer.weight = torch.nn.Parameter(weight)
    layer.quant_method = unquantized_method()
    return layer


def run_case(torch, case: LinearCase, args: argparse.Namespace, unquantized_method, install_method) -> list[dict[str, Any]]:
    dtype = torch_dtype(torch, case.dtype)
    x = torch.randn((case.m, case.k), device="cuda", dtype=dtype) * 0.25
    weight = torch.randn((case.n, case.k), device="cuda", dtype=dtype) * 0.10
    bias = torch.randn((case.n,), device="cuda", dtype=dtype) * 0.01 if args.bias else None

    layer = make_layer(torch, weight, unquantized_method)

    def stock_fn():
        return layer.quant_method.apply(layer, x, bias)

    expected = stock_fn()
    stock_ms = cuda_time_ms(torch, stock_fn, args.warmup, args.iterations)

    method = install_method(
        layer,
        promoted_families=tuple(args.allow_family),
        materialize_inputs=args.materialize_inputs,
    )

    def overlay_fn():
        return layer.quant_method.apply(layer, x, bias)

    actual = overlay_fn()
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
    overlay_ms = cuda_time_ms(torch, overlay_fn, args.warmup, args.iterations)
    trace = dict(method.last_trace or {})

    problem = {
        "operation": "vllm_unquantized_linear",
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "dtype": case.dtype,
        "bias": bool(args.bias),
    }
    common_tags = {
        "suite": args.suite,
        "case": case.name,
        "framework": "vllm",
        "shape_family": shape_family(case.m, case.n, case.k),
    }
    return [
        {
            "schema_version": 1,
            "problem": problem,
            "provider": "vllm_stock",
            "status": "success",
            "kernel": "vllm.UnquantizedLinearMethod",
            "performance": {
                "warmup_iterations": args.warmup,
                "profiling_iterations": args.iterations,
                "median_ms": stock_ms,
                "tflops": tflops(case, stock_ms),
            },
            "tags": common_tags,
        },
        {
            "schema_version": 1,
            "problem": problem,
            "provider": "zcutlass_vllm_overlay",
            "status": "success",
            "kernel": "ZCutlassUnquantizedLinearMethod",
            "performance": {
                "warmup_iterations": args.warmup,
                "profiling_iterations": args.iterations,
                "median_ms": overlay_ms,
                "tflops": tflops(case, overlay_ms),
            },
            "tags": {
                **common_tags,
                "hit_rate": method.stats.hit_rate,
                "hit_count": method.stats.hits,
                "miss_count": method.stats.misses,
                "fallback_reasons": dict(method.stats.fallback_reasons),
                "last_trace": trace,
                "promoted_families": args.allow_family,
                "materialize_inputs": args.materialize_inputs,
                "speedup_vs_stock": stock_ms / overlay_ms if overlay_ms else 0.0,
            },
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="smoke", choices=("smoke", "llm-v1.5"))
    parser.add_argument("--dtype", default="f16", choices=("f16", "bf16", "both"))
    parser.add_argument("--allow-family", action="append", default=[])
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--materialize-inputs", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rtol", type=float, default=5.0e-2)
    parser.add_argument("--atol", type=float, default=5.0e-2)
    parser.add_argument("--require-hit", action="store_true")
    parser.add_argument("--output", type=pathlib.Path, default=repo_root() / "build" / "vllm_linear_method.jsonl")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root() / "python"))

    import torch
    import vllm.plugins
    import zcutlass_vllm
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from zcutlass_torch import extension_available
    from zcutlass_vllm import install_zcutlass_unquantized_linear_method

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not extension_available():
        raise SystemExit("zcutlass_torch extension is unavailable")

    vllm.plugins.load_general_plugins()
    if not zcutlass_vllm.is_registered():
        raise SystemExit("vLLM plugin loader did not register zcutlass")

    dtypes = ["f16", "bf16"] if args.dtype == "both" else [args.dtype]
    records: list[dict[str, Any]] = []
    for case in suite_cases(args.suite, dtypes):
        case_records = run_case(
            torch,
            case,
            args,
            UnquantizedLinearMethod,
            install_zcutlass_unquantized_linear_method,
        )
        records.extend(case_records)
        if args.summary:
            stock = case_records[0]["performance"]["median_ms"]
            overlay = case_records[1]["performance"]["median_ms"]
            tags = case_records[1]["tags"]
            trace = tags["last_trace"]
            print(
                f"{case.name} {case.dtype} m={case.m} n={case.n} k={case.k} "
                f"stock={stock:.4f} ms overlay={overlay:.4f} ms "
                f"speedup={stock / overlay if overlay else 0.0:.3f}x "
                f"hit_rate={tags['hit_rate']:.2f} "
                f"kernel={trace.get('kernel_name', '-')}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote {args.output}")

    if args.require_hit:
        overlay_records = [record for record in records if record["provider"] == "zcutlass_vllm_overlay"]
        if not any(record["tags"]["hit_count"] > 0 for record in overlay_records):
            raise SystemExit("no zcutlass hits were recorded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
