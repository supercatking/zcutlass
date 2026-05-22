#!/usr/bin/env python3
"""Benchmark stock PyTorch matmul against the zcutlass PyTorch overlay.

This is the M1 proof harness. It measures explicit GEMM callsites only; it does
not hook PyTorch globally. Results are schema-v1 JSONL so they can be archived
next to kernel-level zcutlass reports.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Shape:
    m: int
    n: int
    k: int
    dtype: str

    @property
    def label(self) -> str:
        return f"{self.dtype} {self.m}x{self.n}x{self.k}"


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def parse_shape(text: str, dtype: str) -> Shape:
    parts = text.lower().split("x")
    if len(parts) != 3:
        raise SystemExit(f"Invalid shape '{text}', expected MxNxK")
    return Shape(int(parts[0]), int(parts[1]), int(parts[2]), dtype)


def suite_shapes(suite: str, dtypes: Iterable[str]) -> list[Shape]:
    if suite == "llm-v1.5":
        base = [
            (8, 4096, 4096),
            (128, 4096, 4096),
            (128, 16384, 4096),
            (128, 4096, 16384),
            (4096, 4096, 4096),
        ]
    elif suite == "smoke":
        base = [(8, 256, 256), (32, 512, 512), (128, 1024, 1024)]
    else:
        raise SystemExit(f"Unknown suite '{suite}'")
    return [Shape(m, n, k, dtype) for dtype in dtypes for (m, n, k) in base]


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


def tflops(shape: Shape, ms: float) -> float:
    return 2.0 * shape.m * shape.n * shape.k / (ms * 1.0e-3) / 1.0e12


def shape_family(shape: Shape) -> str:
    if shape.m <= 16 and shape.n >= 1024 and shape.k >= 1024:
        return "decode"
    if 32 <= shape.m <= 256 and shape.n >= 1024 and shape.k >= 1024:
        return "prefill"
    if shape.m >= 512 and shape.n >= 512 and shape.k >= 512:
        return "large"
    return "fallback"


def run_shape(torch, overlay_cls, shape: Shape, args: argparse.Namespace) -> list[dict]:
    dtype = torch_dtype(torch, shape.dtype)
    a = torch.randn((shape.m, shape.k), device="cuda", dtype=dtype) * 0.25
    b = torch.randn((shape.k, shape.n), device="cuda", dtype=dtype) * 0.25
    bias = torch.randn((shape.n,), device="cuda", dtype=dtype) * 0.25 if args.bias else None

    def stock_fn():
        out = torch.matmul(a, b)
        return out + bias if bias is not None else out

    stock = stock_fn()
    stock_ms = cuda_time_ms(torch, stock_fn, args.warmup, args.iterations)

    overlay = overlay_cls(
        enable_zcutlass=not args.disable_zcutlass,
        routing_policy=args.routing_policy_cls(
            enabled=not args.disable_routing_policy,
            promoted_families=tuple(args.allow_family),
        ),
        force_zcutlass=args.force_zcutlass,
    )

    def overlay_fn():
        return overlay.gemm(a, b, bias=bias)

    actual = overlay_fn()
    kernel_path = overlay.last_kernel_path
    kernel_name = overlay.last_kernel_name
    selected_config = overlay.last_config
    route_family = overlay.last_family
    fallback_reason = overlay.last_fallback_reason
    torch.testing.assert_close(actual, stock, rtol=args.rtol, atol=args.atol)
    overlay_ms = cuda_time_ms(torch, overlay_fn, args.warmup, args.iterations)

    common_problem = {
        "operation": "gemm",
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "dtype": shape.dtype,
        "layout": "row,row,row,row",
        "alpha": 1.0,
        "beta": 0.0,
        "bias": bool(args.bias),
    }
    common_tags = {
        "suite": args.suite,
        "shape_family": shape_family(shape),
        "framework": "pytorch",
        "callsite": args.callsite,
    }
    return [
        {
            "schema_version": 1,
            "problem": common_problem,
            "provider": "pytorch_stock",
            "status": "success",
            "kernel": "torch.matmul",
            "performance": {
                "warmup_iterations": args.warmup,
                "profiling_iterations": args.iterations,
                "median_ms": stock_ms,
                "tflops": tflops(shape, stock_ms),
            },
            "tags": common_tags,
        },
        {
            "schema_version": 1,
            "problem": common_problem,
            "provider": "zcutlass_overlay",
            "status": "success",
            "kernel": "torch.ops.zcutlass_torch.gemm",
            "performance": {
                "warmup_iterations": args.warmup,
                "profiling_iterations": args.iterations,
                "median_ms": overlay_ms,
                "tflops": tflops(shape, overlay_ms),
            },
            "tags": {
                **common_tags,
                "route_family": route_family,
                "kernel_path": kernel_path,
                "kernel_name": kernel_name,
                "selected_config": selected_config,
                "fallback_reason": fallback_reason,
                "hit_count": overlay.stats.hits,
                "miss_count": overlay.stats.misses,
                "hit_rate": overlay.stats.hit_rate,
                "fallback_reasons": overlay.stats.fallback_reasons,
                "force_zcutlass": args.force_zcutlass,
                "routing_policy_enabled": not args.disable_routing_policy,
                "promoted_families": args.allow_family,
            },
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="smoke", choices=("smoke", "llm-v1.5"))
    parser.add_argument("--shape", action="append", help="Explicit shape MxNxK; may be repeated.")
    parser.add_argument("--dtype", default="f16", choices=("f16", "bf16", "both"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--disable-zcutlass", action="store_true")
    parser.add_argument("--disable-routing-policy", action="store_true")
    parser.add_argument("--force-zcutlass", action="store_true")
    parser.add_argument(
        "--allow-family",
        action="append",
        default=[],
        choices=("decode", "prefill", "large"),
        help="Promote a shape family into the zcutlass path for policy-gated runs.",
    )
    parser.add_argument("--require-extension", action="store_true")
    parser.add_argument("--callsite", default="synthetic_gemm")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--output", type=pathlib.Path, default=repo_root() / "build" / "torch_overlay.jsonl")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root() / "python"))
    try:
        import torch
    except Exception as exc:
        raise SystemExit(f"PyTorch is required for this benchmark: {exc}") from exc

    from zcutlass_torch import RoutingPolicy, ZCutlassGemmOverlay, extension_available

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch CUDA is not available")
    if args.require_extension and not extension_available():
        raise SystemExit("zcutlass_torch extension is not installed")
    args.routing_policy_cls = RoutingPolicy

    dtypes = ["f16", "bf16"] if args.dtype == "both" else [args.dtype]
    shapes = (
        [parse_shape(token, dtype) for dtype in dtypes for token in args.shape]
        if args.shape
        else suite_shapes(args.suite, dtypes)
    )

    records: list[dict] = []
    for shape in shapes:
        shape_records = run_shape(torch, ZCutlassGemmOverlay, shape, args)
        records.extend(shape_records)
        if args.summary:
            stock = shape_records[0]["performance"]["median_ms"]
            overlay = shape_records[1]["performance"]["median_ms"]
            speedup = stock / overlay if overlay > 0 else 0.0
            hit_rate = shape_records[1]["tags"]["hit_rate"]
            kernel_path = shape_records[1]["tags"]["kernel_path"]
            reason = shape_records[1]["tags"]["fallback_reason"] or "-"
            print(f"{shape.label:22s} stock={stock:.4f} ms overlay={overlay:.4f} ms "
                  f"speedup={speedup:.3f}x hit_rate={hit_rate:.2f} "
                  f"path={kernel_path} reason={reason}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
