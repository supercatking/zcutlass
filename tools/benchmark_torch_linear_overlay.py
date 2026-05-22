#!/usr/bin/env python3
"""Benchmark an explicit PyTorch Linear callsite against zcutlass overlay.

This harness is closer to framework integration than raw GEMM: stock PyTorch
uses `torch.nn.functional.linear(x, weight, bias)` with weight shaped `[N, K]`,
while the zcutlass fast path receives a pre-transposed contiguous `[K, N]`
weight because zcutlass v1 only supports row-major B.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LinearCase:
    m: int
    n: int
    k: int
    dtype: str
    module: str

    @property
    def label(self) -> str:
        return f"{self.module} {self.dtype} {self.m}x{self.n}x{self.k}"


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def parse_shape(text: str, dtype: str, module: str) -> LinearCase:
    parts = text.lower().split("x")
    if len(parts) != 3:
        raise SystemExit(f"Invalid shape '{text}', expected MxNxK")
    return LinearCase(int(parts[0]), int(parts[1]), int(parts[2]), dtype, module)


def suite_cases(suite: str, dtypes: Iterable[str]) -> list[LinearCase]:
    if suite == "llm-v1.5":
        base = [
            (8, 4096, 4096, "decode_linear"),
            (128, 4096, 4096, "prefill_qkv_or_o"),
            (128, 16384, 4096, "prefill_mlp_up"),
            (128, 4096, 16384, "prefill_mlp_down"),
            (4096, 4096, 4096, "large_linear"),
        ]
    elif suite == "smoke":
        base = [
            (8, 256, 256, "smoke_decode"),
            (32, 512, 512, "smoke_prefill"),
            (128, 1024, 1024, "smoke_prefill"),
        ]
    else:
        raise SystemExit(f"Unknown suite '{suite}'")
    return [LinearCase(m, n, k, dtype, module) for dtype in dtypes for (m, n, k, module) in base]


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


def tflops(case: LinearCase, ms: float) -> float:
    return 2.0 * case.m * case.n * case.k / (ms * 1.0e-3) / 1.0e12


def shape_family(case: LinearCase) -> str:
    if case.m <= 16 and case.n >= 1024 and case.k >= 1024:
        return "decode"
    if 32 <= case.m <= 256 and case.n >= 1024 and case.k >= 1024:
        return "prefill"
    if case.m >= 512 and case.n >= 1024 and case.k >= 1024:
        return "large"
    return "fallback"


def run_case(torch, overlay_cls, routing_policy_cls, case: LinearCase, args: argparse.Namespace) -> list[dict]:
    dtype = torch_dtype(torch, case.dtype)
    x = torch.randn((case.m, case.k), device="cuda", dtype=dtype) * 0.25
    weight = torch.randn((case.n, case.k), device="cuda", dtype=dtype) * 0.25
    weight_t = weight.t().contiguous()
    bias = torch.randn((case.n,), device="cuda", dtype=dtype) * 0.25 if args.bias else None

    def stock_fn():
        return torch.nn.functional.linear(x, weight, bias)

    stock = stock_fn()
    stock_ms = cuda_time_ms(torch, stock_fn, args.warmup, args.iterations)

    overlay = overlay_cls(
        enable_zcutlass=not args.disable_zcutlass,
        routing_policy=routing_policy_cls(
            enabled=not args.disable_routing_policy,
            promoted_families=tuple(args.allow_family),
        ),
        force_zcutlass=args.force_zcutlass,
    )

    def overlay_fn():
        return overlay.linear(x, weight_t, bias, weight_is_transposed=True)

    actual = overlay_fn()
    kernel_path = overlay.last_kernel_path
    kernel_name = overlay.last_kernel_name
    selected_config = overlay.last_config
    route_family = overlay.last_family
    fallback_reason = overlay.last_fallback_reason
    torch.testing.assert_close(actual, stock, rtol=args.rtol, atol=args.atol)
    overlay_ms = cuda_time_ms(torch, overlay_fn, args.warmup, args.iterations)

    common_problem = {
        "operation": "linear",
        "m": case.m,
        "n": case.n,
        "k": case.k,
        "dtype": case.dtype,
        "layout": "x_row,weight_row,output_row",
        "weight_fast_path_layout": "pretransposed_row_k_by_n",
        "bias": bool(args.bias),
    }
    common_tags = {
        "suite": args.suite,
        "shape_family": shape_family(case),
        "framework": "pytorch",
        "callsite": args.callsite,
        "module": case.module,
    }
    return [
        {
            "schema_version": 1,
            "problem": common_problem,
            "provider": "pytorch_stock",
            "status": "success",
            "kernel": "torch.nn.functional.linear",
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
            "problem": common_problem,
            "provider": "zcutlass_overlay",
            "status": "success",
            "kernel": "ZCutlassGemmOverlay.linear",
            "performance": {
                "warmup_iterations": args.warmup,
                "profiling_iterations": args.iterations,
                "median_ms": overlay_ms,
                "tflops": tflops(case, overlay_ms),
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
                "weight_is_pretransposed": True,
            },
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="smoke", choices=("smoke", "llm-v1.5"))
    parser.add_argument("--shape", action="append", help="Explicit Linear MxNxK; may be repeated.")
    parser.add_argument("--dtype", default="f16", choices=("f16", "bf16", "both"))
    parser.add_argument("--module", default="explicit_linear")
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
    parser.add_argument("--callsite", default="synthetic_linear")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--output", type=pathlib.Path, default=repo_root() / "build" / "torch_linear_overlay.jsonl")
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

    dtypes = ["f16", "bf16"] if args.dtype == "both" else [args.dtype]
    cases = (
        [parse_shape(token, dtype, args.module) for dtype in dtypes for token in args.shape]
        if args.shape
        else suite_cases(args.suite, dtypes)
    )

    records: list[dict] = []
    for case in cases:
        case_records = run_case(torch, ZCutlassGemmOverlay, RoutingPolicy, case, args)
        records.extend(case_records)
        if args.summary:
            stock_ms = case_records[0]["performance"]["median_ms"]
            overlay_ms = case_records[1]["performance"]["median_ms"]
            speedup = stock_ms / overlay_ms if overlay_ms > 0 else 0.0
            tags = case_records[1]["tags"]
            print(
                f"{case.label:34s} stock={stock_ms:.4f} ms overlay={overlay_ms:.4f} ms "
                f"speedup={speedup:.3f}x hit_rate={tags['hit_rate']:.2f} "
                f"path={tags['kernel_path']} reason={tags['fallback_reason'] or '-'}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
