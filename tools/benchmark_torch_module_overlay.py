#!/usr/bin/env python3
"""Benchmark a tiny torch.nn.Module decoder block with zcutlass overlay.

This harness is the bridge between synthetic GEMM callsites and real framework
integration. It uses normal torch.nn.Linear modules for the stock path, then
builds an overlay module that shares the same weights but routes each Linear
callsite through ZCutlassGemmOverlay with pre-transposed weights.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModuleConfig:
    name: str
    m: int
    hidden: int
    intermediate: int
    dtype: str


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def torch_dtype(torch, dtype: str):
    return torch.float16 if dtype == "f16" else torch.bfloat16


def suite_configs(suite: str, dtypes: list[str]) -> list[ModuleConfig]:
    if suite == "smoke":
        base = [
            ("smoke_decode", 8, 512, 2048),
            ("smoke_prefill", 128, 1024, 4096),
        ]
    elif suite == "llm-v1.5":
        base = [
            ("decode", 8, 4096, 16384),
            ("prefill", 128, 4096, 16384),
        ]
    else:
        raise SystemExit(f"Unknown suite '{suite}'")
    return [ModuleConfig(name, m, hidden, intermediate, dtype) for dtype in dtypes for name, m, hidden, intermediate in base]


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


def shape_family(m: int, n: int, k: int) -> str:
    if m <= 16 and n >= 1024 and k >= 1024:
        return "decode"
    if 32 <= m <= 256 and n >= 1024 and k >= 1024:
        return "prefill"
    if m >= 512 and n >= 1024 and k >= 1024:
        return "large"
    return "fallback"


def module_shapes(config: ModuleConfig) -> list[tuple[str, int, int, int]]:
    h = config.hidden
    i = config.intermediate
    return [
        ("qkv", config.m, 3 * h, h),
        ("o_proj", config.m, h, h),
        ("mlp_up_gate", config.m, 2 * i, h),
        ("mlp_down", config.m, h, i),
    ]


def total_flops(config: ModuleConfig) -> float:
    return sum(2.0 * m * n * k for _name, m, n, k in module_shapes(config))


def make_stock_block(torch, config: ModuleConfig, bias: bool):
    dtype = torch_dtype(torch, config.dtype)
    h = config.hidden
    i = config.intermediate

    class StockBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = torch.nn.Linear(h, 3 * h, bias=bias, device="cuda", dtype=dtype)
            self.o_proj = torch.nn.Linear(h, h, bias=bias, device="cuda", dtype=dtype)
            self.mlp_up_gate = torch.nn.Linear(h, 2 * i, bias=bias, device="cuda", dtype=dtype)
            self.mlp_down = torch.nn.Linear(i, h, bias=bias, device="cuda", dtype=dtype)

        def forward(self, x):
            qkv = self.qkv(x)
            # Keep the attention side synthetic but deterministic: take the value slice.
            v = qkv[:, 2 * h :]
            attn_out = self.o_proj(v)
            up_gate = self.mlp_up_gate(x)
            up, gate = up_gate[:, :i], up_gate[:, i:]
            mlp = self.mlp_down(torch.nn.functional.silu(gate) * up)
            return attn_out + mlp

    return StockBlock().eval()


def make_overlay_block(torch, stock, overlay_cls, routing_policy_cls, config: ModuleConfig, args: argparse.Namespace):
    h = config.hidden
    i = config.intermediate

    def weight_t(module):
        return module.weight.detach().t().contiguous()

    class OverlayBlock(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.overlay = overlay_cls(
                enable_zcutlass=not args.disable_zcutlass,
                routing_policy=routing_policy_cls(
                    enabled=not args.disable_routing_policy,
                    promoted_families=tuple(args.allow_family),
                ),
                force_zcutlass=args.force_zcutlass,
            )
            self.weights = {
                "qkv": weight_t(stock.qkv),
                "o_proj": weight_t(stock.o_proj),
                "mlp_up_gate": weight_t(stock.mlp_up_gate),
                "mlp_down": weight_t(stock.mlp_down),
            }
            self.biases = {
                "qkv": stock.qkv.bias,
                "o_proj": stock.o_proj.bias,
                "mlp_up_gate": stock.mlp_up_gate.bias,
                "mlp_down": stock.mlp_down.bias,
            }
            self.trace: list[dict[str, Any]] = []

        def linear(self, name: str, x):
            materialized_input = False
            if args.materialize_overlay_inputs and not x.is_contiguous():
                x = x.contiguous()
                materialized_input = True
            y = self.overlay.linear(
                x,
                self.weights[name],
                self.biases[name],
                weight_is_transposed=True,
            )
            self.trace.append(
                {
                    "module": name,
                    "family": self.overlay.last_family,
                    "path": self.overlay.last_kernel_path,
                    "kernel_name": self.overlay.last_kernel_name,
                    "tile": self.overlay.last_tile,
                    "selected_config": self.overlay.last_config,
                    "fallback_reason": self.overlay.last_fallback_reason,
                    "materialized_input": materialized_input,
                }
            )
            return y

        def forward(self, x):
            self.trace = []
            qkv = self.linear("qkv", x)
            v = qkv[:, 2 * h :]
            attn_out = self.linear("o_proj", v)
            up_gate = self.linear("mlp_up_gate", x)
            up, gate = up_gate[:, :i], up_gate[:, i:]
            mlp = self.linear("mlp_down", torch.nn.functional.silu(gate) * up)
            return attn_out + mlp

    return OverlayBlock().eval()


def run_config(torch, overlay_cls, routing_policy_cls, config: ModuleConfig, args: argparse.Namespace) -> list[dict[str, Any]]:
    dtype = torch_dtype(torch, config.dtype)
    x = torch.randn((config.m, config.hidden), device="cuda", dtype=dtype) * 0.25
    stock = make_stock_block(torch, config, args.bias)
    overlay = make_overlay_block(torch, stock, overlay_cls, routing_policy_cls, config, args)

    def stock_fn():
        return stock(x)

    def overlay_fn():
        return overlay(x)

    expected = stock_fn()
    actual = overlay_fn()
    torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
    stock_ms = cuda_time_ms(torch, stock_fn, args.warmup, args.iterations)
    overlay_ms = cuda_time_ms(torch, overlay_fn, args.warmup, args.iterations)
    flops = total_flops(config)
    module_trace = list(overlay.trace)
    fallback_reasons = overlay.overlay.stats.fallback_reasons
    total_routes = overlay.overlay.stats.total

    common_problem = {
        "operation": "mini_decoder_block",
        "m": config.m,
        "hidden": config.hidden,
        "intermediate": config.intermediate,
        "dtype": config.dtype,
        "module_count": 4,
        "bias": bool(args.bias),
    }
    common_tags = {
        "suite": args.suite,
        "framework": "pytorch",
        "callsite": "torch_nn_module_mini_decoder",
        "layer": config.name,
    }
    return [
        {
            "schema_version": 1,
            "problem": common_problem,
            "provider": "pytorch_stock",
            "status": "success",
            "kernel": "torch.nn.Module",
            "performance": {
                "warmup_iterations": args.warmup,
                "profiling_iterations": args.iterations,
                "median_ms": stock_ms,
                "tflops": flops / (stock_ms * 1.0e-3) / 1.0e12,
            },
            "tags": common_tags,
        },
        {
            "schema_version": 1,
            "problem": common_problem,
            "provider": "zcutlass_overlay",
            "status": "success",
            "kernel": "ZCutlassGemmOverlay.module",
            "performance": {
                "warmup_iterations": args.warmup,
                "profiling_iterations": args.iterations,
                "median_ms": overlay_ms,
                "tflops": flops / (overlay_ms * 1.0e-3) / 1.0e12,
            },
            "tags": {
                **common_tags,
                "shape_family": shape_family(config.m, config.hidden, config.hidden),
                "hit_count": overlay.overlay.stats.hits,
                "miss_count": overlay.overlay.stats.misses,
                "hit_rate": overlay.overlay.stats.hit_rate,
                "fallback_reasons": fallback_reasons,
                "module_trace": module_trace,
                "module_route_count": total_routes,
                "materialize_overlay_inputs": args.materialize_overlay_inputs,
                "force_zcutlass": args.force_zcutlass,
                "routing_policy_enabled": not args.disable_routing_policy,
                "promoted_families": args.allow_family,
            },
        },
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="smoke", choices=("smoke", "llm-v1.5"))
    parser.add_argument("--dtype", default="f16", choices=("f16", "bf16", "both"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--disable-zcutlass", action="store_true")
    parser.add_argument("--disable-routing-policy", action="store_true")
    parser.add_argument("--force-zcutlass", action="store_true")
    parser.add_argument(
        "--materialize-overlay-inputs",
        action="store_true",
        help="Call contiguous() on non-contiguous overlay inputs before routing to zcutlass.",
    )
    parser.add_argument(
        "--allow-family",
        action="append",
        default=[],
        choices=("decode", "prefill", "large"),
        help="Promote a shape family into the zcutlass path for policy-gated runs.",
    )
    parser.add_argument("--require-extension", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--output", type=pathlib.Path, default=repo_root() / "build" / "torch_module_overlay.jsonl")
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
    records: list[dict[str, Any]] = []
    for config in suite_configs(args.suite, dtypes):
        config_records = run_config(torch, ZCutlassGemmOverlay, RoutingPolicy, config, args)
        records.extend(config_records)
        if args.summary:
            stock_ms = config_records[0]["performance"]["median_ms"]
            overlay_ms = config_records[1]["performance"]["median_ms"]
            tags = config_records[1]["tags"]
            speedup = stock_ms / overlay_ms if overlay_ms > 0 else 0.0
            print(
                f"{config.name} {config.dtype} m={config.m} h={config.hidden} i={config.intermediate} "
                f"stock={stock_ms:.4f} ms overlay={overlay_ms:.4f} ms speedup={speedup:.3f}x "
                f"hit_rate={tags['hit_rate']:.2f} fallbacks={tags['fallback_reasons']}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
