#!/usr/bin/env python3
"""Benchmark a synthetic LLM layer with the zcutlass PyTorch overlay.

The harness models the GEMM-heavy Linear calls in one decoder layer:
QKV projection, output projection, MLP up/gate projection, and MLP down
projection. It is still synthetic, but it preserves the callsite shape pattern
needed before integrating with SGLang or vLLM.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class LayerConfig:
    name: str
    m: int
    hidden: int
    intermediate: int
    dtype: str


@dataclass(frozen=True)
class ModuleCase:
    module: str
    m: int
    n: int
    k: int
    dtype: str
    layer: str

    @property
    def label(self) -> str:
        return f"{self.layer}/{self.module} {self.dtype} {self.m}x{self.n}x{self.k}"


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


def tflops(case: ModuleCase, ms: float) -> float:
    return 2.0 * case.m * case.n * case.k / (ms * 1.0e-3) / 1.0e12


def shape_family(m: int, n: int, k: int) -> str:
    if m <= 16 and n >= 1024 and k >= 1024:
        return "decode"
    if 32 <= m <= 256 and n >= 1024 and k >= 1024:
        return "prefill"
    if m >= 512 and n >= 1024 and k >= 1024:
        return "large"
    return "fallback"


def suite_configs(suite: str, dtypes: list[str]) -> list[LayerConfig]:
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
    return [LayerConfig(name, m, hidden, intermediate, dtype) for dtype in dtypes for (name, m, hidden, intermediate) in base]


def module_cases(config: LayerConfig) -> list[ModuleCase]:
    h = config.hidden
    i = config.intermediate
    return [
        ModuleCase("qkv", config.m, 3 * h, h, config.dtype, config.name),
        ModuleCase("o_proj", config.m, h, h, config.dtype, config.name),
        ModuleCase("mlp_up_gate", config.m, 2 * i, h, config.dtype, config.name),
        ModuleCase("mlp_down", config.m, h, i, config.dtype, config.name),
    ]


def make_linear_inputs(torch, case: ModuleCase, bias: bool):
    dtype = torch_dtype(torch, case.dtype)
    x = torch.randn((case.m, case.k), device="cuda", dtype=dtype) * 0.25
    weight = torch.randn((case.n, case.k), device="cuda", dtype=dtype) * 0.25
    weight_t = weight.t().contiguous()
    bias_tensor = torch.randn((case.n,), device="cuda", dtype=dtype) * 0.25 if bias else None
    return x, weight, weight_t, bias_tensor


def run_module(torch, overlay_cls, routing_policy_cls, case: ModuleCase, args: argparse.Namespace) -> list[dict]:
    x, weight, weight_t, bias = make_linear_inputs(torch, case, args.bias)

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
        "shape_family": shape_family(case.m, case.n, case.k),
        "framework": "pytorch",
        "callsite": "synthetic_llm_layer",
        "layer": case.layer,
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


def aggregate_records(records: list[dict], args: argparse.Namespace) -> list[dict]:
    grouped: dict[tuple[str, str, str], dict[str, float | int | dict[str, int]]] = {}
    for record in records:
        tags = record["tags"]
        key = (record["provider"], tags["layer"], record["problem"]["dtype"])
        entry = grouped.setdefault(
            key,
            {
                "median_ms": 0.0,
                "tflops_numerator": 0.0,
                "hits": 0,
                "misses": 0,
                "fallback_reasons": {},
            },
        )
        entry["median_ms"] = float(entry["median_ms"]) + float(record["performance"]["median_ms"])
        entry["tflops_numerator"] = float(entry["tflops_numerator"]) + (
            2.0 * record["problem"]["m"] * record["problem"]["n"] * record["problem"]["k"]
        )
        if record["provider"] == "zcutlass_overlay":
            entry["hits"] = int(entry["hits"]) + int(tags["hit_count"])
            entry["misses"] = int(entry["misses"]) + int(tags["miss_count"])
            reasons = entry["fallback_reasons"]
            assert isinstance(reasons, dict)
            for reason, count in tags["fallback_reasons"].items():
                reasons[reason] = reasons.get(reason, 0) + int(count)

    out: list[dict] = []
    for (provider, layer, dtype), entry in grouped.items():
        median_ms = float(entry["median_ms"])
        total = int(entry["hits"]) + int(entry["misses"])
        out.append(
            {
                "schema_version": 1,
                "problem": {
                    "operation": "synthetic_llm_layer",
                    "dtype": dtype,
                    "layout": "linear_sequence",
                    "module_count": 4,
                },
                "provider": provider,
                "status": "success",
                "kernel": "synthetic_llm_layer",
                "performance": {
                    "warmup_iterations": args.warmup,
                    "profiling_iterations": args.iterations,
                    "median_ms": median_ms,
                    "tflops": float(entry["tflops_numerator"]) / (median_ms * 1.0e-3) / 1.0e12,
                },
                "tags": {
                    "suite": args.suite,
                    "framework": "pytorch",
                    "callsite": "synthetic_llm_layer",
                    "layer": layer,
                    "hit_count": int(entry["hits"]),
                    "miss_count": int(entry["misses"]),
                    "hit_rate": int(entry["hits"]) / total if total else 0.0,
                    "fallback_reasons": entry["fallback_reasons"],
                    "force_zcutlass": args.force_zcutlass,
                    "routing_policy_enabled": not args.disable_routing_policy,
                    "promoted_families": args.allow_family,
                },
            }
        )
    return out


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
        "--allow-family",
        action="append",
        default=[],
        choices=("decode", "prefill", "large"),
        help="Promote a shape family into the zcutlass path for policy-gated runs.",
    )
    parser.add_argument("--require-extension", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--output", type=pathlib.Path, default=repo_root() / "build" / "torch_llm_overlay.jsonl")
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
    records: list[dict] = []
    for config in suite_configs(args.suite, dtypes):
        for case in module_cases(config):
            case_records = run_module(torch, ZCutlassGemmOverlay, RoutingPolicy, case, args)
            records.extend(case_records)
            if args.summary:
                stock_ms = case_records[0]["performance"]["median_ms"]
                overlay_ms = case_records[1]["performance"]["median_ms"]
                speedup = stock_ms / overlay_ms if overlay_ms > 0 else 0.0
                tags = case_records[1]["tags"]
                print(
                    f"{case.label:42s} stock={stock_ms:.4f} ms overlay={overlay_ms:.4f} ms "
                    f"speedup={speedup:.3f}x hit_rate={tags['hit_rate']:.2f} "
                    f"path={tags['kernel_path']} reason={tags['fallback_reason'] or '-'}"
                )

    records.extend(aggregate_records(records, args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
