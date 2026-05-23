#!/usr/bin/env python3
"""Bounded vLLM real-model overlay smoke runner.

This is a bridge check between LinearMethod probes and full serving benchmarks.
It enables the zcutlass vLLM OOT Linear overlay, optionally loads one
user-specified model through the vLLM ``LLM`` API, runs one deterministic
generation, and summarizes the per-layer route JSONL.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import pathlib
import sys
import traceback
from typing import Any


DEFAULT_PROMPT = (
    "zcutlass overlay smoke prompt: alpha beta gamma delta epsilon zeta eta "
    "theta iota kappa lambda mu nu xi omicron pi rho sigma tau upsilon phi "
    "chi psi omega. Report one concise sentence about deterministic routing."
)
DEFAULT_LAYER_FILTER = "qkv_proj,gate_up_proj,down_proj,o_proj"
DEFAULT_ALLOW_FAMILIES = "prefill"


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def add_repo_python_to_path(root: pathlib.Path) -> None:
    python_dir = str(root / "python")
    if python_dir not in sys.path:
        sys.path.insert(0, python_dir)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [part for part in existing.split(os.pathsep) if part]
    if python_dir not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([python_dir, *parts])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded vLLM real-model zcutlass overlay smoke check."
    )
    parser.add_argument(
        "--model",
        help=(
            "Optional Hugging Face model id or local model path. When omitted, "
            "the check stops after vLLM import and OOT registration."
        ),
    )
    parser.add_argument("--tokenizer", help="Optional tokenizer id/path.")
    parser.add_argument("--revision", help="Optional model revision.")
    parser.add_argument("--tokenizer-revision", help="Optional tokenizer revision.")
    parser.add_argument("--download-dir", type=pathlib.Path, help="Optional HF download/cache dir.")
    parser.add_argument(
        "--allow-downloads",
        action="store_true",
        help="Allow vLLM/Hugging Face to fetch missing model files.",
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("auto", "half", "float16", "bfloat16", "float", "float32"),
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--route-log",
        type=pathlib.Path,
        default=repo_root() / "build" / "reports" / "vllm-real-model-overlay-routes.jsonl",
    )
    parser.add_argument(
        "--append-route-log",
        action="store_true",
        help="Append to an existing route log instead of resetting it first.",
    )
    parser.add_argument("--allow-families", default=DEFAULT_ALLOW_FAMILIES)
    parser.add_argument("--layer-filter", default=DEFAULT_LAYER_FILTER)
    parser.add_argument("--require-vllm", action="store_true")
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Fail if --model is omitted or the model cannot generate.",
    )
    parser.add_argument(
        "--require-route-rows",
        action="store_true",
        help="Fail after generation if no zcutlass route log rows were written.",
    )
    parser.add_argument(
        "--show-exception",
        action="store_true",
        help="Print tracebacks for skipped vLLM/model import and load failures.",
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace, root: pathlib.Path) -> pathlib.Path:
    add_repo_python_to_path(root)
    route_log = args.route_log.expanduser().resolve()
    route_log.parent.mkdir(parents=True, exist_ok=True)
    if not args.append_route_log:
        route_log.write_text("", encoding="utf-8")

    os.environ["VLLM_PLUGINS"] = "zcutlass_overlay"
    os.environ["ZCUTLASS_VLLM_ENABLE"] = "1"
    os.environ["ZCUTLASS_VLLM_ALLOW_FAMILIES"] = args.allow_families
    os.environ["ZCUTLASS_VLLM_LAYER_FILTER"] = args.layer_filter
    os.environ["ZCUTLASS_VLLM_LOG_ROUTES"] = "1"
    os.environ["ZCUTLASS_VLLM_ROUTE_LOG"] = str(route_log)
    if args.model:
        os.environ["ZCUTLASS_VLLM_MODEL_ID"] = args.model

    if not args.allow_downloads:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    return route_log


def skip_or_fail(message: str, *, required: bool) -> int:
    prefix = "FAIL" if required else "SKIP"
    print(f"{prefix}: {message}")
    return 1 if required else 0


def maybe_traceback(args: argparse.Namespace) -> None:
    if args.show_exception:
        traceback.print_exc()


def load_vllm(args: argparse.Namespace):
    try:
        import vllm.plugins
        from vllm import LLM, SamplingParams
    except Exception as exc:
        maybe_traceback(args)
        required = args.require_vllm or args.require_model
        return None, None, skip_or_fail(f"vLLM unavailable: {exc}", required=required)

    return (vllm.plugins, LLM, SamplingParams), None, None


def register_overlay(vllm_plugins: Any) -> dict[str, Any]:
    import zcutlass_vllm
    from zcutlass_vllm.plugin import state as plugin_state

    loader_error = ""
    try:
        vllm_plugins.load_general_plugins()
    except Exception as exc:
        loader_error = str(exc)
    zcutlass_vllm.register()
    state = plugin_state()
    return {
        "plugin_registered": zcutlass_vllm.is_registered(),
        "extension_available": state.extension_available,
        "model_registered": state.model_registered,
        "oot_linear_registered": state.oot_linear_registered,
        "loader_error": loader_error,
        "message": state.message,
    }


def summarize_route_log(path: pathlib.Path) -> dict[str, Any]:
    route_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    families: Counter[str] = Counter()
    layer_classes: Counter[str] = Counter()
    rows = 0

    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    route_counts["invalid_json"] += 1
                    continue
                route = str(row.get("route") or "unknown")
                route_counts[route] += 1
                families[str(row.get("shape_family") or "unknown")] += 1
                layer_classes[str(row.get("layer_class") or "unknown")] += 1
                if route == "fallback":
                    reason = str(row.get("fallback_reason") or "unknown")
                    fallback_reasons[reason] += 1

    return {
        "path": str(path),
        "rows": rows,
        "route_counts": dict(sorted(route_counts.items())),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "shape_families": dict(sorted(families.items())),
        "layer_classes": dict(sorted(layer_classes.items())),
    }


def print_route_summary(summary: dict[str, Any]) -> None:
    route_counts = summary["route_counts"]
    hits = int(route_counts.get("zcutlass", 0))
    fallbacks = int(route_counts.get("fallback", 0))
    others = sum(int(value) for key, value in route_counts.items() if key not in {"zcutlass", "fallback"})
    print(f"route_log={summary['path']}")
    print(
        "routes: "
        f"rows={summary['rows']} zcutlass_hits={hits} fallbacks={fallbacks} other={others}"
    )
    if summary["fallback_reasons"]:
        print(f"fallback_reasons={json.dumps(summary['fallback_reasons'], sort_keys=True)}")
    if summary["shape_families"]:
        print(f"shape_families={json.dumps(summary['shape_families'], sort_keys=True)}")


def build_llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": args.dtype,
        "seed": args.seed,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cpu_offload_gb": args.cpu_offload_gb,
        "max_model_len": args.max_model_len,
        "trust_remote_code": args.trust_remote_code,
        "enforce_eager": True,
    }
    if args.tokenizer:
        kwargs["tokenizer"] = args.tokenizer
    if args.revision:
        kwargs["revision"] = args.revision
    if args.tokenizer_revision:
        kwargs["tokenizer_revision"] = args.tokenizer_revision
    if args.download_dir:
        kwargs["download_dir"] = str(args.download_dir.expanduser())
    return kwargs


def generate_once(args: argparse.Namespace, LLM: Any, SamplingParams: Any) -> str:
    llm = LLM(**build_llm_kwargs(args))
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    outputs = llm.generate([args.prompt], sampling)
    if not outputs or not getattr(outputs[0], "outputs", None):
        return ""
    return str(outputs[0].outputs[0].text)


def main() -> int:
    args = parse_args()
    root = repo_root()
    route_log = configure_environment(args, root)

    loaded, _, status = load_vllm(args)
    if status is not None:
        print_route_summary(summarize_route_log(route_log))
        return int(status)

    vllm_plugins, LLM, SamplingParams = loaded
    registration = register_overlay(vllm_plugins)
    print(f"registration={json.dumps(registration, sort_keys=True, default=str)}")
    if not registration["plugin_registered"] or not registration["oot_linear_registered"]:
        print_route_summary(summarize_route_log(route_log))
        return skip_or_fail(
            "zcutlass vLLM OOT Linear registration unavailable",
            required=args.require_vllm or args.require_model,
        )

    if not args.model:
        print_route_summary(summarize_route_log(route_log))
        return skip_or_fail("no --model supplied; model generation skipped", required=args.require_model)

    try:
        generated = generate_once(args, LLM, SamplingParams)
        preview = generated.replace("\n", "\\n")[:160]
        print(f"generated_chars={len(generated)} generated_preview={json.dumps(preview)}")
    except Exception as exc:
        maybe_traceback(args)
        print_route_summary(summarize_route_log(route_log))
        return skip_or_fail(
            f"model unavailable or generation failed for {args.model!r}: {exc}",
            required=args.require_model,
        )

    summary = summarize_route_log(route_log)
    print_route_summary(summary)
    if args.require_route_rows and int(summary["rows"]) == 0:
        return skip_or_fail("generation completed but route log has no rows", required=True)
    print("PASS: vLLM real-model overlay smoke completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
