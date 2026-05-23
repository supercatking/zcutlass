"""JSONL route logging for vLLM real-model overlay experiments."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


def env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value not in ("", "0", "false", "False", "FALSE", "no", "No", "NO")


def csv_env(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return tuple(part.strip() for part in value.split(",") if part.strip())


def tensor_shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    return [int(dim) for dim in shape] if shape is not None else []


def tensor_dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else "unknown"


def first_tensor(value: Any) -> Any:
    if isinstance(value, tuple) and value:
        return value[0]
    return value


def shape_family(m: int, n: int, k: int) -> str:
    if m <= 16 and n >= 1024 and k >= 1024:
        return "decode"
    if 32 <= m <= 256 and n >= 1024 and k >= 1024:
        return "prefill"
    if m >= 512 and n >= 1024 and k >= 1024:
        return "large"
    return "fallback"


class RouteLogger:
    """Small append-only JSONL logger."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def write(self, row: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def make_route_logger() -> RouteLogger | None:
    if not env_enabled("ZCUTLASS_VLLM_LOG_ROUTES"):
        return None
    path = os.environ.get("ZCUTLASS_VLLM_ROUTE_LOG")
    if not path:
        return None
    return RouteLogger(path)


def build_route_row(
    *,
    layer: Any,
    layer_class: str,
    input_tensor: Any,
    output: Any,
    latency_us: float,
    trace: dict[str, Any] | None,
    install_error: str | None = None,
) -> dict[str, Any]:
    trace = trace or {}
    output_tensor = first_tensor(output)
    input_shape = tensor_shape(input_tensor)
    weight = getattr(layer, "weight", None)
    weight_shape = tensor_shape(weight)
    output_shape = tensor_shape(output_tensor)
    m = int(input_shape[0]) if len(input_shape) >= 2 else 0
    k = int(input_shape[-1]) if len(input_shape) >= 2 else 0
    n = int(weight_shape[0]) if len(weight_shape) >= 2 else 0
    fallback_reason = trace.get("fallback_reason")
    route = "zcutlass" if fallback_reason is None and trace.get("hit_count", 0) else "fallback"
    return {
        "ts_ns": time.time_ns(),
        "pid": os.getpid(),
        "rank": os.environ.get("RANK", ""),
        "local_rank": os.environ.get("LOCAL_RANK", ""),
        "tp_rank": getattr(layer, "tp_rank", ""),
        "tp_size": getattr(layer, "tp_size", ""),
        "model_id": os.environ.get("ZCUTLASS_VLLM_MODEL_ID", ""),
        "architecture": os.environ.get("ZCUTLASS_VLLM_ARCHITECTURE", ""),
        "layer_prefix": getattr(layer, "prefix", ""),
        "layer_class": layer_class,
        "input_shape": input_shape,
        "weight_shape": weight_shape,
        "output_shape": output_shape,
        "dtype": tensor_dtype(input_tensor),
        "bias": bool(getattr(layer, "bias", None) is not None),
        "m": m,
        "n": n,
        "k": k,
        "shape_family": shape_family(m, n, k),
        "route": route,
        "fallback_reason": fallback_reason,
        "kernel_path": trace.get("kernel_path", ""),
        "kernel_name": trace.get("kernel_name", ""),
        "tile": trace.get("tile", {}),
        "selected_config": trace.get("selected_config"),
        "weight_cache": trace.get("weight_cache", ""),
        "materialize_inputs": trace.get("materialize_inputs", False),
        "latency_us": latency_us,
        "layer_hits": trace.get("hit_count", 0),
        "layer_misses": trace.get("miss_count", 0),
        "fallback_reasons": trace.get("fallback_reasons", {}),
        "install_error": install_error,
    }
