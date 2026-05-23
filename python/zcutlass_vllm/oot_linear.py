"""Out-of-tree vLLM Linear layer registration for zcutlass overlay proof."""

from __future__ import annotations

import os
import time
from typing import Any

from .linear_method import install_zcutlass_unquantized_linear_method
from .route_logger import build_route_row, csv_env, env_enabled, make_route_logger


DEFAULT_LAYER_FILTER = ("qkv_proj", "gate_up_proj", "down_proj", "o_proj")
TARGET_LAYER_CLASS_NAMES = (
    "QKVParallelLinear",
    "MergedColumnParallelLinear",
    "ColumnParallelLinear",
    "RowParallelLinear",
)

_REGISTERED = False


def _layer_filter() -> tuple[str, ...]:
    return csv_env("ZCUTLASS_VLLM_LAYER_FILTER", DEFAULT_LAYER_FILTER)


def _allow_families() -> tuple[str, ...]:
    return csv_env("ZCUTLASS_VLLM_ALLOW_FAMILIES", ())


def _matches_layer_filter(prefix: str) -> bool:
    filters = _layer_filter()
    if not filters:
        return False
    if "*" in filters or "all" in filters:
        return True
    return any(prefix.endswith(token) or token in prefix for token in filters)


def _skip_install_for_profit_policy(layer: Any) -> str | None:
    policy = os.environ.get("ZCUTLASS_VLLM_PROFIT_POLICY", "off").strip().lower()
    if policy != "measured":
        return None
    prefix = str(getattr(layer, "prefix", "")).lower()
    if "qkv_proj" not in prefix:
        return "profit_policy_layer_not_promoted"
    weight = getattr(layer, "weight", None)
    dtype = str(getattr(weight, "dtype", "")).lower()
    if dtype not in ("torch.float16", "float16", "f16", "half", "torch.half"):
        return "profit_policy_dtype_not_promoted"
    return None


class ZCutlassOotLinearMixin:
    """Mixin installed on selected vLLM Linear classes."""

    def _zcutlass_post_init(self) -> None:
        self._zcutlass_route_logger = make_route_logger()
        self._zcutlass_install_error = None
        self._zcutlass_target_layer = False
        self._zcutlass_overlay_method = None
        if not env_enabled("ZCUTLASS_VLLM_ENABLE"):
            return
        prefix = str(getattr(self, "prefix", ""))
        if not _matches_layer_filter(prefix):
            return
        skip_reason = _skip_install_for_profit_policy(self)
        if skip_reason is not None:
            self._zcutlass_install_error = skip_reason
            return
        self._zcutlass_target_layer = True
        try:
            self._zcutlass_overlay_method = install_zcutlass_unquantized_linear_method(
                self,
                promoted_families=_allow_families(),
                materialize_inputs=True,
                cache_transposed_weight=True,
            )
        except Exception as exc:
            self._zcutlass_install_error = str(exc)

    def _zcutlass_log_forward(self, input_tensor: Any, output: Any, latency_us: float) -> None:
        logger = getattr(self, "_zcutlass_route_logger", None)
        if logger is None or not getattr(self, "_zcutlass_target_layer", False):
            return
        method = getattr(self, "quant_method", None)
        trace = getattr(method, "last_trace", None)
        row = build_route_row(
            layer=self,
            layer_class=self.__class__.__name__,
            input_tensor=input_tensor,
            output=output,
            latency_us=latency_us,
            trace=trace,
            install_error=getattr(self, "_zcutlass_install_error", None),
        )
        logger.write(row)

    def forward(self, input_):
        if (
            getattr(self, "_zcutlass_route_logger", None) is None
            or not getattr(self, "_zcutlass_target_layer", False)
        ):
            return super().forward(input_)
        start_ns = time.perf_counter_ns()
        output = super().forward(input_)
        latency_us = (time.perf_counter_ns() - start_ns) / 1000.0
        self._zcutlass_log_forward(input_, output, latency_us)
        return output


def register_oot_linear_layers() -> bool:
    """Register zcutlass OOT classes for vLLM Linear layers."""

    global _REGISTERED
    if _REGISTERED:
        return True
    if not env_enabled("ZCUTLASS_VLLM_ENABLE"):
        return False

    from vllm.model_executor.custom_op import PluggableLayer, op_registry_oot
    from vllm.model_executor.layers.linear import (
        ColumnParallelLinear,
        MergedColumnParallelLinear,
        QKVParallelLinear,
        RowParallelLinear,
    )

    if any(
        name in op_registry_oot
        and not str(getattr(op_registry_oot[name], "__module__", "")).startswith("zcutlass_vllm")
        for name in TARGET_LAYER_CLASS_NAMES
    ):
        return False

    if "ColumnParallelLinear" not in op_registry_oot:
        @PluggableLayer.register_oot(name="ColumnParallelLinear")
        class ZCutlassColumnParallelLinear(ZCutlassOotLinearMixin, ColumnParallelLinear):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._zcutlass_post_init()

    if "MergedColumnParallelLinear" not in op_registry_oot:
        @PluggableLayer.register_oot(name="MergedColumnParallelLinear")
        class ZCutlassMergedColumnParallelLinear(ZCutlassOotLinearMixin, MergedColumnParallelLinear):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._zcutlass_post_init()

    if "QKVParallelLinear" not in op_registry_oot:
        @PluggableLayer.register_oot(name="QKVParallelLinear")
        class ZCutlassQKVParallelLinear(ZCutlassOotLinearMixin, QKVParallelLinear):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._zcutlass_post_init()

    if "RowParallelLinear" not in op_registry_oot:
        @PluggableLayer.register_oot(name="RowParallelLinear")
        class ZCutlassRowParallelLinear(ZCutlassOotLinearMixin, RowParallelLinear):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self._zcutlass_post_init()

    _REGISTERED = all(name in op_registry_oot for name in TARGET_LAYER_CLASS_NAMES)
    return _REGISTERED
