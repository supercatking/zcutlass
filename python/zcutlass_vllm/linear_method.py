"""vLLM LinearMethod wrapper for explicit zcutlass routing.

This module is intentionally opt-in. It does not patch vLLM globally; callers
install the wrapper on selected Linear layers after deciding that a callsite is
eligible for zcutlass experiments.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from .linear import ZCutlassVllmLinearAdapter

try:  # vLLM is optional outside serving integration environments.
    from vllm.model_executor.layers.linear import LinearMethodBase, UnquantizedLinearMethod
except Exception:  # pragma: no cover - exercised in non-vLLM environments.
    LinearMethodBase = object  # type: ignore[assignment,misc]
    UnquantizedLinearMethod = None  # type: ignore[assignment]


class ZCutlassUnquantizedLinearMethod(LinearMethodBase):  # type: ignore[misc,valid-type]
    """Wrap vLLM's unquantized Linear method with zcutlass-first routing."""

    def __init__(
        self,
        delegate: Optional[Any] = None,
        *,
        promoted_families: tuple[str, ...] = (),
        force_zcutlass: bool = False,
        materialize_inputs: bool = True,
        cache_transposed_weight: bool = True,
    ) -> None:
        if UnquantizedLinearMethod is None:
            raise ImportError("vLLM is required to construct ZCutlassUnquantizedLinearMethod")
        self.delegate = delegate if delegate is not None else UnquantizedLinearMethod()
        self.adapter = ZCutlassVllmLinearAdapter(
            promoted_families=promoted_families,
            force_zcutlass=force_zcutlass,
            materialize_inputs=materialize_inputs,
        )
        self.cache_transposed_weight = cache_transposed_weight
        self.last_weight_cache = "unknown"

    @property
    def stats(self):
        return self.adapter.stats

    @property
    def last_trace(self):
        return self.adapter.last_trace

    def create_weights(self, *args: Any, **kwargs: Any):
        return self.delegate.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer) -> None:
        if hasattr(self.delegate, "process_weights_after_loading"):
            self.delegate.process_weights_after_loading(layer)
        clear_zcutlass_weight_cache(layer)

    def _delegate_without_transpose_reason(self, layer, x) -> tuple[str, str] | None:
        if self.adapter.force_zcutlass:
            return None
        x_shape = getattr(x, "shape", None)
        weight = getattr(layer, "weight", None)
        weight_shape = getattr(weight, "shape", None)
        if x_shape is None or weight_shape is None or len(x_shape) != 2 or len(weight_shape) != 2:
            return ("rank_not_2", "unknown")
        m = int(x_shape[0])
        k = int(x_shape[1])
        n = int(weight_shape[0])
        weight_k = int(weight_shape[1])
        if k != weight_k:
            return ("inner_dimension_mismatch", "unknown")
        family = self.adapter.routing_policy.family(m, n, k)
        if family == "fallback":
            return ("shape_not_target_bucket", family)
        if family not in self.adapter.promoted_families:
            return ("family_not_promoted", family)
        profit_policy_reason = self._profit_policy_reject_reason(layer, x, m, n, k, family)
        if profit_policy_reason is not None:
            return (profit_policy_reason, family)
        return None

    def _profit_policy_reject_reason(self, layer, x, m: int, n: int, k: int, family: str) -> str | None:
        """Reject vLLM routes outside the locally measured profitable subset."""

        policy = os.environ.get("ZCUTLASS_VLLM_PROFIT_POLICY", "off").strip().lower()
        if policy in ("", "0", "off", "none", "false", "disabled"):
            return None
        if policy in ("all", "experimental"):
            return None
        if policy != "measured":
            return "profit_policy_unknown"

        dtype = str(getattr(x, "dtype", "")).lower()
        if dtype not in ("torch.float16", "float16", "f16", "half", "torch.half"):
            return "profit_policy_dtype_not_promoted"
        if family != "prefill":
            return "profit_policy_family_not_promoted"
        if not bool(getattr(layer, "bias", None) is not None):
            return "profit_policy_bias_not_promoted"

        prefix = str(getattr(layer, "prefix", "")).lower()
        is_qkv = "qkv_proj" in prefix
        if not is_qkv:
            return "profit_policy_layer_not_promoted"

        if m == 256 and n == 2048 and k == 1536:
            return None
        return "profit_policy_shape_not_promoted"

    def _transposed_weight(self, layer):
        weight = layer.weight
        if not self.cache_transposed_weight:
            self.last_weight_cache = "disabled"
            return weight.t().contiguous()

        signature = (
            int(weight.data_ptr()) if hasattr(weight, "data_ptr") else 0,
            tuple(weight.shape),
            str(weight.dtype),
            str(weight.device),
        )
        cached = getattr(layer, "_zcutlass_weight_t_cache", None)
        if cached is not None:
            cached_signature, cached_weight = cached
            if cached_signature == signature:
                self.last_weight_cache = "hit"
                return cached_weight
        weight_t = weight.t().contiguous()
        setattr(layer, "_zcutlass_weight_t_cache", (signature, weight_t))
        self.last_weight_cache = "miss"
        return weight_t

    def apply(self, layer, x, bias=None):
        fast_delegate = self._delegate_without_transpose_reason(layer, x)
        if fast_delegate is not None:
            reason, family = fast_delegate
            self.last_weight_cache = "skipped"
            self.adapter._record_fallback_trace(
                reason=reason,
                family=family,
                fallback_path_name="vllm_unquantized_fallback",
            )
            if self.adapter.last_trace is not None:
                self.adapter.last_trace["weight_cache"] = self.last_weight_cache
            return self.delegate.apply(layer, x, bias)

        weight_t = self._transposed_weight(layer)
        output = self.adapter.run(
            x,
            weight_t,
            bias,
            weight_is_transposed=True,
            fallback_fn=lambda: self.delegate.apply(layer, x, bias),
            fallback_path_name="vllm_unquantized_fallback",
        )
        if self.adapter.last_trace is not None:
            self.adapter.last_trace["weight_cache"] = self.last_weight_cache
        return output


def clear_zcutlass_weight_cache(layer) -> None:
    if hasattr(layer, "_zcutlass_weight_t_cache"):
        delattr(layer, "_zcutlass_weight_t_cache")


def install_zcutlass_unquantized_linear_method(
    layer,
    *,
    promoted_families: tuple[str, ...] = (),
    force_zcutlass: bool = False,
    materialize_inputs: bool = True,
    cache_transposed_weight: bool = True,
) -> ZCutlassUnquantizedLinearMethod:
    """Replace one vLLM Linear layer's unquantized method with zcutlass wrapper."""

    if UnquantizedLinearMethod is None:
        raise ImportError("vLLM is required to install ZCutlassUnquantizedLinearMethod")

    current = getattr(layer, "quant_method", None)
    if isinstance(current, ZCutlassUnquantizedLinearMethod):
        return current
    if current is not None and not isinstance(current, UnquantizedLinearMethod):
        raise TypeError(
            "zcutlass can only wrap vLLM UnquantizedLinearMethod layers in v1.5"
        )

    method = ZCutlassUnquantizedLinearMethod(
        current,
        promoted_families=promoted_families,
        force_zcutlass=force_zcutlass,
        materialize_inputs=materialize_inputs,
        cache_transposed_weight=cache_transposed_weight,
    )
    layer.quant_method = method
    return method
