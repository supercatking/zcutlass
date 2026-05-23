"""vLLM LinearMethod wrapper for explicit zcutlass routing.

This module is intentionally opt-in. It does not patch vLLM globally; callers
install the wrapper on selected Linear layers after deciding that a callsite is
eligible for zcutlass experiments.
"""

from __future__ import annotations

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
