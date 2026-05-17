"""vLLM-facing Linear adapter built on the zcutlass PyTorch overlay.

This adapter is deliberately explicit. It can be used by a vLLM custom model or
worker experiment to route selected Linear callsites through zcutlass while
keeping unpromoted shapes on the stock PyTorch path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from zcutlass_torch import RoutingPolicy, ZCutlassGemmOverlay


@dataclass
class VllmLinearStats:
    calls: int = 0
    hits: int = 0
    misses: int = 0
    fallback_reasons: Dict[str, int] = field(default_factory=dict)

    def update_from_overlay(self, overlay: ZCutlassGemmOverlay) -> None:
        self.calls += 1
        self.hits += overlay.stats.hits
        self.misses += overlay.stats.misses
        for reason, count in overlay.stats.fallback_reasons.items():
            self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + count

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class ZCutlassVllmLinearAdapter:
    """Explicit Linear adapter for vLLM integration experiments."""

    def __init__(
        self,
        *,
        promoted_families: tuple[str, ...] = (),
        force_zcutlass: bool = False,
        materialize_inputs: bool = False,
    ) -> None:
        self.promoted_families = promoted_families
        self.force_zcutlass = force_zcutlass
        self.materialize_inputs = materialize_inputs
        self.stats = VllmLinearStats()
        self.last_trace: Optional[dict[str, Any]] = None

    def __call__(self, x, weight, bias=None, *, weight_is_transposed: bool = False):
        """Run one Linear callsite.

        Args:
            x: `[M, K]` activation tensor.
            weight: stock `[N, K]` weight unless `weight_is_transposed=True`,
                in which case it must be contiguous `[K, N]`.
            bias: optional `[N]` bias.
            weight_is_transposed: set true for the zcutlass fast path.
        """

        if self.materialize_inputs and hasattr(x, "is_contiguous") and not x.is_contiguous():
            x = x.contiguous()

        overlay = ZCutlassGemmOverlay(
            routing_policy=RoutingPolicy(promoted_families=self.promoted_families),
            force_zcutlass=self.force_zcutlass,
        )
        out = overlay.linear(x, weight, bias, weight_is_transposed=weight_is_transposed)
        self.stats.update_from_overlay(overlay)
        self.last_trace = {
            "family": overlay.last_family,
            "kernel_path": overlay.last_kernel_path,
            "fallback_reason": overlay.last_fallback_reason,
            "hit_count": overlay.stats.hits,
            "miss_count": overlay.stats.misses,
            "fallback_reasons": dict(overlay.stats.fallback_reasons),
            "promoted_families": self.promoted_families,
            "force_zcutlass": self.force_zcutlass,
            "materialize_inputs": self.materialize_inputs,
        }
        return out


def make_linear_adapter(**kwargs: Any) -> ZCutlassVllmLinearAdapter:
    return ZCutlassVllmLinearAdapter(**kwargs)
