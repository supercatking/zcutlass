from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


def _torch():
    import torch

    return torch


def _load_extension() -> bool:
    try:
        import zcutlass_torch._C  # noqa: F401
    except Exception:
        return False
    return True


def extension_available() -> bool:
    return _load_extension()


@dataclass
class OverlayStats:
    hits: int = 0
    misses: int = 0
    fallback_reasons: Dict[str, int] = field(default_factory=dict)

    def record_hit(self) -> None:
        self.hits += 1

    def record_miss(self, reason: str) -> None:
        self.misses += 1
        self.fallback_reasons[reason] = self.fallback_reasons.get(reason, 0) + 1

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0


@dataclass(frozen=True)
class RoutingPolicy:
    """Shape policy for v1.5 explicit overlay experiments.

    The default policy only allows LLM-shaped buckets into zcutlass. This keeps
    the product proof honest: non-target or currently unvalidated shapes still
    run correctly through PyTorch fallback and produce fallback telemetry.
    """

    enabled: bool = True
    min_n: int = 1024
    min_k: int = 1024
    decode_max_m: int = 16
    prefill_min_m: int = 32
    prefill_max_m: int = 256
    large_min_m: int = 512
    promoted_families: tuple[str, ...] = ()

    def family(self, m: int, n: int, k: int) -> str:
        if m <= self.decode_max_m and n >= self.min_n and k >= self.min_k:
            return "decode"
        if self.prefill_min_m <= m <= self.prefill_max_m and n >= self.min_n and k >= self.min_k:
            return "prefill"
        if m >= self.large_min_m and n >= self.min_n and k >= self.min_k:
            return "large"
        return "fallback"

    def reject_reason(self, m: int, n: int, k: int) -> Optional[str]:
        if not self.enabled:
            return None
        family = self.family(m, n, k)
        if family == "fallback":
            return "shape_not_target_bucket"
        if family not in self.promoted_families:
            return "family_not_promoted"
        return None


class ZCutlassGemmOverlay:
    """Explicit PyTorch overlay for zcutlass GEMM proof-of-value experiments.

    This class never globally hooks PyTorch or cuBLAS. Callers opt in at a known
    Linear/GEMM callsite. Unsupported inputs fall back to the stock PyTorch path
    and record a reason for serving-level reports.
    """

    def __init__(
        self,
        enable_zcutlass: bool = True,
        *,
        routing_policy: Optional[RoutingPolicy] = None,
        force_zcutlass: bool = False,
    ) -> None:
        self.enable_zcutlass = enable_zcutlass
        self.routing_policy = routing_policy or RoutingPolicy()
        self.force_zcutlass = force_zcutlass
        self.stats = OverlayStats()
        self._loaded = False
        self.last_family = "unknown"
        self.last_kernel_path = "unknown"
        self.last_fallback_reason: Optional[str] = None

    def _ensure_loaded(self) -> bool:
        if not self.enable_zcutlass:
            return False
        if self._loaded:
            return True
        self._loaded = _load_extension()
        return self._loaded

    def _fallback(self, reason: str, a, b, c=None, bias=None, alpha: float = 1.0, beta: float = 0.0):
        torch = _torch()
        self.stats.record_miss(reason)
        self.last_kernel_path = "pytorch_fallback"
        self.last_fallback_reason = reason
        out = torch.matmul(a, b)
        if alpha != 1.0:
            out = out * alpha
        if c is not None and beta != 0.0:
            out = out + c * beta
        if bias is not None:
            out = out + bias
        return out

    def _reject_reason(self, a, b, c=None, bias=None, beta: float = 0.0) -> Optional[str]:
        torch = _torch()
        self.last_family = "unknown"
        self.last_kernel_path = "unknown"
        self.last_fallback_reason = None
        if not self.enable_zcutlass:
            return "zcutlass_disabled"
        if not self._ensure_loaded():
            return "extension_unavailable"
        if c is None and beta != 0.0:
            return "beta_without_c"
        if not a.is_cuda or not b.is_cuda:
            return "cpu_tensor"
        if a.dim() != 2 or b.dim() != 2:
            return "rank_not_2"
        if a.shape[1] != b.shape[0]:
            return "inner_dimension_mismatch"
        self.last_family = self.routing_policy.family(int(a.shape[0]), int(b.shape[1]), int(a.shape[1]))
        if not a.is_contiguous() or not b.is_contiguous():
            return "non_contiguous_ab"
        if a.dtype != b.dtype:
            return "mixed_dtype"
        if a.dtype not in (torch.float16, torch.bfloat16):
            return "unsupported_dtype"
        if c is not None:
            if not c.is_cuda:
                return "cpu_c_tensor"
            if c.shape != (a.shape[0], b.shape[1]):
                return "c_shape_mismatch"
            if c.dtype != a.dtype:
                return "c_dtype_mismatch"
            if not c.is_contiguous():
                return "non_contiguous_c"
        if bias is not None:
            if not bias.is_cuda:
                return "cpu_bias_tensor"
            if bias.dim() != 1 or bias.shape[0] != b.shape[1]:
                return "bias_shape_mismatch"
            if bias.dtype != a.dtype:
                return "bias_dtype_mismatch"
            if not bias.is_contiguous():
                return "non_contiguous_bias"
        if not self.force_zcutlass:
            policy_reason = self.routing_policy.reject_reason(int(a.shape[0]), int(b.shape[1]), int(a.shape[1]))
            if policy_reason is not None:
                return policy_reason
        return None

    def gemm(self, a, b, c=None, bias=None, alpha: float = 1.0, beta: float = 0.0):
        reason = self._reject_reason(a, b, c, bias, beta)
        if reason is not None:
            return self._fallback(reason, a, b, c, bias, alpha, beta)
        torch = _torch()
        self.stats.record_hit()
        self.last_kernel_path = "zcutlass"
        self.last_fallback_reason = None
        return torch.ops.zcutlass_torch.gemm(a, b, c, bias, float(alpha), float(beta))

    def linear(self, x, weight, bias=None, *, weight_is_transposed: bool = False):
        """Run a Linear callsite with opt-in zcutlass routing.

        PyTorch stores Linear weight as [out_features, in_features]. zcutlass v1
        only supports row-major B as [K, N], so the fast path requires callers to
        pass pre-transposed contiguous weight and set weight_is_transposed=True.
        The default path preserves stock PyTorch semantics.
        """

        torch = _torch()
        if not weight_is_transposed:
            self.stats.record_miss("weight_not_pretransposed")
            return torch.nn.functional.linear(x, weight, bias)
        return self.gemm(x, weight, None, bias, 1.0, 0.0)


_DEFAULT_OVERLAY = ZCutlassGemmOverlay()


def gemm(a, b, c=None, bias=None, alpha: float = 1.0, beta: float = 0.0):
    return _DEFAULT_OVERLAY.gemm(a, b, c, bias, alpha, beta)


def linear(x, weight, bias=None, *, weight_is_transposed: bool = False):
    return _DEFAULT_OVERLAY.linear(x, weight, bias, weight_is_transposed=weight_is_transposed)
