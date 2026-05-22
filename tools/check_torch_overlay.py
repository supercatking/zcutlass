#!/usr/bin/env python3
"""Smoke-check the zcutlass PyTorch overlay package.

This check is intentionally skip-friendly because the base WSL environment may
not have PyTorch installed. When PyTorch and the extension are available, it
runs a tiny CUDA GEMM through the overlay and compares it with torch.matmul.
"""

from __future__ import annotations

import argparse
import pathlib
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-torch", action="store_true")
    parser.add_argument("--require-extension", action="store_true")
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "python"))

    try:
        import torch
    except Exception as exc:
        print(f"SKIP: PyTorch is not installed: {exc}")
        return 1 if args.require_torch else 0

    from zcutlass_torch import ZCutlassGemmOverlay, extension_available, selected_gemm_config

    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available through PyTorch")
        return 1 if args.require_torch else 0

    if not extension_available():
        print("SKIP: zcutlass_torch extension is not installed")
        return 1 if args.require_extension else 0

    overlay = ZCutlassGemmOverlay(force_zcutlass=True)
    a = torch.randn((8, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    bias = torch.randn((128,), device="cuda", dtype=torch.float16)
    actual = overlay.gemm(a, b, bias=bias)
    expected = torch.matmul(a, b) + bias
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    config = selected_gemm_config(a, b, bias=bias)
    assert config is not None
    assert overlay.last_kernel_name == config["kernel_name"]
    assert overlay.last_tile == config["tile"]

    policy_overlay = ZCutlassGemmOverlay()
    fallback = policy_overlay.gemm(a, b, bias=bias)
    torch.testing.assert_close(fallback, expected, rtol=1e-2, atol=1e-2)
    assert policy_overlay.stats.fallback_reasons.get("shape_not_target_bucket") == 1

    print(
        "PASS: "
        f"forced_hits={overlay.stats.hits} "
        f"kernel={overlay.last_kernel_name} "
        f"tile={overlay.last_tile} "
        f"policy_misses={policy_overlay.stats.misses} "
        f"policy_reasons={policy_overlay.stats.fallback_reasons}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
