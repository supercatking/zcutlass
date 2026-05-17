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

    from zcutlass_torch import ZCutlassGemmOverlay, extension_available

    if not torch.cuda.is_available():
        print("SKIP: CUDA is not available through PyTorch")
        return 1 if args.require_torch else 0

    if not extension_available():
        print("SKIP: zcutlass_torch extension is not installed")
        return 1 if args.require_extension else 0

    overlay = ZCutlassGemmOverlay()
    a = torch.randn((8, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 128), device="cuda", dtype=torch.float16)
    bias = torch.randn((128,), device="cuda", dtype=torch.float16)
    actual = overlay.gemm(a, b, bias=bias)
    expected = torch.matmul(a, b) + bias
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    print(f"PASS: hits={overlay.stats.hits} misses={overlay.stats.misses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
