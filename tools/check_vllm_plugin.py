#!/usr/bin/env python3
"""Check that vLLM can discover the zcutlass general plugin.

The check is skip-friendly when vLLM is not installed. It still validates the
Python entry point and adapter package in environments that only have PyTorch.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import pathlib
import sys


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def find_entry_point() -> bool:
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):
        selected = eps.select(group="vllm.general_plugins")
    else:
        selected = eps.get("vllm.general_plugins", [])
    return any(ep.name == "zcutlass_overlay" for ep in selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-vllm", action="store_true")
    parser.add_argument("--require-entry-point", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(repo_root() / "python"))

    try:
        import zcutlass_vllm
    except Exception as exc:
        raise SystemExit(f"FAIL: zcutlass_vllm import failed: {exc}") from exc

    zcutlass_vllm.register()
    if not zcutlass_vllm.is_registered():
        raise SystemExit("FAIL: zcutlass_vllm plugin did not register")

    entry_point_ok = find_entry_point()
    if args.require_entry_point and not entry_point_ok:
        raise SystemExit("FAIL: zcutlass_overlay entry point is not installed")

    try:
        import vllm  # noqa: F401

        vllm_status = "installed"
    except Exception as exc:
        if args.require_vllm:
            raise SystemExit(f"FAIL: vLLM is required but unavailable: {exc}") from exc
        vllm_status = f"not_installed ({exc})"

    print(
        "PASS: zcutlass_vllm import/register ok; "
        f"entry_point_installed={entry_point_ok}; vllm={vllm_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
