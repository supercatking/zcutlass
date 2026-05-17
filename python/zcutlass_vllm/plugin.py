"""vLLM general plugin entry point for zcutlass.

The plugin is intentionally small and side-effect-light. Loading it proves that a
vLLM process can discover zcutlass and opt in to the adapter package. It does
not globally patch vLLM, PyTorch, or cuBLAS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class PluginState:
    registered: bool = False
    extension_available: Optional[bool] = None
    message: str = ""


_STATE = PluginState()


def register() -> None:
    """Register the zcutlass overlay plugin with a vLLM process.

    vLLM general plugins may be loaded more than once in test and worker
    processes, so this function is idempotent.
    """

    if _STATE.registered:
        return
    try:
        from zcutlass_torch import extension_available

        _STATE.extension_available = extension_available()
        _STATE.message = "zcutlass vLLM overlay plugin loaded"
        os.environ["ZCUTLASS_VLLM_PLUGIN_LOADED"] = "1"
        os.environ["ZCUTLASS_TORCH_EXTENSION_AVAILABLE"] = "1" if _STATE.extension_available else "0"
    except Exception as exc:  # pragma: no cover - defensive for vLLM import environments.
        _STATE.extension_available = False
        _STATE.message = f"zcutlass vLLM overlay plugin failed to inspect extension: {exc}"
        os.environ["ZCUTLASS_VLLM_PLUGIN_LOADED"] = "1"
        os.environ["ZCUTLASS_TORCH_EXTENSION_AVAILABLE"] = "0"
    _STATE.registered = True


def is_registered() -> bool:
    return _STATE.registered


def state() -> PluginState:
    return _STATE
