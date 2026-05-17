"""Experimental vLLM model probe that uses the zcutlass Linear adapter.

This class is intentionally tiny. It is not a production language model and is
not intended to load HuggingFace checkpoints. Its purpose is to prove the vLLM
custom model registration path without modifying the vLLM source tree.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .linear import ZCutlassVllmLinearAdapter


class ZCutlassToyForCausalLM(nn.Module):
    """Minimal vLLM-compatible model that routes one Linear through zcutlass."""

    def __init__(self, vllm_config: Any | None = None, prefix: str = "") -> None:
        super().__init__()
        model_config = getattr(vllm_config, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        self.hidden_size = int(getattr(hf_config, "hidden_size", 1024))
        self.vocab_size = int(getattr(hf_config, "vocab_size", 4096))
        self.prefix = prefix

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size)
        self.proj_weight = nn.Parameter(torch.empty(self.hidden_size, self.hidden_size))
        self.proj_bias = nn.Parameter(torch.empty(self.hidden_size))
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.adapter = ZCutlassVllmLinearAdapter(
            promoted_families=("decode", "prefill", "large"),
            materialize_inputs=True,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.proj_weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj_bias)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        del positions
        hidden_states = self.embed_input_ids(input_ids)
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        weight_t = self.proj_weight.t().contiguous()
        return self.adapter(
            hidden_states,
            weight_t,
            self.proj_bias,
            weight_is_transposed=True,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_weights(self, weights):
        del weights
        return set()
