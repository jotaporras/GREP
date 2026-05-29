"""Llama 3.1 8B with rotary positional encodings (RoPE) disabled.

In GREP-PRISM the positional signal is meant to come from the injected graph
positional encodings (GREPs), not from the LLM's native RoPE.  This module
provides a Llama 3.1 8B variant whose rotary embedding is a no-op so the
transformer's only source of position information is the added GREPs.
"""

import torch
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
)

LLAMA_3_1_8B = "meta-llama/Llama-3.1-8B"


class _IdentityRotaryEmbedding(LlamaRotaryEmbedding):
    """RoPE module that returns the identity rotation.

    ``apply_rotary_pos_emb`` computes ``q * cos + rotate_half(q) * sin``, so
    returning ``cos = 1`` and ``sin = 0`` leaves the query/key tensors
    unchanged — RoPE is effectively switched off.  We call the parent forward
    only to inherit the correct shape, dtype and device for the tensors.
    """

    @torch.no_grad()
    def forward(self, x, position_ids):
        cos, sin = super().forward(x, position_ids)
        return torch.ones_like(cos), torch.zeros_like(sin)


class LlamaNoRoPEForCausalLM(LlamaForCausalLM):
    """Llama 3.1 8B with RoPE disabled across every attention layer.

    ``LlamaModel`` builds a single ``rotary_emb`` and feeds its ``(cos, sin)``
    output to all decoder layers, so swapping that one module suffices to
    disable RoPE model-wide.
    """

    def __init__(self, config):
        super().__init__(config)
        self.model.rotary_emb = _IdentityRotaryEmbedding(config=config)


def disable_rope(model: LlamaForCausalLM) -> LlamaNoRoPEForCausalLM:
    """Disable RoPE on an already-loaded Llama model, in place.

    Re-typing the live model (rather than re-instantiating via
    ``from_pretrained``) keeps the trained weights and their current device
    placement untouched.  The identity rotary embedding is created on the same
    device as the existing one so the swap is device-consistent.
    """
    device = model.model.rotary_emb.inv_freq.device
    model.model.rotary_emb = _IdentityRotaryEmbedding(config=model.config).to(device)
    model.__class__ = LlamaNoRoPEForCausalLM
    return model
