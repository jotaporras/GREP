"""Lightweight Ψ producer for the vLLM eval/rollout path (no vllm import).

Serving a graph checkpoint through vLLM leaves the driver process needing
exactly two things from the HF side: the trained Ψ tower (``pe_model`` /
``pe_proj`` / ``pe_gain`` / ``pe_norm`` from ``gnn_weights.pt``) and the base
model's TOKEN EMBEDDING TABLE (Ψ magnitude scaling and, for
``pe_node_features='word_embeddings'``, the node features). Loading the full
31B HF model for that would double the memory bill next to the engine, so this
module rebuilds the SAME ``GraphAugmentedLLM`` the eval loader would — via the
shared ``loaders.additive_model_from_config`` — around an embeddings-only shim.

The shim is exact, not an approximation: token embeddings are a plain
``embed_tokens`` lookup (never LoRA-targeted in this repo), and the loaded
``pe_norm`` weights override the shim-derived text-RMS init. Any attempt to run
the shim as an LLM fails loud (it has no layers and no ``generate``).
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import torch
from torch import nn
from transformers import AutoConfig
from transformers.utils import cached_file

from prism.models import loaders


class _EmbeddingsOnlyLLM(nn.Module):
    """The minimal surface GraphAugmentedLLM needs from its ``llm``."""

    def __init__(self, config, embed: nn.Embedding):
        super().__init__()
        self.config = config
        self._embed = embed
        # _decoder_layers() finds this and _install_pe_injection returns early
        # on the empty list — no attention to patch, Ψ is consumed by the engine.
        self.model = SimpleNamespace(layers=[])

    def get_input_embeddings(self) -> nn.Embedding:
        return self._embed


def _load_embed_weight(base_model: str) -> torch.Tensor:
    """The text ``embed_tokens.weight`` from a (possibly sharded) HF checkpoint."""
    index_path = cached_file(
        base_model, "model.safetensors.index.json",
        _raise_exceptions_for_missing_entries=False)
    from safetensors import safe_open

    if index_path is None:
        shard = cached_file(base_model, "model.safetensors")
        with safe_open(shard, framework="pt") as f:
            keys = [k for k in f.keys() if k.endswith("embed_tokens.weight")]
            key = _pick_text_embed_key(keys, base_model)
            return f.get_tensor(key)

    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    keys = [k for k in weight_map if k.endswith("embed_tokens.weight")]
    key = _pick_text_embed_key(keys, base_model)
    shard = cached_file(base_model, weight_map[key])
    with safe_open(shard, framework="pt") as f:
        return f.get_tensor(key)


def _pick_text_embed_key(keys: list[str], base_model: str) -> str:
    """The LANGUAGE tower's embedding key (multimodal checkpoints also carry
    audio/vision embedders)."""
    text = [k for k in keys if not any(t in k for t in ("audio", "vision", "per_layer"))]
    if len(text) != 1:
        raise ValueError(
            f"could not identify the text embed_tokens key for {base_model}: "
            f"candidates={keys}")
    return text[0]


def load_psi_producer(checkpoint_dir: str, device: str = "cpu"):
    """The checkpoint's Ψ tower as an eval-mode ``GraphAugmentedLLM`` over an
    embeddings-only shim. Additive archs only — enforced by the caller's
    ``checkpoint_engine_policy``; the shared loader branch would rebuild mask
    archs wrongly, so refusal must happen before this call."""
    gnn_cfg = loaders.load_gnn_config(checkpoint_dir)
    base_model = gnn_cfg["base_model"]
    config = AutoConfig.from_pretrained(base_model)
    embed_weight = _load_embed_weight(base_model)
    embed = nn.Embedding.from_pretrained(embed_weight.float(), freeze=True)
    shim = _EmbeddingsOnlyLLM(config, embed)
    producer = loaders.additive_model_from_config(shim, gnn_cfg, checkpoint_dir)
    return producer.to(device).eval()
