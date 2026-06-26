"""Checkpoint discovery + loading shared by every eval path.

A trained PRISM run dir is one of two kinds, told apart by which marker file the
trainer wrote:

* **graph-augmented** — has ``gnn_config.json`` (+ ``gnn_weights.pt`` / the LoRA
  adapter). Loaded via :func:`prism.models.loaders.graph_augmented_llm_from_pretrained`.
* **plain LLM** — has only the PEFT ``adapter_config.json`` (+ ``train_config.json``).
  Loaded via :func:`prism.models.loaders.from_pretrained`.

This module centralises the three operations every consumer needs — kind
detection, train-time ``text_edge_list`` policy recovery, and the actual model
load — so the post-hoc driver (``scalability_evaluation``) and the in-process
post-train eval (``train_v3`` / ``train_v2``) share one implementation.
"""
from __future__ import annotations

import json
import os

from prism.models import loaders


def is_gnn_checkpoint(path: str) -> bool:
    """True iff ``path`` is a graph-augmented checkpoint (has ``gnn_config.json``)."""
    return os.path.exists(os.path.join(path, "gnn_config.json"))


def resolve_text_edge_list(checkpoint: str, is_gnn: bool, cli_override: str | None) -> str:
    """Recover the train-time ``text_edge_list`` policy so eval matches training,
    i.e. if the checkpoint was trained with edge lists, eval should also use edge lists.

    TO DO: it seems the only reason this exists is because configs divergesd
    for graph-based and LLM-based checkpoints. We should consolidate the configs.
    """
    if cli_override is not None:
        return cli_override
    if is_gnn:
        gnn_cfg_path = os.path.join(checkpoint, "gnn_config.json")
        with open(gnn_cfg_path) as f:
            gnn_cfg = json.load(f)
        text_edge_list = gnn_cfg.get("text_edge_list")
        if text_edge_list is None:
            raise KeyError(
                f"{gnn_cfg_path} does not record 'text_edge_list'; pass --text-edge-list "
                f"present|none explicitly to evaluate this checkpoint."
            )
        return text_edge_list
    train_cfg_path = os.path.join(checkpoint, "train_config.json")
    if not os.path.exists(train_cfg_path):
        raise FileNotFoundError(
            f"{checkpoint} has no train_config.json recording the train-time "
            f"text_edge_list policy. Cannot infer whether the LLM-facing scene-graph "
            f"block was trained with edge bullets; pass --text-edge-list present|none "
            f"explicitly to evaluate this checkpoint."
        )
    with open(train_cfg_path) as f:
        train_cfg = json.load(f)
    text_edge_list = train_cfg.get("text_edge_list")
    if text_edge_list is None:
        raise KeyError(
            f"{train_cfg_path} does not record 'text_edge_list'; pass --text-edge-list "
            f"present|none explicitly to evaluate this checkpoint."
        )
    return text_edge_list


def load_checkpoint(checkpoint: str, four_bit: bool, device: int):
    """Load a trained checkpoint for eval. Returns ``(model, tokenizer, is_gnn)``.

    Dispatches on :func:`is_gnn_checkpoint`. Raises ``FileNotFoundError`` if the
    dir has neither ``gnn_config.json`` nor ``adapter_config.json`` (not a
    recognisable checkpoint).
    """
    if is_gnn_checkpoint(checkpoint):
        model, tok = loaders.graph_augmented_llm_from_pretrained(
            checkpoint, load_in_4bit=four_bit, device=device,
        )
        return model, tok, True
    if not os.path.exists(os.path.join(checkpoint, "adapter_config.json")):
        raise FileNotFoundError(
            f"{checkpoint} has neither gnn_config.json nor adapter_config.json — "
            "not a recognisable checkpoint dir."
        )
    model, tok = loaders.from_pretrained(checkpoint, load_in_4bit=four_bit, device=device)
    return model, tok, False
