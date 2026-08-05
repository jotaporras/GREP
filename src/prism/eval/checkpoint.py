"""Checkpoint discovery + loading shared by every eval path.

Every trained run dir carries a single ``train_config.json`` with the shared run
metadata (``architecture``, ``base_model``, ``text_edge_list``,
``injection_scope``) at the top level; **graph-augmented** checkpoints
additionally nest their architecture hyperparameters under a ``"gnn"`` key (and
ship ``gnn_weights.pt`` / the LoRA adapter), while **plain-LLM** checkpoints have
no ``"gnn"`` key (only the PEFT ``adapter_config.json``).

Legacy checkpoints (pre-cleanup) used a separate flat ``gnn_config.json`` for
graph runs instead; every reader here falls back to it, so old run dirs stay
loadable.

This module centralises the three operations every consumer needs — kind
detection, train-time ``text_edge_list`` policy recovery, and the actual model
load — so the post-hoc driver (``scalability_evaluation``), the in-process
post-train eval (``train_v3``), and the diagnostics share one implementation.
"""
from __future__ import annotations

import json
import os

from prism.models import loaders

# Re-exported so eval-side consumers don't need a loaders import for config reads.
load_gnn_config = loaders.load_gnn_config


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def is_gnn_checkpoint(path: str) -> bool:
    """True iff ``path`` is a graph-augmented checkpoint.

    Current format: ``train_config.json`` carries a ``"gnn"`` key. Legacy format:
    a standalone ``gnn_config.json`` marker file.
    """
    tc = _read_json(os.path.join(path, "train_config.json"))
    if tc is not None and "gnn" in tc:
        return True
    return os.path.exists(os.path.join(path, "gnn_config.json"))


def resolve_text_edge_list(checkpoint: str, is_gnn: bool, cli_override: str | None) -> str:
    """Recover the train-time ``text_edge_list`` policy so eval matches training,
    i.e. if the checkpoint was trained with edge lists, eval should also use edge lists.

    Reads ``train_config.json`` (both checkpoint kinds record the key at the top
    level); legacy graph checkpoints fall back to ``gnn_config.json``. Raises
    loudly when the policy cannot be recovered — a silent default here is a
    train/eval mismatch.
    """
    if cli_override is not None:
        return cli_override

    tc = _read_json(os.path.join(checkpoint, "train_config.json"))
    if tc is not None and tc.get("text_edge_list") is not None:
        return tc["text_edge_list"]

    if is_gnn:
        legacy = _read_json(os.path.join(checkpoint, "gnn_config.json"))
        if legacy is not None and legacy.get("text_edge_list") is not None:
            return legacy["text_edge_list"]
        raise KeyError(
            f"{checkpoint} does not record 'text_edge_list' (train_config.json / "
            f"gnn_config.json); pass --text-edge-list present|none explicitly to "
            f"evaluate this checkpoint."
        )
    if tc is None:
        raise FileNotFoundError(
            f"{checkpoint} has no train_config.json recording the train-time "
            f"text_edge_list policy. Cannot infer whether the LLM-facing scene-graph "
            f"block was trained with edge bullets; pass --text-edge-list present|none "
            f"explicitly to evaluate this checkpoint."
        )
    raise KeyError(
        f"{os.path.join(checkpoint, 'train_config.json')} does not record "
        f"'text_edge_list'; pass --text-edge-list present|none explicitly to "
        f"evaluate this checkpoint."
    )


def resolve_prompt_policy(checkpoint: str) -> tuple:
    """Recover the train-time ``(data.spine_tools, data.icl_examples)`` prompt policy.

    Read from ``train_config.json``, top level first and then the nested ``"gnn"`` block:
    graph checkpoints written before these keys joined ``GraphSFTTrainer._RUN_META_KEYS``
    put them under ``"gnn"``, and reading only the top level made every such run resolve
    to the "predates the knob" fallback — i.e. an ICL-trained checkpoint silently
    re-evaluated zero-shot. Missing from BOTH means the checkpoint really does predate the
    knobs, and every such run was trained tool-free and zero-shot — so ``("none", 0)`` is
    the exact historical value, not a guess (same rule as :func:`resolve_edge_weights`).
    """
    tc = _read_json(os.path.join(checkpoint, "train_config.json")) or {}
    nested = tc.get("gnn") or {}
    spine_tools = tc.get("spine_tools", nested.get("spine_tools")) or "none"
    icl_examples = tc.get("icl_examples", nested.get("icl_examples"))
    return spine_tools, 0 if icl_examples is None else int(icl_examples)


def resolve_edge_weights(checkpoint: str) -> str:
    """Recover the train-time ``data.edge_weights`` policy ("gaussian" | "binary")
    so eval-time graph parsing matches training.

    Read from ``train_config.json``. A missing key means the checkpoint predates
    the knob, and every such run was trained with the Gaussian affinity — so
    "gaussian" is the exact historical value, not a guess.
    """
    tc = _read_json(os.path.join(checkpoint, "train_config.json"))
    if tc is not None and tc.get("edge_weights") is not None:
        return tc["edge_weights"]
    return "gaussian"


def resolve_injection_scope(checkpoint: str) -> str:
    """Recover the train-time ``data.injection_scope`` so generation wiring matches
    training (``decode_consistent`` checkpoints need decode-time injection armed).

    Read from ``train_config.json``. A missing key means the checkpoint predates the
    knob; every such run was trained with full-sequence maps — the exact historical
    value, not a guess (generation behavior is identical for every value except
    ``decode_consistent``).
    """
    tc = _read_json(os.path.join(checkpoint, "train_config.json"))
    if tc is not None and tc.get("injection_scope") is not None:
        return tc["injection_scope"]
    return "full_sequence"


def load_checkpoint(checkpoint: str, four_bit: bool, device: int):
    """Load a trained checkpoint for eval. Returns ``(model, tokenizer, is_gnn)``.

    Dispatches on :func:`is_gnn_checkpoint`. Raises ``FileNotFoundError`` if the
    dir has neither a graph config nor ``adapter_config.json`` (not a
    recognisable checkpoint).
    """
    if is_gnn_checkpoint(checkpoint):
        model, tok = loaders.graph_augmented_llm_from_pretrained(
            checkpoint, load_in_4bit=four_bit, device=device,
        )
        return model, tok, True
    if not os.path.exists(os.path.join(checkpoint, "adapter_config.json")):
        raise FileNotFoundError(
            f"{checkpoint} has neither a graph config (train_config.json with 'gnn' / "
            "gnn_config.json) nor adapter_config.json — not a recognisable checkpoint dir."
        )
    model, tok = loaders.from_pretrained(checkpoint, load_in_4bit=four_bit, device=device)
    return model, tok, False
