"""Driver-side Ψ transport for the vLLM graph engine (torch + prism only, no vllm).

The engine never sees the graph: Ψ is computed here, outside vLLM, from a loaded
:class:`~prism.models.gnn_llm.GraphAugmentedLLM` (trained ``pe_model`` /
``pe_proj`` / ``pe_gain`` / ``pe_norm``), then shipped per request as a
``[seq_len, hidden+1]`` tensor through the multimodal channel:

- columns ``[:hidden]`` — Ψ, zero off node-name token spans (built by the
  model's own ``build_pe_signal``, so permutation and ``pe_node_features``
  are honoured exactly as in HF eval);
- column ``[hidden]`` — injected-span indicator, consumed by the identity-RoPE
  path in :mod:`prism.models.vllm_graph.attention`. A separate column, not a
  Ψ≠0 heuristic, so span membership is exact.
"""
from __future__ import annotations

import torch

from prism.models.gnn_llm import (
    build_injection_map,
    find_last_graph_scope,
    node_token_variants,
)

# One extra transport column beyond the text hidden size: the span-indicator mask.
TRANSPORT_EXTRA_COLS = 1


def transport_dim(hidden_size: int) -> int:
    return hidden_size + TRANSPORT_EXTRA_COLS


def prompt_injection_map(tokenizer, prompt_ids: list[int], node_names) -> dict:
    """Injection map over the prompt, scoped to the LAST scene graph (ICL-safe)."""
    node_token_seqs = node_token_variants(list(node_names), tokenizer)
    scope_start = find_last_graph_scope(prompt_ids, tokenizer)
    return build_injection_map(prompt_ids, node_token_seqs, scope_start=scope_start)


def build_psi_transport(
    graph_model,
    tokenizer,
    prompt_ids: list[int],
    pyg_graph,
    *,
    permutation=None,
) -> tuple[torch.Tensor, dict]:
    """Ψ transport tensor ([seq, hidden+1], cpu) + injection map for one prompt.

    ``graph_model`` must be the unwrapped ``GraphAugmentedLLM`` core (callers
    peel PEFT with ``inference._core_graph_model``). Ψ comes from the model's
    own ``build_pe_signal`` so the numbers match the HF eval path bit-for-bit
    on the driver side.

    Fails loud when the checkpoint trained with identity-RoPE and the prompt
    ends inside a node mention — same contract as
    ``inference._identity_rope_kwargs``: the engine derives decode positions
    from the final prompt position, so a zeroed final position would silently
    restart the rollout at position 1.
    """
    injection_map = prompt_injection_map(tokenizer, prompt_ids, pyg_graph.node_names)
    seq_len = len(prompt_ids)

    device = next(graph_model.pe_proj.parameters()).device
    ids = torch.tensor([prompt_ids], device=device)
    with torch.no_grad():
        embeddings = graph_model.llm.get_input_embeddings()(ids)
        psi = graph_model.build_pe_signal(
            embeddings, [pyg_graph], [injection_map], permutation=permutation
        )[0]  # [seq, hidden]

    span = torch.zeros(seq_len, TRANSPORT_EXTRA_COLS, dtype=psi.dtype, device=psi.device)
    for spans in injection_map.values():
        for start, end in spans:
            span[start:min(end, seq_len)] = 1.0

    if graph_model._disable_graph_token_rope and seq_len > 1 and span[-1, 0] > 0:
        raise RuntimeError(
            "identity-RoPE zeroed the FINAL prompt position: the engine derives each "
            "generated position from the last one, so every generated token would be "
            "numbered from 1. The prompt must not end inside a node mention "
            f"(seq_len={seq_len})."
        )

    return torch.cat([psi, span], dim=-1).float().cpu(), injection_map
