"""Helpers for the teacher-forced injection-ablation diagnostic.

Driver: ``scripts/diag_injection_ablation.py``. The diagnostic separates the answer
token positions that measure *which node the model commits to* (decision) from the
positions that merely finish or repeat a name (completion / repeat), and grades a
teacher-forced forward pass on each set. Aggregate metrics such as
``graph_acc/answer_nodes`` are dominated by completion/repeat positions (a no-graph
baseline scores ~98% on them), so only the decision split can tell whether a graph
channel is being used.
"""
import torch

from prism.models import gnn_llm


# Token-position splits graded by the diagnostic (all within the assistant answer).
POSITION_SETS = ("decision", "completion", "repeat", "all_answer_nodes")


def partition_answer_node_positions(
    injection_map: dict[int, list[tuple[int, int]]],
    answer_start: int,
) -> dict[str, list[int]]:
    """Split answer-side node-mention token positions into decision/completion/repeat.

    Args:
        injection_map: full-sequence map ``{node_idx: [(start, end), ...]}`` (as built
            by ``build_injection_map`` over prompt + answer).
        answer_start: token index where the assistant answer begins.

    Returns:
        Dict with disjoint, sorted position lists:
        - ``decision``: first token of each node's FIRST mention at/after
          ``answer_start`` — where the model commits to a node identity.
        - ``completion``: remaining tokens of those first mentions (name continuations,
          copyable from the prompt's node list).
        - ``repeat``: every token of later answer mentions of the same node (copyable
          from earlier in the answer).
        - ``all_answer_nodes``: union of the above (the ``graph_acc/answer_nodes`` set).
    """
    decision: list[int] = []
    completion: list[int] = []
    repeat: list[int] = []
    for spans in injection_map.values():
        answer_spans = sorted(
            (start, end) for start, end in spans if start >= answer_start
        )
        if not answer_spans:
            continue
        first_start, first_end = answer_spans[0]
        decision.append(first_start)
        completion.extend(range(first_start + 1, first_end))
        for start, end in answer_spans[1:]:
            repeat.extend(range(start, end))
    decision, completion, repeat = sorted(decision), sorted(completion), sorted(repeat)
    return {
        "decision": decision,
        "completion": completion,
        "repeat": repeat,
        "all_answer_nodes": sorted(decision + completion + repeat),
    }


# Canonical implementation lives with the other map utilities in gnn_llm;
# aliased here so the diagnostic namespace stays complete.
decode_style_query_map = gnn_llm.decode_style_query_map


def decode_trail_query_map(
    injection_map: dict[int, list[tuple[int, int]]],
    answer_start: int,
    seq_len: int,
) -> dict[int, list[tuple[int, int]]]:
    """``decode_style_query_map`` + current-location context (decode-computable).

    After an answer-side mention completes, its node id rides the QUERY row of every
    subsequent position up to (not including) the start of the next answer-side
    mention — i.e. the inter-mention tokens (", then ") carry "the plan is currently
    at node i", giving the next-hop decision a direct bias toward i's neighbors.
    Everything here is computable from the generated prefix at decode time.
    """
    out = decode_style_query_map(injection_map, answer_start)
    answer_spans = sorted(
        (s, e, node_idx)
        for node_idx, spans in injection_map.items()
        for s, e in spans if s >= answer_start
    )
    for i, (s, e, node_idx) in enumerate(answer_spans):
        trail_end = answer_spans[i + 1][0] if i + 1 < len(answer_spans) else seq_len
        trail_end = min(trail_end, seq_len)
        if e - 1 < trail_end:
            out[node_idx] = sorted(set(out[node_idx]) - {(e - 1, e)} | {(e - 1, trail_end)})
    return out


def grade_positions(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    positions: list[int],
) -> dict[str, float]:
    """Next-token argmax accuracy and NLL at ``positions`` (batch size 1).

    Target frame: the token at position ``p`` is predicted by ``logits[p - 1]``
    (same convention as ``GraphTokenAccuracyMixin._accumulate_token_acc``).

    Args:
        logits: ``[1, S, V]`` model logits.
        input_ids: ``[1, S]`` token ids.
        positions: token positions to grade (position 0 has no prediction; out-of-range
            positions are skipped).

    Returns:
        ``{"n": count, "correct": argmax hits, "nll_sum": summed NLL}``.
    """
    pos = [p for p in positions if 1 <= p < input_ids.shape[1]]
    if not pos:
        return {"n": 0, "correct": 0, "nll_sum": 0.0}
    idx = torch.as_tensor(pos, device=logits.device)
    rows = logits[0, idx - 1].float()                       # [P, V]
    targets = input_ids[0, idx]                             # [P]
    log_probs = torch.log_softmax(rows, dim=-1)
    nll = -log_probs[torch.arange(len(pos), device=logits.device), targets]
    correct = (rows.argmax(dim=-1) == targets).sum()
    return {
        "n": len(pos),
        "correct": int(correct.item()),
        "nll_sum": float(nll.sum().item()),
    }


def merge_counts(total: dict[str, float], part: dict[str, float]) -> None:
    """Accumulate one example's ``grade_positions`` counts into ``total`` in place."""
    total["n"] += part["n"]
    total["correct"] += part["correct"]
    total["nll_sum"] += part["nll_sum"]


def summarize(counts: dict[str, float]) -> dict[str, float]:
    """Turn accumulated counts into ``{"n", "acc", "mean_nll"}`` (NaN-free on n=0)."""
    n = counts["n"]
    return {
        "n": n,
        "acc": counts["correct"] / n if n else 0.0,
        "mean_nll": counts["nll_sum"] / n if n else 0.0,
    }
