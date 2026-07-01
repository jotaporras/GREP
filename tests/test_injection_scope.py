"""Tests for injection-scope clamping and the diagnostic position partition.

Covers:
  1. ``clamp_injection_map`` — spans before / straddling / after the boundary.
  2. ``partition_answer_node_positions`` — decision vs completion vs repeat splits.
  3. ``SpineDataCollator._extract_graph`` — ``injection_scope='prompt_only'`` clamps
     the map at ``answer_start``; ``'full_sequence'`` preserves answer-side spans.

Run with::

    PYTHONPATH=src pytest tests/test_injection_scope.py -v
"""
from prism.data import data as data_mod
from prism.eval import injection_diag
from prism.models import gnn_llm


# ── 1. clamp_injection_map ──────────────────────────────────────────────

def test_clamp_keeps_spans_before_boundary():
    imap = {0: [(2, 4)], 1: [(5, 7)]}
    assert gnn_llm.clamp_injection_map(imap, 10) == {0: [(2, 4)], 1: [(5, 7)]}


def test_clamp_drops_spans_after_boundary():
    imap = {0: [(2, 4), (12, 14)], 1: [(15, 17)]}
    assert gnn_llm.clamp_injection_map(imap, 10) == {0: [(2, 4)]}


def test_clamp_truncates_straddling_span():
    imap = {0: [(8, 12)]}
    assert gnn_llm.clamp_injection_map(imap, 10) == {0: [(8, 10)]}


def test_clamp_boundary_exact():
    # Span starting exactly at the boundary is answer-side: dropped.
    imap = {0: [(10, 12)], 1: [(9, 10)]}
    assert gnn_llm.clamp_injection_map(imap, 10) == {1: [(9, 10)]}


# ── 2. partition_answer_node_positions ─────────────────────────────────

def test_partition_splits_decision_completion_repeat():
    # Node 0: prompt span (2,4); first answer mention (10,13); repeat (20,22).
    # Node 1: prompt-only mentions -> contributes nothing.
    imap = {0: [(2, 4), (10, 13), (20, 22)], 1: [(5, 7)]}
    sets = injection_diag.partition_answer_node_positions(imap, answer_start=8)
    assert sets["decision"] == [10]
    assert sets["completion"] == [11, 12]
    assert sets["repeat"] == [20, 21]
    assert sets["all_answer_nodes"] == [10, 11, 12, 20, 21]


def test_partition_sets_are_disjoint_and_cover():
    imap = {0: [(10, 12), (14, 16)], 1: [(12, 14)]}
    sets = injection_diag.partition_answer_node_positions(imap, answer_start=0)
    d, c, r = set(sets["decision"]), set(sets["completion"]), set(sets["repeat"])
    assert d.isdisjoint(c) and d.isdisjoint(r) and c.isdisjoint(r)
    assert sorted(d | c | r) == sets["all_answer_nodes"]


def test_partition_single_token_mention_has_no_completion():
    imap = {0: [(10, 11), (15, 16)]}
    sets = injection_diag.partition_answer_node_positions(imap, answer_start=8)
    assert sets["decision"] == [10]
    assert sets["completion"] == []
    assert sets["repeat"] == [15]


# ── 3. SpineDataCollator injection_scope wiring ─────────────────────────

# Token stream (id, decoded piece); "scene graph: •" anchors the scope, node
# mentions appear in both the scene block and the answer (answer_start = 8).
_PIECES = {
    1: "<bos>", 2: "task ", 3: "scene graph:", 4: " • ",
    101: "shed_1", 102: "field_11", 5: " , ", 6: " plan: ", 7: " -> ",
}
_INPUT_IDS = [1, 2, 3, 4, 101, 5, 102, 6, 101, 7, 102]
_ANSWER_START = 8

_SCENE_GRAPH = {
    "objects": [{"name": "shed_1", "coords": [0, 0]}],
    "regions": [{"name": "field_11", "coords": [1, 1]}],
    "object_connections": [["shed_1", "field_11"]],
    "region_connections": [],
    "robot_location": "field_11",
}


class _FakeTokenizer:
    """Minimal tokenizer for _extract_graph: encode node names + per-token decode."""

    _ENCODE = {
        "shed_1": [101], " shed_1": [101],
        "field_11": [102], " field_11": [102],
    }

    def encode(self, text, add_special_tokens=False):
        return list(self._ENCODE[text])

    def batch_decode(self, sequences, clean_up_tokenization_spaces=False):
        return ["".join(_PIECES[t] for t in seq) for seq in sequences]


def _make_collator(scope):
    collator = object.__new__(data_mod.SpineDataCollator)
    collator.tokenizer = _FakeTokenizer()
    collator.injection_scope = scope
    return collator


_EXAMPLE = {
    "input_ids": _INPUT_IDS,
    "answer_start": _ANSWER_START,
    "scene_graph_dict": _SCENE_GRAPH,
}


def test_extract_graph_full_sequence_includes_answer_mentions():
    _, imap = _make_collator("full_sequence")._extract_graph(_EXAMPLE)
    starts = sorted(s for spans in imap.values() for s, _ in spans)
    assert starts == [4, 6, 8, 10]  # scene mentions AND answer mentions


def test_extract_graph_prompt_only_clamps_at_answer_start():
    _, imap = _make_collator("prompt_only")._extract_graph(_EXAMPLE)
    starts = sorted(s for spans in imap.values() for s, _ in spans)
    assert starts == [4, 6]  # answer-side spans (8, 10) removed
    ends = [e for spans in imap.values() for _, e in spans]
    assert all(e <= _ANSWER_START for e in ends)


def test_extract_graph_prompt_only_keeps_all_nodes_covered():
    pyg_graph, imap = _make_collator("prompt_only")._extract_graph(_EXAMPLE)
    # Every graph node keeps at least one prompt-side span (needed e.g. by
    # gt_llm's word_embeddings feature mode).
    assert set(imap.keys()) == set(range(pyg_graph.num_nodes))
