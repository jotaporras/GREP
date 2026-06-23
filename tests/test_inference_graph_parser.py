"""Tests for GraphAugmentedInMemoryLLM._parse_all_pyg_graphs and _generate_tokens."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import torch


from prism.models.inference import GraphAugmentedInMemoryLLM  # noqa: E402
from prism.models import gnn_llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SINGLE_LINE_GRAPH = (
    "{'objects': [{'name': 'house_1', 'coords': [0, 0]}], "
    "'regions': [{'name': 'field_1', 'coords': [1, 1]}], "
    "'object_connections': [['house_1', 'field_1']], "
    "'region_connections': [['field_1', 'field_1']], "
    "'robot_location': 'field_1'}"
)

MULTILINE_GRAPH_DICT = {
    "objects": [{"name": "pickup_truck_1", "coords": [2, 3]}],
    "regions": [{"name": "field_1", "coords": [1, 1]}],
    "object_connections": [["pickup_truck_1", "field_1"]],
    "region_connections": [["field_1", "field_1"]],
    "robot_location": "field_1",
}
MULTILINE_GRAPH = json.dumps(MULTILINE_GRAPH_DICT, indent=2)


def _make_llm() -> GraphAugmentedInMemoryLLM:
    """Return a GraphAugmentedInMemoryLLM with mocked model/tokenizer (not used in parsing)."""
    llm = GraphAugmentedInMemoryLLM.__new__(GraphAugmentedInMemoryLLM)
    llm.model = MagicMock()
    llm.tokenizer = MagicMock()
    llm.device = "cpu"
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_parses_single_line_scene_graph():
    llm = _make_llm()
    msg = [{"role": "user", "content": f"scene graph: {SINGLE_LINE_GRAPH}"}]
    graphs = llm._parse_all_pyg_graphs(msg)
    assert len(graphs) == 1
    assert graphs[0].robot_location == "field_1"


def test_parses_multiline_json_scene_graph():
    """re.DOTALL must be set so multi-line JSON from json.dumps(indent=2) is captured."""
    llm = _make_llm()
    msg = [{"role": "user", "content": f"scene graph: {MULTILINE_GRAPH}"}]
    graphs = llm._parse_all_pyg_graphs(msg)
    assert len(graphs) == 1
    assert graphs[0].robot_location == "field_1"


def test_parses_all_graphs_from_multi_message_prompt():
    """A SPINE prompt with 5 few-shot examples + 1 real task should yield 6 graphs."""
    llm = _make_llm()
    example_msg = {"role": "user", "content": f"scene graph: {SINGLE_LINE_GRAPH}"}
    real_msg = {"role": "user", "content": f"scene graph: {MULTILINE_GRAPH}"}
    msg = [example_msg] * 5 + [{"role": "assistant", "content": "plan: ..."}, real_msg]
    graphs = llm._parse_all_pyg_graphs(msg)
    assert len(graphs) == 6


def test_last_graph_is_real_task_graph():
    """The last parsed graph must be from the real task message, not a few-shot example."""
    llm = _make_llm()

    example_graph = (
        "{'objects': [{'name': 'example_node_1', 'coords': [0, 0]}], "
        "'regions': [{'name': 'example_region_1', 'coords': [1, 1]}], "
        "'object_connections': [['example_node_1', 'example_region_1']], "
        "'region_connections': [['example_region_1', 'example_region_1']], "
        "'robot_location': 'example_node_1'}"
    )
    example_msg = {"role": "user", "content": f"scene graph: {example_graph}"}
    real_msg = {"role": "user", "content": f"scene graph: {MULTILINE_GRAPH}"}
    msg = [example_msg] * 5 + [real_msg]

    graphs = llm._parse_all_pyg_graphs(msg)
    assert graphs[-1].robot_location == "field_1", (
        f"Expected real graph robot_location='field_1', got '{graphs[-1].robot_location}'"
    )


def test_ignores_non_user_messages():
    llm = _make_llm()
    msg = [
        {"role": "system", "content": f"scene graph: {SINGLE_LINE_GRAPH}"},
        {"role": "assistant", "content": f"scene graph: {SINGLE_LINE_GRAPH}"},
    ]
    graphs = llm._parse_all_pyg_graphs(msg)
    assert graphs == []


def test_returns_empty_list_when_no_graph():
    llm = _make_llm()
    msg = [{"role": "user", "content": "What should I do next?"}]
    graphs = llm._parse_all_pyg_graphs(msg)
    assert graphs == []


def test_skips_malformed_graph_and_continues():
    """Bad JSON in one message should not prevent parsing valid graphs in others."""
    llm = _make_llm()
    msg = [
        {"role": "user", "content": "scene graph: {this is not valid python}"},
        {"role": "user", "content": f"scene graph: {SINGLE_LINE_GRAPH}"},
    ]
    graphs = llm._parse_all_pyg_graphs(msg)
    assert len(graphs) == 1
    assert graphs[0].robot_location == "field_1"


# ---------------------------------------------------------------------------
# _generate_tokens: last-graph-only PE injection (in-attention architecture)
# ---------------------------------------------------------------------------
#
# The graph PE is injected INSIDE attention: plain token embeddings go to generate,
# and Ψ is delivered out-of-band via ``graph_model._pe_signal``. Which tokens receive
# Ψ is decided by ``find_last_graph_scope`` (scope = the last scene-graph block) plus
# ``build_injection_map`` (match node tokens only at/after that scope), so only the
# real task graph — never an ICL-example graph — is injected.

# Fake token IDs: a scene-graph block marker, two node names, and filler.
TOKEN_MARKER = 1     # decodes to the "scene graph: •" block signature
TOKEN_ALPHA = 10     # "alpha_node"
TOKEN_BETA = 20      # "beta_node"
TOKEN_OTHER = 99     # filler / unrelated token


class _FakeTokenizer:
    """Minimal id<->piece tokenizer for the injection helpers.

    ``batch_decode`` is per-token (one id per sublist) so ``find_last_graph_scope``
    can rebuild the text; ``encode`` ignores a leading space so both node-name
    variants (standalone and space-preceded) map to the same id.
    """

    _ID_TO_PIECE = {
        TOKEN_MARKER: "scene graph: •",
        TOKEN_ALPHA: "alpha_node",
        TOKEN_BETA: "beta_node",
        TOKEN_OTHER: " ",
    }
    _PIECE_TO_ID = {"alpha_node": [TOKEN_ALPHA], "beta_node": [TOKEN_BETA]}
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return self._PIECE_TO_ID[text.strip()]

    def batch_decode(self, seqs, clean_up_tokenization_spaces=False):
        return ["".join(self._ID_TO_PIECE[int(t)] for t in seq) for seq in seqs]


def test_injection_targets_only_last_graph_scope():
    """find_last_graph_scope + build_injection_map inject ONLY the last graph block.

    The same node name appears in an earlier (ICL) block and the final (task) block;
    only the occurrence at/after the last ``scene graph: •`` marker is injected.
    """
    tok = _FakeTokenizer()
    # [marker, alpha, filler, marker, alpha, filler] — alpha appears in both blocks.
    input_ids = [TOKEN_MARKER, TOKEN_ALPHA, TOKEN_OTHER,
                 TOKEN_MARKER, TOKEN_ALPHA, TOKEN_OTHER]

    scope_start = gnn_llm.find_last_graph_scope(input_ids, tok)
    assert scope_start == 3, f"scope should start at the LAST marker (idx 3), got {scope_start}"

    node_seqs = gnn_llm.node_token_variants(["alpha_node"], tok)   # last graph's nodes
    injection_map = gnn_llm.build_injection_map(input_ids, node_seqs, scope_start=scope_start)

    spans = [span for spans in injection_map.values() for span in spans]
    assert spans == [(4, 5)], (
        f"only the last-graph alpha (idx 4) should be injected, not the ICL one (idx 1); "
        f"got {spans}"
    )


def test_generate_tokens_passes_plain_embeddings_and_arms_pe_signal():
    """Base R-PEARL path: plain token embeddings reach generate (Ψ is NOT added to
    inputs_embeds); Ψ is armed on ``_pe_signal`` for in-attention injection instead."""
    H = 4
    llm = _make_llm()
    llm.tokenizer = _FakeTokenizer()
    llm.permutation = None

    input_ids = torch.tensor([[TOKEN_MARKER, TOKEN_ALPHA, TOKEN_OTHER]])
    base = torch.full((1, 3, H), 0.1)
    captured = {}

    class _StubGraphModel:
        """Stands in for the unwrapped GraphAugmentedLLM core. Not a PEFT/real model
        type, so ``_core_graph_model`` returns it as-is and the base R-PEARL branch runs."""

        def __init__(self):
            self._pe_signal = None
            self.llm = MagicMock()
            self.llm.get_input_embeddings.return_value = lambda ids: base.clone()

            def _generate(**kwargs):
                captured.update(kwargs)
                captured["pe_signal_armed"] = self._pe_signal
                return torch.zeros(1, 1, dtype=torch.long)

            self.llm.generate.side_effect = _generate

        def build_pe_signal(self, embeddings, graphs, maps, permutation=None):
            captured["pe_signal_input"] = embeddings.clone()
            return torch.ones_like(embeddings)   # a non-zero Ψ

    llm.model = _StubGraphModel()

    pyg = MagicMock()
    pyg.robot_location = "alpha_node"
    pyg.node_names = ["alpha_node"]

    llm._generate_tokens(
        input_ids, attention_mask=torch.ones(1, 3, dtype=torch.long),
        pyg_graphs=[pyg], max_new_tokens=8,
    )

    # generate received PLAIN embeddings — Ψ was not folded into inputs_embeds.
    assert "inputs_embeds" in captured
    assert torch.allclose(captured["inputs_embeds"], base), (
        "generate must receive plain token embeddings; Ψ is injected in-attention, "
        "not added to inputs_embeds"
    )
    # build_pe_signal saw the same plain embeddings, and Ψ was armed during generate
    # then cleared afterward.
    assert torch.allclose(captured["pe_signal_input"], base)
    assert captured["pe_signal_armed"] is not None, "_pe_signal must be armed during generate"
    assert llm.model._pe_signal is None, "_pe_signal must be cleared after generate"


def test_graph_order_matches_message_order():
    """Each parsed graph must correspond to its source message in forward order."""
    llm = _make_llm()

    def _make_msg(robot_loc: str) -> dict:
        graph = {
            "objects": [{"name": f"obj_{robot_loc}", "coords": [0, 0]}],
            "regions": [{"name": robot_loc, "coords": [1, 1]}],
            "object_connections": [[f"obj_{robot_loc}", robot_loc]],
            "region_connections": [[robot_loc, robot_loc]],
            "robot_location": robot_loc,
        }
        return {"role": "user", "content": f"scene graph: {graph}"}

    locations = ["loc_a", "loc_b", "loc_c", "loc_d", "loc_e", "real_loc"]
    msg = [_make_msg(loc) for loc in locations]

    graphs = llm._parse_all_pyg_graphs(msg)

    assert len(graphs) == len(locations)
    for i, (graph, expected_loc) in enumerate(zip(graphs, locations)):
        assert graph.robot_location == expected_loc, (
            f"Graph at index {i}: expected robot_location='{expected_loc}', "
            f"got '{graph.robot_location}'"
        )
