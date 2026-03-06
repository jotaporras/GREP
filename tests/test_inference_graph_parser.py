"""Tests for GraphAugmentedInMemoryLLM._parse_all_pyg_graphs and _generate_tokens."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import torch


from prism.models.inference import GraphAugmentedInMemoryLLM  # noqa: E402


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
# _generate_tokens: PE injection tests
# ---------------------------------------------------------------------------

def _make_two_graph_msg():
    """Two user messages, each with one node whose name encodes to a distinct token ID."""
    graph_a = {
        "objects": [],
        "regions": [{"name": "alpha_node", "coords": [0, 0]}],
        "object_connections": [],
        "region_connections": [["alpha_node", "alpha_node"]],
        "robot_location": "alpha_node",
    }
    graph_b = {
        "objects": [],
        "regions": [{"name": "beta_node", "coords": [1, 1]}],
        "object_connections": [],
        "region_connections": [["beta_node", "beta_node"]],
        "robot_location": "beta_node",
    }
    return [
        {"role": "user", "content": f"scene graph: {graph_a}"},
        {"role": "user", "content": f"scene graph: {graph_b}"},
    ]


TOKEN_ALPHA = 10   # fake token ID for "alpha_node"
TOKEN_BETA  = 20   # fake token ID for "beta_node"
TOKEN_OTHER = 99   # unrelated token


def test_generate_tokens_only_injects_last_graph_pe():
    """Only the last graph (real task) should have its PE injected."""
    H = 4  # hidden size

    llm = _make_llm()

    # input_ids: [OTHER, ALPHA, BETA, OTHER]  (batch size 1)
    input_ids = torch.tensor([[TOKEN_OTHER, TOKEN_ALPHA, TOKEN_BETA, TOKEN_OTHER]])

    # Base embeddings: all zeros so any addition is easy to see
    base = torch.zeros(1, 4, H)
    llm.model.llm.get_input_embeddings.return_value = MagicMock(
        return_value=base.clone()
    )

    # Tokenizer: encode node names -> their respective fake token IDs
    def _encode(name, add_special_tokens=False):
        return {
            "alpha_node": [TOKEN_ALPHA],
            "beta_node":  [TOKEN_BETA],
        }[name]
    llm.tokenizer.encode.side_effect = _encode

    # PE values: graph_a node gets [1,0,0,0], graph_b node gets [0,1,0,0]
    pe_alpha = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pe_beta  = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    llm.model.pe_model.side_effect = lambda g: g   # pass-through
    llm.model.pe_proj.side_effect  = (
        lambda g: pe_alpha if g.robot_location == "alpha_node" else pe_beta
    )

    # Capture the embeddings passed to model.generate
    captured = {}
    def _fake_generate(**kwargs):
        captured["embeds"] = kwargs["inputs_embeds"].clone()
        return torch.zeros(1, 1, dtype=torch.long)
    llm.model.generate.side_effect = _fake_generate

    llm._generate_tokens(input_ids, _make_two_graph_msg(), max_new_tokens=16)

    embeds = captured["embeds"]  # shape [1, 4, H]

    # Only graph_b (last graph) PE should be injected.
    # Position 0 (OTHER): unchanged
    assert torch.all(embeds[0, 0] == 0), "OTHER token should not be modified"
    # Position 1 (ALPHA): unchanged — graph_a PE is NOT injected
    assert torch.all(embeds[0, 1] == 0), \
        f"alpha_node from ICL graph should NOT get PE, got {embeds[0, 1]}"
    # Position 2 (BETA): should have pe_beta added (last graph's node)
    assert torch.allclose(embeds[0, 2], pe_beta[0]), \
        f"beta_node position: expected {pe_beta[0]}, got {embeds[0, 2]}"
    # Position 3 (OTHER): unchanged
    assert torch.all(embeds[0, 3] == 0), "trailing OTHER token should not be modified"


def test_generate_tokens_uses_only_last_graph_even_with_shared_nodes():
    """With last-graph-only injection, PE is applied once even if multiple graphs share a node."""
    H = 4
    llm = _make_llm()

    # One node name shared across both graphs
    input_ids = torch.tensor([[TOKEN_ALPHA]])
    base = torch.zeros(1, 1, H)
    llm.model.llm.get_input_embeddings.return_value = MagicMock(return_value=base.clone())

    llm.tokenizer.encode.return_value = [TOKEN_ALPHA]

    pe_val = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    llm.model.pe_model.side_effect = lambda g: g
    llm.model.pe_proj.return_value = pe_val   # same PE for both graphs

    captured = {}
    def _fake_generate(**kwargs):
        captured["embeds"] = kwargs["inputs_embeds"].clone()
        return torch.zeros(1, 1, dtype=torch.long)
    llm.model.generate.side_effect = _fake_generate

    # Two messages → two graphs, same node token in both
    msg = [
        {"role": "user", "content": f"scene graph: {{'objects': [], 'regions': [{{'name': 'alpha_node', 'coords': [0,0]}}], 'object_connections': [], 'region_connections': [['alpha_node','alpha_node']], 'robot_location': 'alpha_node'}}"},
        {"role": "user", "content": f"scene graph: {{'objects': [], 'regions': [{{'name': 'alpha_node', 'coords': [0,0]}}], 'object_connections': [], 'region_connections': [['alpha_node','alpha_node']], 'robot_location': 'alpha_node'}}"},
    ]

    llm._generate_tokens(input_ids, msg, max_new_tokens=16)

    embeds = captured["embeds"]
    # Only last graph is used → PE applied once, not twice
    assert torch.allclose(embeds[0, 0], pe_val[0]), \
        f"Expected single PE {pe_val[0]}, got {embeds[0, 0]}"


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
