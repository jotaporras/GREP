"""data.edge_weights plumbing: "gaussian" (historical affinity) vs "binary" (plain adjacency).

Covers, without any LLM:
  * ``scene_graph_dict_to_pyg`` — gaussian formula regression (exp(-d^2/2σ^2), σ = median
    edge length), binary attaches NO edge_weight, identical edge_index/distance_m in both
    modes, invalid value raises, default stays gaussian (backward compat).
  * Consumption — the GCN backbone actually reads the weights: forward output differs
    between the gaussian and binary graphs (otherwise "binary support" would be a no-op).
  * Collator threading — ``SpineDataCollator.edge_weights`` reaches
    ``scene_graph_dict_to_pyg`` (class-attr pattern shared with ``injection_scope``).
"""

from __future__ import annotations

import math

import pytest  # type: ignore[import-not-found]
import torch

from prism.data import data as data_mod
from prism.data import utils
from prism.models import gcn


# Two "communities" (regions 10m apart), one object hanging 1m off each region.
# Edge lengths: obj–region = 1.0 (x2), region–region = 10.0  =>  σ = median = 1.0.
SCENE = {
    "objects": [
        {"name": "obj_a", "coords": [0.0, 1.0], "description": ""},
        {"name": "obj_b", "coords": [10.0, 1.0], "description": ""},
    ],
    "regions": [
        {"name": "region_1", "coords": [0.0, 0.0], "description": ""},
        {"name": "region_2", "coords": [10.0, 0.0], "description": ""},
    ],
    "object_connections": [["obj_a", "region_1"], ["obj_b", "region_2"]],
    "region_connections": [["region_1", "region_2"]],
    "robot_location": "region_1",
}


def test_gaussian_matches_formula_and_is_default():
    g_default = utils.scene_graph_dict_to_pyg(dict(SCENE))
    g_gauss = utils.scene_graph_dict_to_pyg(dict(SCENE), edge_weights="gaussian")

    assert torch.equal(g_default.edge_weight, g_gauss.edge_weight)
    # σ = median(1, 1, 10) = 1.0; short edges → exp(-0.5), long edge → exp(-50).
    expected = {
        1.0: math.exp(-0.5),
        10.0: math.exp(-50.0),
    }
    for d, w in zip(g_gauss.distance_m.tolist(), g_gauss.edge_weight.tolist()):
        assert w == pytest.approx(expected[round(d, 6)], rel=1e-5)


def test_binary_attaches_no_edge_weight_and_keeps_structure():
    g_gauss = utils.scene_graph_dict_to_pyg(dict(SCENE), edge_weights="gaussian")
    g_bin = utils.scene_graph_dict_to_pyg(dict(SCENE), edge_weights="binary")

    assert getattr(g_bin, "edge_weight", None) is None
    assert torch.equal(g_bin.edge_index, g_gauss.edge_index)
    assert torch.equal(g_bin.distance_m, g_gauss.distance_m)
    assert g_bin.node_names == g_gauss.node_names


def test_invalid_edge_weights_value_raises():
    with pytest.raises(ValueError, match="edge_weights"):
        utils.scene_graph_dict_to_pyg(dict(SCENE), edge_weights="heat_kernel")


def test_gcn_consumes_the_weights():
    """Same GCN, same graph: gaussian vs binary outputs must differ."""
    g_gauss = utils.scene_graph_dict_to_pyg(dict(SCENE), edge_weights="gaussian")
    g_bin = utils.scene_graph_dict_to_pyg(dict(SCENE), edge_weights="binary")

    torch.manual_seed(0)
    net = gcn.GCN(in_channels=4, hidden_channels=8, num_layers=2, dropout=0.0, k=2)
    net.eval()
    x = torch.randn(g_gauss.num_nodes, 4)
    g_gauss.x = x.clone()
    g_bin.x = x.clone()

    with torch.no_grad():
        out_gauss = net(g_gauss)
        out_bin = net(g_bin)
    assert not torch.allclose(out_gauss, out_bin)


# Minimal tokenizer + token stream for _extract_graph (pattern from
# test_injection_scope.py): "scene graph:" anchors the scope, node mentions follow.
_PIECES = {
    1: "<bos>", 2: "task ", 3: "scene graph:", 4: " • ",
    101: "shed_1", 102: "field_11", 5: " , ", 6: " plan: ",
}
_INPUT_IDS = [1, 2, 3, 4, 101, 5, 102, 6]

_SMALL_SCENE = {
    "objects": [{"name": "shed_1", "coords": [0.0, 0.0]}],
    "regions": [{"name": "field_11", "coords": [1.0, 1.0]}],
    "object_connections": [["shed_1", "field_11"]],
    "region_connections": [],
    "robot_location": "field_11",
}


class _FakeTokenizer:
    _ENCODE = {
        "shed_1": [101], " shed_1": [101],
        "field_11": [102], " field_11": [102],
    }

    def encode(self, text, add_special_tokens=False):
        return list(self._ENCODE[text])

    def batch_decode(self, sequences, clean_up_tokenization_spaces=False):
        return ["".join(_PIECES[t] for t in seq) for seq in sequences]


def _make_collator(edge_weights):
    collator = object.__new__(data_mod.SpineDataCollator)
    collator.tokenizer = _FakeTokenizer()
    collator.injection_scope = "full_sequence"
    collator.edge_weights = edge_weights
    return collator


_EXAMPLE = {
    "input_ids": _INPUT_IDS,
    "answer_start": len(_INPUT_IDS),
    "scene_graph_dict": _SMALL_SCENE,
}


def test_collator_threads_edge_weights_binary():
    pyg_graph, imap, _, _ = _make_collator("binary")._extract_graph(_EXAMPLE)
    assert getattr(pyg_graph, "edge_weight", None) is None
    assert imap  # injection map still built

def test_collator_threads_edge_weights_gaussian():
    pyg_graph, _, _, _ = _make_collator("gaussian")._extract_graph(_EXAMPLE)
    assert pyg_graph.edge_weight is not None
    # Single edge → σ = its own length → weight = exp(-1/2) on every copy.
    assert torch.allclose(
        pyg_graph.edge_weight,
        torch.full_like(pyg_graph.edge_weight, math.exp(-0.5)),
    )
