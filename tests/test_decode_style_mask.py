"""Tests for the decode-consistency (asymmetric query/key) injection wiring.

Covers the two pieces added for the decode-style diagnostic (decode-time design
note §3):

1. ``injection_diag.decode_style_query_map`` / ``decode_trail_query_map`` — the
   causal QUERY-role maps (prompt spans whole; answer spans reduced to their final
   token, optionally trailing the id through inter-mention tokens).
2. ``build_structural_mask(..., key_injection_maps=...)`` — bias rows wired from
   the query map, columns from the key map; ``key_injection_maps=None`` must
   reproduce the historical symmetric behavior exactly.

All tiny, random-init, CPU — no GPU required.
"""
import sys
sys.path.insert(0, "src")

import torch
from torch_geometric.data import Data

from prism.eval import injection_diag
from prism.models.gnn_llm import GraphMaskLLM


def _tiny_llm(hidden=32):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=hidden, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, attn_implementation="eager")
    return LlamaForCausalLM(cfg)


def _graph(n, edges):
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    g = Data(x=torch.zeros(n, 1), edge_index=ei, num_nodes=n)
    g.node_names = [f"node{i}" for i in range(n)]
    return g


NEG = torch.finfo(torch.float32).min

# Path graph 0-1-2 (node 0 and 2 NON-adjacent at k_hops=1).
GRAPH = _graph(3, [[0, 1], [1, 2]])

# Layout (seq_len=12): prompt mentions node0@[1,3), node1@[3,5), node2@[5,7);
# answer starts at 7 with mentions node0@[7,9), node2@[10,12).
FULL_MAP = {0: [(1, 3), (7, 9)], 1: [(3, 5)], 2: [(5, 7), (10, 12)]}
ANSWER_START = 7
SEQ = 12


def test_decode_style_query_map_reduces_answer_spans_to_final_token():
    q = injection_diag.decode_style_query_map(FULL_MAP, ANSWER_START)
    assert q[0] == [(1, 3), (8, 9)]          # prompt span whole, answer span -> final tok
    assert q[1] == [(3, 5)]                  # prompt-only node untouched
    assert q[2] == [(5, 7), (11, 12)]


def test_decode_trail_query_map_trails_until_next_mention():
    t = injection_diag.decode_trail_query_map(FULL_MAP, ANSWER_START, SEQ)
    # node0's answer mention ends at 9; next answer mention starts at 10 -> trail [8,10).
    assert t[0] == [(1, 3), (8, 10)]
    # node2's is the last mention -> trails to seq_len.
    assert t[2] == [(5, 7), (11, 12)]
    assert t[1] == [(3, 5)]


def test_key_maps_none_reproduces_symmetric_mask():
    model = GraphMaskLLM(_tiny_llm(), k_hops=1, symmetrize=True).eval()
    sym = model.build_structural_mask(SEQ, [GRAPH], [FULL_MAP], "cpu",
                                      dtype=torch.float32)
    sym2 = model.build_structural_mask(SEQ, [GRAPH], [FULL_MAP], "cpu",
                                       dtype=torch.float32,
                                       key_injection_maps=[FULL_MAP])
    assert torch.equal(sym, sym2)


def test_asymmetric_mask_rows_and_columns():
    model = GraphMaskLLM(_tiny_llm(), k_hops=1, symmetrize=True).eval()
    q_map = injection_diag.decode_style_query_map(FULL_MAP, ANSWER_START)
    bias = model.build_structural_mask(SEQ, [GRAPH], [q_map], "cpu",
                                       dtype=torch.float32,
                                       key_injection_maps=[FULL_MAP])[0, 0]

    # In-span (non-final) answer QUERY rows are unwired: row 7 (node0 mention start)
    # gets no blocking anywhere, even against non-adjacent node2 keys.
    assert (bias[7] == 0).all()
    # The span-final answer query row IS wired: position 8 (node0) is blocked against
    # node2's key columns (prompt 5-6 and answer 10-11) — non-adjacent...
    assert bias[8, 5] == NEG and bias[8, 10] == NEG
    # ...but allowed toward node1 (adjacent, prompt cols 3-4) and its own columns.
    assert bias[8, 3] == 0 and bias[8, 1] == 0
    # KEY columns of answer mentions are live for wired queries: prompt node2 query
    # (row 5) is blocked against node0's ANSWER key column 7 (non-adjacent pair).
    assert bias[5, 7] == NEG
    # Unwired-key columns never blocked: column 9 (non-node) is 0 everywhere.
    assert (bias[:, 9] == 0).all()


def test_forward_accepts_key_injection_maps():
    model = GraphMaskLLM(_tiny_llm(), k_hops=1, symmetrize=True).eval()
    ids = torch.randint(0, 64, (1, SEQ))
    q_map = injection_diag.decode_style_query_map(FULL_MAP, ANSWER_START)
    with torch.no_grad():
        sym = model(input_ids=ids, graphs=[GRAPH], injection_maps=[FULL_MAP]).logits
        asym = model(input_ids=ids, graphs=[GRAPH], injection_maps=[q_map],
                     key_injection_maps=[FULL_MAP]).logits
    # Different wiring must change the logits (node rows lose in-span blocking).
    assert not torch.allclose(sym, asym)
