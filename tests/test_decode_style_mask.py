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
# Backing tokens + name inventory (no name is a token-prefix of another).
MAP_SEQS = [[[10, 11]], [[12, 13]], [[14, 15]]]
MAP_IDS = [0, 10, 11, 12, 13, 14, 15, 10, 11, 9, 14, 15]


def test_decode_style_query_map_reduces_answer_spans_to_final_token():
    q = injection_diag.decode_style_query_map(FULL_MAP, ANSWER_START, MAP_IDS, MAP_SEQS)
    assert q[0] == [(1, 3), (8, 9)]          # prompt span whole, answer span -> final tok
    assert q[1] == [(3, 5)]                  # prompt-only node untouched
    assert q[2] == [(5, 7), (11, 12)]


def test_decode_trail_query_map_trails_until_next_mention():
    t = injection_diag.decode_trail_query_map(FULL_MAP, ANSWER_START, MAP_IDS, MAP_SEQS)
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
    q_map = injection_diag.decode_style_query_map(FULL_MAP, ANSWER_START, MAP_IDS, MAP_SEQS)
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
    q_map = injection_diag.decode_style_query_map(FULL_MAP, ANSWER_START, MAP_IDS, MAP_SEQS)
    with torch.no_grad():
        sym = model(input_ids=ids, graphs=[GRAPH], injection_maps=[FULL_MAP]).logits
        asym = model(input_ids=ids, graphs=[GRAPH], injection_maps=[q_map],
                     key_injection_maps=[FULL_MAP]).logits
    # Different wiring must change the logits (node rows lose in-span blocking).
    assert not torch.allclose(sym, asym)


# ---------------------------------------------------------------------------
# Decode-time parity (design note §4.2): the per-step rows armed by
# MaskDecodeInjector must reproduce the teacher-forced asymmetric bias exactly,
# and step-by-step cached decode must reproduce teacher-forced logits.
# ---------------------------------------------------------------------------
from prism.models.gnn_llm import MaskDecodeInjector, build_injection_map
from prism.models.gnn_llm import decode_style_query_map as dsqm

# node0 = [10, 11], node1 = [12], node2 = [13, 14]; path graph 0-1-2.
SEQS = [[[10, 11]], [[12]], [[13, 14]]]
PROMPT = [1, 10, 11, 12, 13, 14, 2]                 # node list in the prompt
ANSWER = [3, 10, 11, 4, 13, 14, 5]                  # mentions node0 then node2
FULL = PROMPT + ANSWER
A_START = len(PROMPT)


# Second fixture with PREFIX-AMBIGUOUS names: nodeA=[20] is a strict token-prefix
# of nodeB=[20,21] (the region_1 / region_10 pattern under digit-split BPE). The
# answer mentions nodeA resolved NEGATIVELY (followed by a non-extending token) —
# the case where the query tag must fire at the RESOLVING position (e, not e-1).
SEQS_AMB = [[[20]], [[20, 21]], [[22]]]
PROMPT_AMB = [1, 20, 5, 20, 21, 6, 22, 7]      # nodeA@(1,2) nodeB@(3,5) nodeC@(6,7)
ANSWER_AMB = [20, 9, 22, 4]                     # nodeA (resolves at 9), nodeC


def _parity_fixture(seqs=None, prompt=None, answer=None):
    seqs, prompt, answer = seqs or SEQS, prompt or PROMPT, answer or ANSWER
    n_nodes = len(seqs)
    model = GraphMaskLLM(_tiny_llm(), k_hops=1, symmetrize=True).eval()
    g = _graph(n_nodes, [[i, i + 1] for i in range(n_nodes - 1)])
    full = prompt + answer
    full_map = build_injection_map(full, seqs, scope_start=0)
    q_map = dsqm(full_map, len(prompt), full, seqs)
    prompt_map = build_injection_map(prompt, seqs, scope_start=0)
    return model, g, full_map, q_map, prompt_map, seqs, prompt, answer


def _assert_rows_parity(seqs=None, prompt=None, answer=None):
    model, g, full_map, q_map, prompt_map, seqs, prompt, answer = _parity_fixture(
        seqs, prompt, answer)
    full = prompt + answer
    ref = model.build_structural_mask(len(full), [g], [q_map], "cpu",
                                      dtype=torch.float32,
                                      key_injection_maps=[full_map])[0, 0]
    injector = MaskDecodeInjector(model, g, prompt_map, len(prompt), seqs)
    tagged = 0
    for i, tok in enumerate(answer):
        p = len(prompt) + i
        injector.pre_hook(None, (), {"input_ids": torch.tensor([[tok]])})
        row = model._decode_bias_row
        if row is None:
            assert (ref[p, :p + 1] == 0).all(), f"pos {p}: ref row nonzero but no decode row"
        else:
            tagged += 1
            assert row.shape == (1, 1, 1, p + 1)
            assert torch.equal(row[0, 0, 0], ref[p, :p + 1]), f"pos {p}: row mismatch"
    # Guard against vacuous parity: at least one decode step must actually arm a row.
    assert tagged >= 2, f"only {tagged} tagged steps — fixture exercises nothing"


def test_decode_rows_match_teacher_forced_bias():
    _assert_rows_parity()


def test_decode_rows_match_teacher_forced_bias_ambiguous_names():
    _assert_rows_parity(SEQS_AMB, PROMPT_AMB, ANSWER_AMB)


def test_decode_logits_match_teacher_forced():
    model, g, full_map, q_map, prompt_map, *_ = _parity_fixture()
    ids = torch.tensor([FULL])
    with torch.no_grad():
        ref = model(input_ids=ids, graphs=[g], injection_maps=[q_map],
                    key_injection_maps=[full_map]).logits[0]

    injector = MaskDecodeInjector(model, g, prompt_map, len(PROMPT), SEQS)
    model._struct_bias = model.build_structural_mask(
        len(PROMPT), [g], [prompt_map], "cpu")
    with torch.no_grad():
        out = model.llm(input_ids=torch.tensor([PROMPT]), use_cache=True)
        past = out.past_key_values
        assert torch.allclose(ref[:len(PROMPT)], out.logits[0], atol=1e-4)
        for i, tok in enumerate(ANSWER):
            injector.pre_hook(None, (), {"input_ids": torch.tensor([[tok]])})
            step = model.llm(input_ids=torch.tensor([[tok]]),
                             past_key_values=past, use_cache=True)
            past = step.past_key_values
            p = A_START + i
            assert torch.allclose(ref[p], step.logits[0, 0], atol=1e-4), f"pos {p}"
    model._struct_bias = None
    model._decode_bias_row = None


def test_partial_mention_ambiguity_defers_assignment():
    # nodeA = [20], nodeB = [20, 21]: after generating [20] alone the span is
    # extendable and must be deferred; a following non-extending token finalizes it.
    model = GraphMaskLLM(_tiny_llm(), k_hops=1, symmetrize=True).eval()
    g = _graph(2, [[0, 1]])
    seqs = [[[20]], [[20, 21]]]
    injector = MaskDecodeInjector(model, g, {}, 3, seqs)
    injector.pre_hook(None, (), {"input_ids": torch.tensor([[20]])})
    assert injector._suffix_spans() == {}          # deferred: [20] extendable to [20,21]
    injector.pre_hook(None, (), {"input_ids": torch.tensor([[22]])})
    assert injector._suffix_spans() == {0: [(0, 1)]}   # finalized as nodeA
    injector.pre_hook(None, (), {"input_ids": torch.tensor([[20]])})
    injector.pre_hook(None, (), {"input_ids": torch.tensor([[21]])})
    spans = injector._suffix_spans()
    assert spans[1] == [(2, 4)]                    # completed [20,21] = nodeB
    model._decode_bias_row = None
