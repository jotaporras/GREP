"""e17 candidate E — pointer-fusion (logit-space node-distribution bias) tests.

Invariants this file locks: with ``ptr_gain = 0`` (the init), an enabled
pointer pathway is a BITWISE no-op on the lm_head logits; a nonzero gain moves
exactly the candidate token logits; gradients reach ptr_gain (live at init)
and, once the gain is open, the tower through p_gt; and the candidate
machinery (fresh starts + prefix continuations, prompt-boundary rule) is
deterministic in the tokens alone — teacher-forced and decode-side agree.
"""
import sys
sys.path.insert(0, "src")
sys.path.insert(0, "tests")

import torch

from prism.models.gnn_llm import (
    LearnableGraphMaskLLM,
    pointer_candidate_pairs,
    pointer_prefix_maps,
    pointer_step_candidates,
)
from test_learnable_graph_mask import _tiny_llm, _StubPE, _graph, DEVICE


def _model(pointer_fusion=True):
    torch.manual_seed(0)
    llm = _tiny_llm()
    return LearnableGraphMaskLLM(
        llm, _StubPE(d=8).to(DEVICE), alpha=0.7, layer_scope="all",
        pointer_fusion=pointer_fusion, fusion_d_gt=8).to(DEVICE)


def _inputs(model):
    g = _graph(3, [(0, 1), (1, 2)])
    imap = {0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}
    ids = torch.randint(0, 64, (1, 8), generator=torch.Generator().manual_seed(2)
                        ).to(DEVICE)
    return ids, [g], [imap]


# node 0: single-token [10] and [11]; node 1: two-token [20, 21]; node 2 shares
# node 1's first token then diverges — exercises the trie's shared-prefix case.
SEQS = [[[10], [11]], [[20, 21]], [[20, 22, 23]]]


def test_prefix_maps():
    first, prefmap, maxp = pointer_prefix_maps(SEQS)
    assert (0, 10) in first and (0, 11) in first
    assert (1, 20) in first and (2, 20) in first
    assert maxp == 2
    assert set(prefmap[(20,)]) == {(1, 21), (2, 22)}
    assert prefmap[(20, 22)] == [(2, 23)]


def test_step_candidates_continue_and_fresh():
    first, prefmap, maxp = pointer_prefix_maps(SEQS)
    # Empty suffix: fresh starts only.
    assert set(pointer_step_candidates([], first, prefmap, maxp)) == set(first)
    # After emitting 20: fresh starts PLUS both continuations.
    cands = set(pointer_step_candidates([99, 20], first, prefmap, maxp))
    assert cands == set(first) | {(1, 21), (2, 22)}
    # After 20, 22: the three-token name's last token is live.
    cands = set(pointer_step_candidates([20, 22], first, prefmap, maxp))
    assert (2, 23) in cands and (1, 21) not in cands


def test_candidate_pairs_prompt_boundary():
    # Continuations must NOT match across the prompt boundary (decode-side
    # parity: the injector state sees only generated tokens).
    toks = [1, 2, 20, 21]           # prompt = [1, 2, 20], completion = [21]
    pairs = pointer_candidate_pairs(toks, prompt_len=3, node_token_seqs=SEQS)
    at_2 = {(n, t) for s, n, t in pairs if s == 2}      # empty suffix state
    first, _, _ = pointer_prefix_maps(SEQS)
    assert at_2 == set(first)       # 20 in the prompt does NOT arm (1, 21)
    at_3 = {(n, t) for s, n, t in pairs if s == 3}      # suffix = [21]
    assert at_3 == set(first)       # 21 is no variant prefix


def test_zero_gain_is_bitwise_noop():
    m_ptr = _model(pointer_fusion=True)
    m_plain = _model(pointer_fusion=False)
    ids, gs, imaps = _inputs(m_ptr)
    cand = [[(s, 0, 5) for s in range(4, 8)]]
    with torch.no_grad():
        out_ptr = m_ptr(input_ids=ids, graphs=gs, injection_maps=imaps,
                        pointer_candidates=cand).logits
        out_plain = m_plain(input_ids=ids, graphs=gs,
                            injection_maps=imaps).logits
    assert torch.equal(out_ptr, out_plain)


def test_nonzero_gain_moves_only_candidate_logits():
    m = _model(pointer_fusion=True)
    ids, gs, imaps = _inputs(m)
    cand = [[(6, 0, 5), (6, 1, 7)]]
    with torch.no_grad():
        base = m(input_ids=ids, graphs=gs, injection_maps=imaps,
                 pointer_candidates=cand).logits
        m.ptr_gain.data.fill_(2.0)
        moved = m(input_ids=ids, graphs=gs, injection_maps=imaps,
                  pointer_candidates=cand).logits
    diff = (moved - base).abs()
    assert diff[0, 6, 5] > 0 and diff[0, 6, 7] > 0
    mask = torch.zeros_like(diff, dtype=torch.bool)
    mask[0, 6, 5] = mask[0, 6, 7] = True
    assert diff[~mask].abs().max() == 0


def test_gradients_reach_gain_then_tower():
    m = _model(pointer_fusion=True)
    ids, gs, imaps = _inputs(m)
    cand = [[(6, 0, 5)]]
    loss = m(input_ids=ids, graphs=gs, injection_maps=imaps,
             pointer_candidates=cand).logits.sum()
    loss.backward()
    # ptr_gain is upstream of tanh(0) with nonzero d/dgain — must receive grad.
    assert m.ptr_gain.grad is not None and m.ptr_gain.grad.abs().sum() > 0
    # Open the gain: the tower must now receive pointer-gradient through p_gt
    # (in ADDITION to the mask-bias gradient it already gets).
    m2 = _model(pointer_fusion=True)
    m2.ptr_gain.data.fill_(1.0)
    loss2 = m2(input_ids=ids, graphs=gs, injection_maps=imaps,
               pointer_candidates=cand).logits.sum()
    loss2.backward()
    assert m2.ptr_q.weight.grad is not None
    assert m2.ptr_q.weight.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in m2.pe_model.parameters())


def test_ptr_params_train_at_base_lr():
    m = _model(pointer_fusion=True)
    struct = set(map(id, m.structural_parameters()))
    base = set(map(id, m.base_lr_parameters()))
    for p in (m.ptr_gain, m.ptr_scale, m.ptr_q.weight, m.ptr_gate.weight):
        assert id(p) in base
        assert id(p) not in struct
    assert not (struct & base)
    assert _model(pointer_fusion=False).base_lr_parameters() == []


def test_decode_step_bias():
    # A cached decode step ([B,1] forward) under an armed per-step candidate
    # set must bias exactly that step's candidate logits.
    m = _model(pointer_fusion=True)
    ids, gs, imaps = _inputs(m)
    with torch.no_grad():
        psi = m.pe_model(gs[0]).float()
        m.ptr_gain.data.fill_(2.0)
        m._ptr_state = {"psi": [psi], "cand": None, "seq_len": 8}
        m._ptr_decode_cand = [[(0, 5)]]
        step = ids[:, :1]
        out_armed = m.llm(input_ids=step).logits
        m._ptr_decode_cand = None
        m._ptr_state = None
        out_plain = m.llm(input_ids=step).logits
    diff = (out_armed - out_plain).abs()
    assert diff[0, 0, 5] > 0
    mask = torch.zeros_like(diff, dtype=torch.bool)
    mask[0, 0, 5] = True
    assert diff[~mask].abs().max() == 0
