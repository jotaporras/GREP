"""Smoke tests for the magnetic-composite attention mask (``gnn.mask_composite``).

``MagCompGraphLLM`` adds ONE additive pre-softmax bias over the scene-graph scope::

    Phi'(q) = T(Phi(q; L_bar^(r), H))     GT blocks applied PER PROBE, inside E_q
    Psi     = E_q[Phi']
    C       = E_q[Phi' Phi'^T] - Psi Psi^T
    A       = sm[ (Q x_n)^T (K x_m) / sqrt(d_h) + beta * C_tok[n, m] + M_causal ]

The five properties asserted here are the ones a silent misconfiguration would break
without changing a shape:

1. ``C_tok`` is symmetric PSD (it is a centered Gram matrix; a sign slip or an
   un-centered second moment loses that).
2. ``beta_init = 0`` reproduces the base LLM's logits EXACTLY — the arm's cold start.
3. The composite ``edge_index`` carries the four edge classes with the right
   directedness and node counts (the crosslink direction is an architecture invariant:
   scene -> token, never the reverse, or ``Theta^(r)`` cancels on those edges).
4. ``pe_pool='gt'`` CHANGES ``C_tok`` versus ``pe_pool='pe'`` — proof that T runs inside
   the probe loop rather than being applied to Psi (T is nonlinear, so the two differ).
5. ``beta`` has gradient at 0 while ``C`` does not (the documented one-step stall), so
   the channel provably opens.

Runs on CPU: the GT blocks go through ``torch.sparse.sampled_addmm``, which has no MPS
kernel, and the models here are tiny by construction (c <= 24, M <= 6).
"""
import sys

sys.path.insert(0, "src")

import pytest
import torch
from torch_geometric.data import Data

from prism.models.gnn_llm import (CompositeDecodeInjector, MagCompGraphLLM,
                                  build_composite_graph, build_injection_map,
                                  defer_open_mentions, node_token_variants)
from prism.models.gt import GraphTransformer

DEVICE = torch.device("cpu")
CYCLE = 24          # small stand-in for the config's mask_cycle_size = 8192
SCOPE_START = 4     # tokens [0, 4) are "system prompt": no bias at all


class _Tokenizer:
    """Minimal stand-in for the pieces ``find_last_graph_scope`` touches.

    It decodes token id ``t`` to ``"<t>"`` except the marker id, which decodes to the
    literal block signature — so the scope starts at whatever position the marker sits at.
    """

    MARKER = 999

    def batch_decode(self, seqs, **kwargs):
        return ["scene graph: •" if s[0] == self.MARKER else f"<{s[0]}>" for s in seqs]

    # Node name -> its token ids. The ONE source the model derives variants from, so a
    # test that needs a different naming (a prefix pair, say) overrides this per instance.
    VOCAB = {"Park": [13, 14, 15], "Office": [19, 20, 21],
             "House": [77, 78], "Hankee": [7, 8, 9, 10]}

    def encode(self, text, **kwargs):
        return list(self.VOCAB.get(text.strip(), []))


def _ids(seq_len=CYCLE + SCOPE_START):
    """Token ids whose last ``scene graph: •`` marker sits at ``SCOPE_START``."""
    ids = list(range(1, seq_len + 1))
    ids[SCOPE_START] = _Tokenizer.MARKER
    return torch.tensor([ids], dtype=torch.long)


def _scene():
    """Park - Office - House - Hankee, undirected (both directions in edge_index)."""
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]
    g = Data(x=torch.zeros(4, 1), edge_index=torch.tensor(edges).t().contiguous(),
             num_nodes=4)
    g.node_names = ["Park", "Office", "House", "Hankee"]
    return g


def _injection_map():
    """The doc's sample mentions, in FULL-sequence coordinates (scope_start = 4).

    Hankee -> 4 sub-tokens, Park -> 3, Office -> 3, House -> none (uncrosslinked, which
    is exactly the case the anchor exists to keep connected).
    """
    return {3: [(6, 10)], 0: [(12, 15)], 1: [(18, 21)]}


def _gt(pe_pool="gt", d_model=8, m=6, seed=0):
    torch.manual_seed(seed)
    return GraphTransformer(
        num_layers=2, pe_hidden_channels=8, pe_num_layers=2, d_model=d_model,
        heads=2, num_samples=m, dropout=0.0, k_pe=3, k_gt=2,
        directed=True, learn_r=True, pe_pool=pe_pool,
        fixed_seed_mode=True, fixed_seed_value=0, center_second_moment=True,
    ).to(DEVICE)


def _llm(hidden=32):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=_Tokenizer.MARKER + 1, hidden_size=hidden,
                      intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128,
                      attn_implementation="eager")
    return LlamaForCausalLM(cfg).to(DEVICE)


def _model(beta_init=0.0, pe_pool="gt", **kw):
    return MagCompGraphLLM(
        _llm(), _gt(pe_pool=pe_pool), tokenizer=_Tokenizer(), cycle_size=CYCLE,
        beta_init=beta_init, layer_scope="all", **kw).to(DEVICE)


def _composite(**kw):
    return build_composite_graph(
        _scene(), _ids()[0].tolist(), _injection_map(), scope_start=SCOPE_START,
        context_window=CYCLE, device=DEVICE,
        crosslink_mention_to_node=False, crosslink_bidirectional=True, **kw)


# --------------------------------------------------------------- the composite graph
def test_four_edge_classes_directedness_and_counts():
    """Cycle DIRECTED, scene UNDIRECTED, crosslinks scene->token ONLY, anchor = 2 edges."""
    g = _composite()
    n_scene, c = 4, CYCLE
    assert g.num_nodes == c + n_scene + 1          # + the anchor
    assert int(g.is_token.sum()) == c

    ei = {(int(u), int(v)) for u, v in g.edge_index.t()}
    mentions = {3: [2, 3, 4, 5], 0: [8, 9, 10], 1: [14, 15, 16]}   # window coordinates

    # 1. token cycle: i -> i+1 mod c present, the reverse absent (a symmetric cycle
    #    cancels sgn(A - A^T) and the rows stop carrying position).
    for i in range(c):
        assert (i, (i + 1) % c) in ei
        assert ((i + 1) % c, i) not in ei

    # 2. scene: both directions, shifted by +c.
    for u, v in ((0, 1), (1, 2), (2, 3)):
        assert (c + u, c + v) in ei and (c + v, c + u) in ei

    # 3. crosslinks: scene node -> token, arrowheads on the cycle, every sub-token.
    for j, toks in mentions.items():
        for t in toks:
            assert (c + j, t) in ei, f"missing crosslink scene {j} -> token {t}"
            assert (t, c + j) not in ei, f"crosslink {j} <- {t} must not exist"

    # 4. anchor: exactly BOS -> a and a -> scene node 0, never fanned.
    a = c + n_scene
    assert {(u, v) for u, v in ei if u == a or v == a} == {(0, a), (a, c)}

    # The class counts add up to the whole edge set, so nothing else was emitted.
    assert g.edge_index.shape[1] == c + 6 + sum(len(t) for t in mentions.values()) + 2


def test_anchor_can_be_ablated():
    g = _composite(anchor=False)
    assert g.num_nodes == CYCLE + 4
    assert g.edge_index.max() < CYCLE + 4


# ------------------------------------------------------------------------ C_tok
def test_c_tok_symmetric_psd():
    model = _model()
    g = _composite()
    with torch.no_grad():
        c_tok = model.covariance_token_block(g, CYCLE)
    assert c_tok.shape == (CYCLE, CYCLE)
    assert torch.allclose(c_tok, c_tok.T, atol=1e-5), "C_tok is not symmetric"
    lam = torch.linalg.eigvalsh(c_tok.double())
    assert lam.min() > -1e-8 * max(1.0, float(lam.max())), f"C_tok not PSD: {lam.min()}"
    assert float(c_tok.diagonal().sum()) > 0


def test_pe_pool_gt_changes_c_tok():
    """T inside E_q != T applied to Psi: the two poolings must give DIFFERENT C_tok.

    Both models share the R-PEARL seed and ``fixed_seed_mode``, so the probe draw is
    identical and the only difference is where the blocks run.
    """
    g = _composite()
    with torch.no_grad():
        gt_pool = _gt(pe_pool="gt", seed=3)
        pe_pool = _gt(pe_pool="pe", seed=3)
        pe_pool.load_state_dict(gt_pool.state_dict())   # same weights, different pooling
        c_gt, _ = gt_pool.pe_model.covariance_token_block(g, CYCLE, pe_pool="gt", gt=gt_pool)
        c_pe, _ = pe_pool.pe_model.covariance_token_block(g, CYCLE)
    assert c_gt.shape == c_pe.shape
    rel = (c_gt - c_pe).norm() / c_pe.norm().clamp_min(1e-12)
    assert rel > 1e-3, f"pe_pool='gt' left C_tok unchanged (rel diff {rel:.2e})"


def test_c_tok_is_chunk_invariant():
    """``max_probe_rows`` is a MEMORY dial, not a math one.

    The covariance is accumulated over gradient-checkpointed probe chunks; if the
    reduction were wrong (double-counted, mis-normalized, or centered per chunk rather
    than over all M) the chunk size would change the answer.
    """
    g = _composite()
    ref = None
    for max_probe_rows, expect_chunk in ((1 << 20, 6), (128, 4), (32, 1)):
        torch.manual_seed(11)
        gt = GraphTransformer(num_layers=2, pe_hidden_channels=8, pe_num_layers=2,
                              d_model=8, heads=2, num_samples=6, dropout=0.0, k_pe=3,
                              k_gt=2, directed=True, learn_r=True, pe_pool="gt",
                              fixed_seed_mode=True, fixed_seed_value=0,
                              max_probe_rows=max_probe_rows)
        assert max(1, min(6, max_probe_rows // g.num_nodes)) == expect_chunk
        with torch.no_grad():
            c_tok, _ = gt.pe_model.covariance_token_block(g, CYCLE, pe_pool="gt", gt=gt)
        if ref is None:
            ref = c_tok
        else:
            assert torch.allclose(c_tok, ref, atol=1e-5), (
                f"chunk={expect_chunk} changed C_tok by "
                f"{float((c_tok - ref).abs().max()):.2e}")


def test_gt_receives_gradient_through_the_chunked_reduction():
    """beta != 0 must reach the MagNet backbone through the checkpointed probe loop.

    Complements the beta=0 stall test: there the GT correctly gets nothing, so this is
    the only test that exercises the two-pass reduction's BACKWARD at all.
    """
    model = _model(beta_init=0.3)
    ids = _ids().to(DEVICE)
    out = model(input_ids=ids, labels=ids, graphs=[_scene()],
                injection_maps=[_injection_map()])
    out.loss.backward()
    r_logit = model.pe_model.pe_model.pe_gcn.convs[0].r_logit
    named = dict(model.pe_model.named_parameters())
    live = [n for n, p in named.items() if p.grad is not None and float(p.grad.abs().max()) > 0]
    assert live, "no GT parameter received gradient"
    assert any(n.startswith("pe_model.pe_gcn") for n in live), "MagNet backbone got nothing"
    assert any(n.startswith("blocks.") for n in live), "the GT blocks got nothing"
    assert r_logit.grad is not None and float(r_logit.grad.abs()) > 0, "the charge r is dead"


def test_pe_pool_and_gt_argument_must_agree():
    g = _composite()
    gt = _gt()
    with pytest.raises(ValueError, match="pe_pool"):
        gt.pe_model.covariance_token_block(g, CYCLE, pe_pool="gt")     # no gt handed in
    with pytest.raises(ValueError, match="pe_pool"):
        gt.pe_model.covariance_token_block(g, CYCLE, pe_pool="pe", gt=gt)


# ------------------------------------------------------------------------- the bias
def test_beta_zero_reproduces_base_logits_exactly():
    """beta = 0 => the bias is identically 0 => the base LLM, bit for bit."""
    model = _model(beta_init=0.0)
    ids = _ids().to(DEVICE)
    kw = dict(graphs=[_scene()], injection_maps=[_injection_map()])
    with torch.no_grad():
        armed = model(input_ids=ids, **kw).logits
        assert model._struct_bias is None            # disarmed in the finally
        base = model.llm(input_ids=ids).logits
    assert torch.equal(armed, base), (
        f"beta=0 changed the logits (max |delta| = {(armed - base).abs().max():.3e})")


def test_bias_is_zero_outside_the_scope_and_c_tok_inside():
    """The system prompt gets NO bias; the scope block IS beta * C_tok; nothing is -inf.

    ``decode_refresh >= c`` puts the whole scope in ONE segment, which is the only regime
    in which the block equals a single C_tok; the autoregressive schedule is the subject
    of ``test_training_bias_grows_autoregressively``.
    """
    model = _model(beta_init=0.5, decode_refresh=CYCLE)
    ids = _ids().to(DEVICE)
    with torch.no_grad():
        bias = model.build_structural_mask(
            ids.shape[1], [_scene()], [_injection_map()], DEVICE,
            dtype=torch.float32, input_ids=ids)
        c_tok = model.covariance_token_block(
            model.composite_graph(_scene(), ids[0].tolist(), _injection_map(),
                                  SCOPE_START, DEVICE), CYCLE)
    assert torch.isfinite(bias).all(), "this arm never hard-blocks"
    assert torch.equal(bias[0, 0, :SCOPE_START, :], torch.zeros(SCOPE_START, ids.shape[1]))
    assert torch.equal(bias[0, 0, :, :SCOPE_START], torch.zeros(ids.shape[1], SCOPE_START))
    assert torch.allclose(bias[0, 0, SCOPE_START:, SCOPE_START:], 0.5 * c_tok, atol=1e-6)


def test_beta_has_gradient_at_zero_and_c_does_not():
    """The documented one-step stall: dL/dbeta != 0 at beta = 0, dL/dC = 0."""
    model = _model(beta_init=0.0)
    ids = _ids().to(DEVICE)
    out = model(input_ids=ids, labels=ids, graphs=[_scene()],
                injection_maps=[_injection_map()])
    out.loss.backward()
    assert model.beta.grad is not None and float(model.beta.grad.abs()) > 0, (
        "beta has no gradient at 0 — the graph channel could never open")
    gt_grads = [p.grad for p in model.pe_model.parameters() if p.grad is not None]
    assert all(float(g.abs().max()) == 0.0 for g in gt_grads), (
        "the GT received gradient at beta=0; dL/dC = beta * dL/dbias should be exactly 0")


# ------------------------------------------------------------------- telemetry
def test_telemetry_reports_beta_and_the_realised_bias_scale():
    """beta's VALUE is the headline; bias_absmax is what says whether it is calibrated."""
    model = _model(beta_init=0.4)
    ids = _ids().to(DEVICE)
    assert model.telemetry() == {}, "no graph armed yet -> nothing to report"
    with torch.no_grad():
        model(input_ids=ids, graphs=[_scene()], injection_maps=[_injection_map()])
    t = model.telemetry()
    assert t["mag/beta"] == pytest.approx(0.4)
    assert t["mag/c_tok_absmax"] > 0
    assert t["mag/bias_absmax"] == pytest.approx(0.4 * t["mag/c_tok_absmax"])
    assert t["mag/cycle_c"] == CYCLE                      # the REALISED c, not the cap
    assert t["mag/scope_start"] == SCOPE_START
    assert t["mag/num_nodes"] == CYCLE + 4 + 1
    assert t["debug/pe_has_nan"] == 0                     # measured, not defaulted
    assert t["debug/pe_output_norm"] == pytest.approx(t["mag/c_tok_fro"])
    assert t["mag/c_tok_fro"] > 0, "a zero PE norm would mean nothing was measured"


def test_telemetry_beta_zero_still_reports_a_live_c_tok():
    """At the cold start the BIAS is zero but C_tok is not — the two must be separable.

    If both read zero there is no way to tell "beta has not opened yet" from "the probe
    covariance is dead", which are different failures with different fixes.
    """
    model = _model(beta_init=0.0)
    ids = _ids().to(DEVICE)
    with torch.no_grad():
        model(input_ids=ids, graphs=[_scene()], injection_maps=[_injection_map()])
    t = model.telemetry()
    assert t["mag/beta"] == 0.0
    assert t["mag/bias_absmax"] == 0.0
    assert t["mag/c_tok_absmax"] > 0


def test_gradient_callback_captures_beta_backbone_and_charge():
    """The callback must split beta / backbone / r_logit out of the aggregates.

    beta is not in pe_model.parameters(), and the MagNet backbone is buried inside the
    R-PEARL aggregate — each can go to zero while the aggregate above it looks healthy.
    """
    from prism.eval.callbacks import GradientDebugCallback

    model = _model(beta_init=0.3)
    ids = _ids().to(DEVICE)
    out = model(input_ids=ids, labels=ids, graphs=[_scene()],
                injection_maps=[_injection_map()])
    out.loss.backward()

    cb = GradientDebugCallback()
    cb._capture_grad_norms(model)
    g = cb._captured_grad_norms
    for key in ("beta", "gnn", "gt_blocks", "rpearl", "backbone", "r_logit"):
        assert key in g, f"{key} grad norm not captured"
        assert g[key] > 0, f"{key} grad norm is zero"
    # The splits must be strictly inside their aggregates, else they are mislabelled.
    assert g["backbone"] <= g["rpearl"] + 1e-6
    assert g["r_logit"] <= g["backbone"] + 1e-6


def test_charge_callback_uses_the_realised_c_not_the_cap():
    """delta = dist(2rc, Z) is meaningless at the wrong c.

    ``mask_cycle_size`` is a CAP; the graph is built over the prompt length past the
    block. A callback that used the cap would report the margin of a graph nobody built.
    """
    from prism.eval.callbacks import ChargeDegeneracyCallback

    model = _model()
    ids = _ids().to(DEVICE)
    with torch.no_grad():
        model(input_ids=ids, graphs=[_scene()], injection_maps=[_injection_map()])
    cb = ChargeDegeneracyCallback(cycle_length=8192)       # the config cap
    assert cb._live_c(model) == CYCLE                      # the realised cycle
    assert cb._live_c(None) == 8192                        # falls back to the cap
    r = cb._read_charge(model.pe_model.pe_model.pe_gcn)
    assert cb._delta(r, CYCLE) != cb._delta(r, 8192)


# ---------------------------------------------------------------- the preconditions
@pytest.mark.parametrize("kw, match", [
    (dict(directed=False, learn_r=True, pe_pool="gt"), "DIRECTED"),
    (dict(directed=True, learn_r=False, pe_pool="gt"), "LEARNABLE charge"),
    (dict(directed=True, learn_r=True, pe_pool="pe"), "pe_pool"),
])
def test_producer_preconditions_fail_loud(kw, match):
    torch.manual_seed(0)
    gt = GraphTransformer(num_layers=2, pe_hidden_channels=8, pe_num_layers=2, d_model=8,
                          heads=2, num_samples=4, dropout=0.0, k_pe=3, k_gt=2, **kw)
    with pytest.raises(ValueError, match=match):
        MagCompGraphLLM(_llm(), gt, tokenizer=_Tokenizer(), cycle_size=CYCLE)


# --------------------------------------------------- the pretrained-MagE-GT channel
def test_pretrained_pinned_charge_maggt_loads_and_r_cold_starts(tmp_path):
    """``gnn.pe_gt_from`` accepts the notebook's PINNED-charge MagE-GT.

    The resistance-regression stage trains with ``learn_r=False`` so its target is a fixed
    function of the topology; its state dict therefore has no ``r_logit``. The loader must
    tolerate exactly that gap (and nothing else), land every other tensor, and leave r at
    THIS run's ``mask_magnet_r`` init.
    """
    from prism.models.loaders import load_navigator_pe_into

    torch.manual_seed(7)
    source = GraphTransformer(num_layers=2, pe_hidden_channels=8, pe_num_layers=2,
                              d_model=8, heads=2, num_samples=4, dropout=0.0, k_pe=3,
                              k_gt=2, directed=True, learn_r=False, pe_pool="gt")
    src_state = source.state_dict()
    assert not any(k.endswith(".r_logit") for k in src_state), "source must pin the charge"
    path = tmp_path / "mag_gt.pt"
    torch.save(src_state, path)

    model = _model(pe_pool="gt")                       # learn_r=True, r = 0.126 default
    r_before = float(model.pe_model.pe_model.pe_gcn.r.detach())
    load_navigator_pe_into(model, str(path), None)

    loaded = model.pe_model.state_dict()
    for k, v in src_state.items():
        assert torch.equal(loaded[k], v), f"{k} did not land"
    assert float(model.pe_model.pe_model.pe_gcn.r.detach()) == pytest.approx(r_before)
    assert model.pe_model.pe_model.pe_gcn.convs[0].r_logit.requires_grad


def test_pretrained_load_still_rejects_a_real_mismatch(tmp_path):
    """Only ``*.r_logit`` is forgiven — a genuine topology mismatch must still raise."""
    from prism.models.loaders import load_navigator_pe_into

    torch.manual_seed(7)
    wrong = GraphTransformer(num_layers=1, pe_hidden_channels=8, pe_num_layers=2,
                             d_model=8, heads=2, num_samples=4, dropout=0.0, k_pe=3,
                             k_gt=2, directed=True, learn_r=False, pe_pool="gt")
    path = tmp_path / "wrong.pt"
    torch.save(wrong.state_dict(), path)
    with pytest.raises(RuntimeError, match="did not match"):
        load_navigator_pe_into(_model(), str(path), None)


def test_multistage_carry_restores_beta(tmp_path):
    """Stage 2 -> Stage 3 (``trainer.init_pe_from``) must carry beta, not reset it.

    beta is the channel's entire gain and is NOT in ``pe_model``, so a carry that moved Ψ
    alone would restart Stage 3 at ``mask_beta_init`` (0.0 = the base LLM) and discard the
    whole of Stage 2 with a clean load and no warning.
    """
    from prism.models.loaders import load_pe_weights_into
    from prism.training.run_dir import save_run_dir

    stage2 = _model(beta_init=0.0)
    with torch.no_grad():
        stage2.beta.fill_(0.42)                      # as if Stage 2 trained it here
    save_run_dir(stage2, {"architecture": "learnable_graph_mask", "mask_composite": True},
                 str(tmp_path))

    stage3 = _model(beta_init=0.0)                   # cold start, as the config says
    assert float(stage3.beta) == 0.0
    load_pe_weights_into(stage3, str(tmp_path), "learnable_graph_mask")
    assert float(stage3.beta.detach()) == pytest.approx(0.42)
    for k, v in stage2.pe_model.state_dict().items():
        assert torch.equal(stage3.pe_model.state_dict()[k], v), f"{k} did not carry"


def test_multistage_carry_without_beta_fails_loud(tmp_path):
    """A source run with no mask_beta must raise, not silently reset the channel."""
    from prism.models.loaders import load_pe_weights_into

    model = _model(beta_init=0.3)
    torch.save({"pe_model": model.pe_model.state_dict()}, tmp_path / "gnn_weights.pt")
    with pytest.raises(KeyError, match="mask_beta"):
        load_pe_weights_into(_model(), str(tmp_path), "learnable_graph_mask")


def test_save_run_dir_carries_beta(tmp_path):
    """beta is the whole channel's gain; a run dir without it reloads as the base LLM."""
    from prism.training.run_dir import save_run_dir

    model = _model(beta_init=0.25)
    save_run_dir(model, {"architecture": "learnable_graph_mask", "mask_composite": True},
                 str(tmp_path))
    weights = torch.load(tmp_path / "gnn_weights.pt", map_location="cpu")
    assert "pe_model" in weights
    assert float(weights["mask_beta"]) == pytest.approx(0.25)


def test_left_padding_is_rejected():
    model = _model()
    ids = _ids().to(DEVICE)
    mask = torch.ones_like(ids)
    mask[0, 0] = 0                                   # left pad
    with pytest.raises(ValueError, match="RIGHT padding"):
        model.build_structural_mask(ids.shape[1], [_scene()], [_injection_map()], DEVICE,
                                    input_ids=ids, attention_mask=mask)


# ------------------------------------------------- decode-time composite growth
def _node_seqs():
    """Token-id sequences for the four node names, matching ``_injection_map``'s spans.

    ``_Tokenizer`` decodes id t to "<t>", so a node "name" is just the id run its
    mentions occupy: Hankee = 6..9, Park = 12..14, Office = 18..20 (ids are 1-based).
    """
    return [[13, 14, 15], [19, 20, 21], [77, 78], [7, 8, 9, 10]]


def test_decode_injector_grows_the_cycle_and_crosslinks():
    """Each refresh rebuilds over prompt + suffix: the cycle gains the generated tokens
    and E_Cross gains the mentions they completed."""
    model = _model(beta_init=1.0)
    prompt = _ids(SCOPE_START + 8)[0].tolist()          # c = 8 at prefill
    inj = CompositeDecodeInjector(model, _scene(), prompt, refresh=1)

    inj.generated = [77, 78]                            # a complete "House" mention
    inj._rebuild()
    c_grown = inj._c_tok.shape[0]
    assert c_grown == len(prompt) + 2 - SCOPE_START, (
        f"the cycle must cover the generated tokens: {c_grown}")

    # The generated mention reaches E_Cross: rebuild the same graph and read its edges.
    ids = prompt + inj.generated
    tau, end = model.scope_span(ids)
    imap = build_injection_map(ids, _node_seqs(), scope_start=tau)
    g = model.composite_graph(_scene(), ids, imap, tau, DEVICE)
    ei = {(int(u), int(v)) for u, v in g.edge_index.t().tolist()}
    c = end - tau
    gen_rows = [p - tau for spans in imap.values() for s, e in spans
                for p in range(s, e) if p >= len(prompt)]
    assert gen_rows, "the generated tokens completed no mention — fixture is wrong"
    assert all(any((c + j, p) in ei for j in range(4)) for p in gen_rows), (
        "a generated token's mention must appear as a scene -> token crosslink")


def test_decode_injector_bias_row_matches_c_tok_and_scope():
    """The armed row is beta * C_tok[q, :] over [tau, tau + c) and zero outside it."""
    model = _model(beta_init=0.7)
    prompt = _ids(SCOPE_START + 8)[0].tolist()
    inj = CompositeDecodeInjector(model, _scene(), prompt, refresh=1)
    inj.pre_hook(None, (), {"input_ids": torch.tensor([[77]])})

    row = model._decode_bias_row
    assert row is not None and row.shape == (1, 1, 1, len(prompt) + 1)
    flat = row[0, 0, 0]
    assert torch.count_nonzero(flat[:SCOPE_START]) == 0, "the system prompt got a bias"
    q = len(prompt) + 1 - 1 - SCOPE_START
    expected = float(model.beta) * inj._c_tok[q, : flat.shape[0] - SCOPE_START]
    assert torch.allclose(flat[SCOPE_START:], expected.float(), atol=1e-5)


def test_decode_injector_refresh_cadence():
    """refresh=k rebuilds once per k tokens, not once per token."""
    model = _model(beta_init=1.0)
    prompt = _ids(SCOPE_START + 8)[0].tolist()
    inj = CompositeDecodeInjector(model, _scene(), prompt, refresh=4)
    calls = {"n": 0}
    real = inj._rebuild

    def counted():
        calls["n"] += 1
        real()

    inj._rebuild = counted
    for t in range(9):
        inj.pre_hook(None, (), {"input_ids": torch.tensor([[77]])})
    assert calls["n"] == 3, f"expected ceil(9/4) rebuilds, got {calls['n']}"


def test_half_written_label_is_not_a_mention():
    """`house` must not crosslink while `house_1` is still being written.

    Node 2 is named by tokens [77, 78] and node 1 by [77] alone, so the suffix `[77]`
    matches node 1 exactly AND is a strict prefix of node 2's name. Until the 78 lands,
    neither is a mention; once it does, node 2 is.
    """
    seqs = [[13, 14, 15], [[77]], [[77, 78]], [90, 91, 92, 93]]
    prompt = _ids(SCOPE_START + 8)[0].tolist()

    open_ids = prompt + [77]
    raw = build_injection_map(open_ids, seqs, scope_start=SCOPE_START)
    assert 1 in raw, "fixture is wrong: [77] must match node 1 outright"
    assert defer_open_mentions(raw, seqs, open_ids) == {}, (
        "a label the sequence is mid-way through writing must not be a mention")

    closed_ids = prompt + [77, 78]
    done = defer_open_mentions(
        build_injection_map(closed_ids, seqs, scope_start=SCOPE_START), seqs, closed_ids)
    assert done == {2: [(len(prompt), len(prompt) + 2)]}, (
        f"completed label must commit: {done}")


def test_decode_injector_defers_the_half_written_label():
    """The composite the injector builds gains no crosslink for a partial label."""
    model = _model(beta_init=1.0)
    seqs = [[13, 14, 15], [[77]], [[77, 78]], [90, 91, 92, 93]]
    prompt = _ids(SCOPE_START + 8)[0].tolist()

    inj = CompositeDecodeInjector(model, _scene(), prompt, refresh=1)
    inj.pre_hook(None, (), {"input_ids": torch.tensor([[77]])})
    ids = prompt + [77]
    tau, end = model.scope_span(ids)
    c = end - tau
    g = model.composite_graph(
        _scene(), ids,
        defer_open_mentions(build_injection_map(ids, seqs, scope_start=tau), seqs, ids),
        tau, DEVICE)
    ei = {(int(u), int(v)) for u, v in g.edge_index.t().tolist()}
    partial_row = len(ids) - 1 - tau
    assert not any((c + j, partial_row) in ei for j in range(4)), (
        "the half-written label got scene-graph out-edges")

    inj.pre_hook(None, (), {"input_ids": torch.tensor([[78]])})
    ids2 = prompt + [77, 78]
    tau2, end2 = model.scope_span(ids2)
    g2 = model.composite_graph(
        _scene(), ids2,
        defer_open_mentions(build_injection_map(ids2, seqs, scope_start=tau2), seqs, ids2),
        tau2, DEVICE)
    ei2 = {(int(u), int(v)) for u, v in g2.edge_index.t().tolist()}
    c2 = end2 - tau2
    rows = [len(prompt) - tau2, len(prompt) + 1 - tau2]
    assert all((c2 + 2, r) in ei2 for r in rows), (
        "the completed label must crosslink from its OWN node (2), on every sub-token")


def test_decode_and_training_build_the_same_composite():
    """The requirement, asserted: for one sequence, the graph decode converges to is the
    graph training builds — same nodes, same edges, same weights.

    Training sees prompt + answer at once; decode reaches the same text one token at a
    time. Both go through ``MagCompGraphLLM.composite_graph``, so the injection map is
    closed by the same rule and the edge families are laid down by the same call.
    """
    model = _model(beta_init=1.0)
    # The model's OWN variants, since it derives them itself now; the answer completes a
    # House ([77, 78]) and a Park ([13, 14, 15]) mention so E_Cross grows with the cycle.
    seqs = model._node_seqs(_scene())
    prompt = _ids(SCOPE_START + 8)[0].tolist()
    answer = [77, 78, 50, 13, 14, 15]

    # Training: the whole sequence at once, exactly as the collator hands it over.
    full = prompt + answer
    tau, end = model.scope_span(full)
    trained = model.composite_graph(
        _scene(), full, build_injection_map(full, seqs, scope_start=tau), tau, DEVICE)

    # Decode: the same text, one token at a time, rebuilt every step.
    inj = CompositeDecodeInjector(model, _scene(), prompt, refresh=1)
    for t in answer:
        inj.pre_hook(None, (), {"input_ids": torch.tensor([[t]])})
    ids = prompt + inj.generated
    decoded = model.composite_graph(
        _scene(), ids, build_injection_map(ids, seqs, scope_start=tau), tau, DEVICE)

    assert decoded.num_nodes == trained.num_nodes, (
        f"node count diverged: {decoded.num_nodes} vs {trained.num_nodes}")
    assert decoded.num_token_nodes == trained.num_token_nodes
    e_dec = {(int(u), int(v), round(float(w), 6)) for (u, v), w in
             zip(decoded.edge_index.t().tolist(), decoded.edge_weight.tolist())}
    e_tra = {(int(u), int(v), round(float(w), 6)) for (u, v), w in
             zip(trained.edge_index.t().tolist(), trained.edge_weight.tolist())}
    assert e_dec == e_tra, (
        f"edges diverged: decode-only {sorted(e_dec - e_tra)[:5]}, "
        f"train-only {sorted(e_tra - e_dec)[:5]}")


def test_training_bias_grows_autoregressively():
    """A query's bias must not depend on a token after it.

    Built once over the whole sequence, ``C_tok[p, p']`` is a function of everything —
    including the answer's own continuation, which decode cannot see. With the schedule,
    row ``p`` comes from the composite over ``ids[:p_ref]``, so truncating the sequence
    after ``p_ref`` leaves that row unchanged.
    """
    model = _model(beta_init=1.0, decode_refresh=1)
    ids = _ids().to(DEVICE)
    answer = SCOPE_START + 4                    # everything past here grows per token
    cut = SCOPE_START + 8
    with torch.no_grad():
        full = model.build_structural_mask(
            ids.shape[1], [_scene()], [_injection_map()], DEVICE,
            dtype=torch.float32, input_ids=ids, answer_starts=[answer])
        short = model.build_structural_mask(
            cut, [_scene()], [_injection_map()], DEVICE,
            dtype=torch.float32, input_ids=ids[:, :cut], answer_starts=[answer])
    assert torch.allclose(full[0, 0, :cut, :cut], short[0, 0], atol=1e-5), (
        "a query's bias changed when LATER tokens were removed — the graph is not "
        "growing autoregressively and the future is leaking into the past")


def test_training_and_decode_bias_agree_position_by_position():
    """The requirement: the bias training gives query p equals the row decode arms at p."""
    model = _model(beta_init=1.0, decode_refresh=1)
    ids = _ids().to(DEVICE)
    prompt_len = SCOPE_START + 6
    prompt = ids[0, :prompt_len].tolist()
    # BOTH sides derive the map from the same variants: decode has no other source, and
    # handing training a different one would compare two different graphs.
    seqs = _node_seqs()
    imap = build_injection_map(ids[0].tolist(), seqs, scope_start=SCOPE_START)

    with torch.no_grad():
        trained = model.build_structural_mask(
            ids.shape[1], [_scene()], [imap], DEVICE,
            dtype=torch.float32, input_ids=ids, answer_starts=[prompt_len])
        inj = CompositeDecodeInjector(model, _scene(), prompt, refresh=1)
        for t in ids[0, prompt_len:].tolist():
            inj.pre_hook(None, (), {"input_ids": torch.tensor([[t]])})
            p = prompt_len + len(inj.generated) - 1
            row = model._decode_bias_row
            assert row is not None, f"no bias armed at position {p}"
            k = row.shape[-1]
            assert torch.allclose(row[0, 0, 0], trained[0, 0, p, :k], atol=1e-5), (
                f"training and decode disagree at position {p}")


def test_frozen_graph_ignores_the_answer_entirely():
    """``decode_refresh=0`` pins every segment to the prompt.

    The bias must then be a function of ``ids[:answer_start]`` alone: changing the answer
    tokens cannot move a single entry, and answer QUERIES carry no bias at all, because a
    prompt-only composite has no token node for them. That is exactly what decode does
    past its window, which is what makes the two identical.
    """
    model = _model(beta_init=1.0, decode_refresh=0)
    ids = _ids().to(DEVICE)
    answer = SCOPE_START + 4
    other = ids.clone()
    other[0, answer:] = 11                      # rewrite the whole answer
    with torch.no_grad():
        a = model.build_structural_mask(
            ids.shape[1], [_scene()], [_injection_map()], DEVICE,
            dtype=torch.float32, input_ids=ids, answer_starts=[answer])
        b = model.build_structural_mask(
            other.shape[1], [_scene()], [_injection_map()], DEVICE,
            dtype=torch.float32, input_ids=other, answer_starts=[answer])
    assert torch.allclose(a, b, atol=1e-5), (
        "frozen bias changed when the answer changed — the graph is still growing")
    assert a[0, 0, answer:, :].abs().max() == 0, (
        "answer queries carry bias under a prompt-only composite, which has no node "
        "for them; decode cannot reproduce that")


def test_frozen_decode_never_rebuilds_and_uses_the_prompt_alone():
    """The injector at ``refresh=0`` builds once, from the prompt, and stops."""
    model = _model(beta_init=1.0, decode_refresh=0)
    prompt = _ids()[0].tolist()
    inj = CompositeDecodeInjector(model, _scene(), prompt, refresh=0)
    calls = []
    real = inj._rebuild

    def counting():
        calls.append(list(inj.generated))
        return real()

    inj._rebuild = counting
    for tok in (11, 12, 13, 14, 15):
        inj.pre_hook(None, (), {"input_ids": torch.tensor([[tok]], device=DEVICE)})
    assert len(calls) == 1, f"frozen decode rebuilt {len(calls)} times, expected exactly 1"
    assert inj._c_tok.shape[0] == model.scope_span(prompt)[1] - model.scope_span(prompt)[0], (
        "frozen C_tok is not the prompt-scope size — the suffix leaked into the graph")


def test_training_and_decode_agree_at_refresh_above_one():
    """Floor rounding: training and decode must agree at EVERY position, at R > 1.

    Training rounds DOWN to the last refresh boundary, as decode does, so a training query
    can no longer see its own future. Between boundaries decode has no node for the query
    and arms no row; training must likewise leave those queries at zero.
    """
    R = 4
    model = _model(beta_init=1.0, decode_refresh=R)
    ids = _ids().to(DEVICE)
    prompt_len = SCOPE_START + 6
    prompt = ids[0, :prompt_len].tolist()
    seqs = _node_seqs()
    imap = build_injection_map(ids[0].tolist(), seqs, scope_start=SCOPE_START)

    with torch.no_grad():
        trained = model.build_structural_mask(
            ids.shape[1], [_scene()], [imap], DEVICE,
            dtype=torch.float32, input_ids=ids, answer_starts=[prompt_len])
        inj = CompositeDecodeInjector(model, _scene(), prompt, refresh=R)
        for t in ids[0, prompt_len:].tolist():
            inj.pre_hook(None, (), {"input_ids": torch.tensor([[t]])})
            p = prompt_len + len(inj.generated) - 1
            row = model._decode_bias_row
            if row is None:                      # decode armed nothing between rebuilds
                assert trained[0, 0, p, :].abs().max() == 0, (
                    f"training biased query {p} where decode has no row at all")
            else:
                k = row.shape[-1]
                assert torch.allclose(row[0, 0, 0], trained[0, 0, p, :k], atol=1e-5), (
                    f"training and decode disagree at position {p} with R={R}")


def test_no_future_leak_at_refresh_above_one():
    """At R > 1 a query's bias must not move when LATER tokens are removed.

    TRUNCATION, not rewriting: shortening the sequence changes ``c`` and so changes
    ``C_tok`` outright. Rewriting tokens to an id that matches no node name leaves the
    graph topology identical, so it would pass whatever the rounding does.
    """
    model = _model(beta_init=1.0, decode_refresh=4)
    ids = _ids().to(DEVICE)
    answer = SCOPE_START + 6
    cut = answer + 5                             # mid-segment, so round-up would overshoot
    with torch.no_grad():
        full = model.build_structural_mask(
            ids.shape[1], [_scene()], [_injection_map()], DEVICE,
            dtype=torch.float32, input_ids=ids, answer_starts=[answer])
        short = model.build_structural_mask(
            cut, [_scene()], [_injection_map()], DEVICE,
            dtype=torch.float32, input_ids=ids[:, :cut], answer_starts=[answer])
    assert torch.allclose(full[0, 0, :cut, :cut], short[0, 0], atol=1e-5), (
        "a query's bias changed when LATER tokens were removed — training is still "
        "rounding up and reading its own future")


def test_isolated_unmentioned_scene_nodes_are_pruned():
    """A scene node with no edge and no mention must not enter the composite.

    Nothing of any edge class touches it, so it would contribute a pure-noise row to C and
    split G into components across which R_eff is infinite. A scene-isolated node that IS
    mentioned stays: MagNet symmetrizes A, so its crosslink carries flow both ways.
    """
    # nodes 0-1 joined; node 2 isolated BUT mentioned; node 3 isolated and unmentioned
    scene = Data(x=torch.zeros(4, 1),
                 edge_index=torch.tensor([[0, 1], [1, 0]]), num_nodes=4)
    g = build_composite_graph(
        scene, input_ids=list(range(10)), scope_start=0,
        injection_map={0: [(2, 3)], 2: [(6, 7)]}, context_window=10,
        crosslink_mention_to_node=False, crosslink_bidirectional=True, device='cpu')
    c = g.num_token_nodes
    assert g.num_scene_nodes == 3, (
        f"expected node 3 pruned (3 scene nodes), got {g.num_scene_nodes}")
    assert g.num_nodes == c + 3 + 1

    # and the result is CONNECTED, which is the whole point
    import networkx as nx
    from torch_geometric.utils import to_networkx
    plain = Data(edge_index=g.edge_index, num_nodes=g.num_nodes)
    assert nx.is_connected(to_networkx(plain, to_undirected=True)), (
        "composite still disconnected after pruning")

    # the surviving mentioned-but-isolated node kept its crosslink (remapped 2 -> 2)
    es = {(int(u), int(v)) for u, v in g.edge_index.t().tolist()}
    assert any(u >= c and v == 6 for u, v in es), "the mentioned isolated node lost its crosslink"


def test_prune_is_a_noop_when_every_scene_node_has_a_neighbour():
    """The n_30 case: nothing is dropped, so no run changes behaviour."""
    scene = Data(x=torch.zeros(4, 1),
                 edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
                 num_nodes=4)
    g = build_composite_graph(
        scene, input_ids=list(range(10)), scope_start=0,
        injection_map={0: [(2, 4)]}, context_window=10,
        crosslink_mention_to_node=False, crosslink_bidirectional=True, device='cpu')
    assert g.num_scene_nodes == 4, "a connected scene graph must be left untouched"
