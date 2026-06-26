"""Fresh-pass verification of R-PEARL (src/prism/models/r_pearl.py) and the Graph
Transformer (src/prism/models/gt.py) with fully initialized, UNTRAINED (raw) models.

Two goals (DL-mode verifier):
  1. Full contract battery on the raw modules — forward/backward shapes & dtypes,
     finiteness, determinism, the learnable output gate, the second-moment / covariance
     readouts, the deterministic (semantic-feature) path, and the sparse-attention
     numerics (independent dense oracle).
  2. The `m_test` removal: probe count is now a SINGLE `self.M = num_samples` used for
     BOTH train and eval. We pin that the `m_test` constructor arg is gone, the
     `m_train`/`m_test` attributes are gone, and train()/eval() now request the SAME
     probe count — and that no `m_test`/`m_train` token survives in src/prism/models/
     or experiments/.

All CPU, tiny, fp32, seeded. The oracles are derived from the docstrings/spec, not by
copying the implementation. No LLM is needed: these are standalone graph modules.
"""

import os
import re
import sys
import warnings

sys.path.insert(0, "src")

import torch
from torch_geometric.data import Data

from prism.models import r_pearl as rp
from prism.models import gt as gtmod


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _graph(n=6, edges=None):
    """Small undirected connected PyG graph with unit node features ([N,1] placeholder)."""
    if edges is None:
        # ring 0-1-2-3-4-5-0
        e = [(i, (i + 1) % n) for i in range(n)]
        e += [(b, a) for a, b in e]  # undirected
    else:
        e = edges
    edge_index = torch.tensor(e, dtype=torch.long).t().contiguous()
    x = torch.ones(n, 1, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, num_nodes=n)


def _rpearl(**over):
    kw = dict(
        pe_hidden_channels=8, pe_num_layers=2, d_model=16,
        num_samples=6, dropout=0.0, k=2,  # pe_num_layers*k = 4 >= 3 (no warning)
    )
    kw.update(over)
    torch.manual_seed(0)
    return rp.RandomGNNPositionalEncodings(**kw)


def _gt(**over):
    kw = dict(
        num_layers=2, pe_hidden_channels=8, pe_num_layers=2, d_model=16,
        heads=4, num_samples=6, dropout=0.0, k_pe=2, k_gt=2,
    )
    kw.update(over)
    torch.manual_seed(0)
    return gtmod.GraphTransformer(**kw)


# --------------------------------------------------------------------------- #
# R-PEARL: random-probe forward
# --------------------------------------------------------------------------- #
def test_rpearl_forward_shape_finite_dtype():
    """forward returns [N, d_model] fp32, all finite."""
    m = _rpearl().eval()
    out = m(_graph(n=6))
    assert out.shape == (6, 16), out.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_rpearl_fixed_seed_determinism_and_stochasticity():
    """fixed_seed_mode=True ⇒ identical Ψ across calls; False ⇒ probes resample ⇒ differ."""
    det = _rpearl(fixed_seed_mode=True, fixed_seed_value=7).eval()
    a, b = det(_graph()), det(_graph())
    assert torch.allclose(a, b), "fixed_seed_mode must give identical output"

    sto = _rpearl(fixed_seed_mode=False).eval()
    c, d = sto(_graph()), sto(_graph())
    assert not torch.allclose(c, d), "resampled probes must change the output"


def test_rpearl_grad_flow():
    """backward populates finite, not-all-zero grads on GCN, projection, gate, norm."""
    m = _rpearl().train()
    out = m(_graph())
    out.sum().backward()
    checked = []
    for name in ["output_gain", "output_projection.weight", "norm.weight"]:
        p = dict(m.named_parameters())[name]
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        checked.append(p.grad.abs().sum().item())
    # at least one GCN conv weight gets signal
    gcn_grads = [p.grad for n, p in m.named_parameters()
                 if "pe_gcn" in n and p.grad is not None]
    assert any(g.abs().sum() > 0 for g in gcn_grads), "no grad reached the GCN"
    assert any(v > 0 for v in checked)


def test_rpearl_output_gate_zero():
    """output_gain=0 ⇒ tanh(0)=0 ⇒ Ψ is exactly zero (gate semantics)."""
    m = _rpearl(fixed_seed_mode=True).eval()
    with torch.no_grad():
        m.output_gain.fill_(0.0)
    out = m(_graph())
    assert torch.allclose(out, torch.zeros_like(out)), out.abs().max().item()


def test_rpearl_single_node_no_edges():
    """Degenerate graph (1 node, 0 edges) ⇒ finite [1, d_model], no NaN/Inf."""
    g = Data(x=torch.ones(1, 1), edge_index=torch.empty(2, 0, dtype=torch.long), num_nodes=1)
    out = _rpearl().eval()(g)
    assert out.shape == (1, 16)
    assert torch.isfinite(out).all()


def test_rpearl_rademacher_probes():
    """rademacher probe distribution runs and produces finite output."""
    out = _rpearl(probe_distribution="rademacher").eval()(_graph())
    assert torch.isfinite(out).all() and out.shape == (6, 16)


def test_rpearl_bad_probe_distribution_raises():
    """Invalid probe_distribution must fail loud (ValueError), not silently default."""
    try:
        _rpearl(probe_distribution="cauchy")
    except ValueError:
        return
    assert False, "expected ValueError for unknown probe_distribution"


def test_rpearl_multihop_warning():
    """pe_num_layers*k < 3 emits the limited-multi-hop warning (soft guard)."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _rpearl(pe_num_layers=2, k=1)  # 2*1 = 2 < 3
    assert any("multi-hop" in str(x.message) for x in w), [str(x.message) for x in w]


# --------------------------------------------------------------------------- #
# R-PEARL: second-moment / covariance readouts
# --------------------------------------------------------------------------- #
def test_second_moment_apply_shape_and_gate():
    """C·s is [N, d_model] finite; scale_to_signal toggles ONLY the tanh(g) gate.

    Oracle: with identical probes (fixed seed), gated == ungated * tanh(output_gain).
    """
    m = _rpearl(fixed_seed_mode=True, fixed_seed_value=3).eval()
    g = _graph()
    s = torch.randn(6, 16)
    gated = m.second_moment_apply(g, s, scale_to_signal=True)
    ungated = m.second_moment_apply(g, s, scale_to_signal=False)
    assert gated.shape == (6, 16) and torch.isfinite(gated).all()
    factor = torch.tanh(m.output_gain).item()
    assert torch.allclose(gated, ungated * factor, atol=1e-5), \
        (gated - ungated * factor).abs().max().item()


def test_second_moment_centering_changes_result():
    """center_second_moment=True subtracts the ΨΨᵀ rank-1 bias ⇒ differs from uncentered."""
    g = _graph()
    s = torch.randn(6, 16)
    cen = _rpearl(fixed_seed_mode=True, fixed_seed_value=5,
                  center_second_moment=True).eval()
    unc = _rpearl(fixed_seed_mode=True, fixed_seed_value=5,
                  center_second_moment=False).eval()
    # same init (manual_seed(0) in factory) + same fixed probes ⇒ only the centering differs
    a = cen.second_moment_apply(g, s, scale_to_signal=False)
    b = unc.second_moment_apply(g, s, scale_to_signal=False)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert not torch.allclose(a, b), "centering must change C·s"


def test_covariance_token_block_psd_symmetric():
    """covariance_token_block returns (C_tok, Psi_tok), both [c,c], symmetric & PSD."""
    m = _rpearl(num_samples=12, fixed_seed_mode=True).eval()
    c = 4
    C_tok, Psi_tok = m.covariance_token_block(_graph(n=6), c)
    assert C_tok.shape == (c, c) and Psi_tok.shape == (c, c)
    for name, M in [("C_tok", C_tok), ("Psi_tok", Psi_tok)]:
        assert torch.isfinite(M).all(), name
        assert torch.allclose(M, M.t(), atol=1e-5), f"{name} not symmetric"
        ev = torch.linalg.eigvalsh(M.float())
        assert ev.min() > -1e-4, f"{name} not PSD: min eig {ev.min().item()}"


# --------------------------------------------------------------------------- #
# R-PEARL: deterministic (semantic node-feature) path
# --------------------------------------------------------------------------- #
def test_deterministic_path_forward_and_guards():
    """node_feature_dim set ⇒ deterministic GCN over data.x; probe readouts are forbidden."""
    nf = 5
    m = _rpearl(node_feature_dim=nf).eval()
    g = Data(x=torch.randn(6, nf), edge_index=_graph().edge_index, num_nodes=6)
    out = m(g)
    assert out.shape == (6, 16) and torch.isfinite(out).all()
    # determinism: no probes ⇒ two eval calls identical
    assert torch.allclose(out, m(Data(x=g.x.clone(), edge_index=g.edge_index, num_nodes=6)))

    for fn in [
        lambda: m(g, permutation=object()),
        lambda: m.second_moment_apply(g, torch.randn(6, 16)),
        lambda: m.covariance_token_block(g, 3),
    ]:
        raised = False
        try:
            fn()
        except NotImplementedError:
            raised = True
        assert raised, "semantic-feature path must reject random-probe readouts"


# --------------------------------------------------------------------------- #
# m_test REMOVAL — the focus of the task
# --------------------------------------------------------------------------- #
def test_mtest_kwarg_removed_rpearl():
    """The removed `m_test` constructor arg must now be rejected (TypeError)."""
    try:
        _rpearl(m_test=128)
    except TypeError:
        return
    assert False, "m_test kwarg should no longer be accepted by RandomGNNPositionalEncodings"


def test_mtest_kwarg_removed_gt():
    """GraphTransformer must no longer accept `m_test` (it stopped forwarding it)."""
    try:
        _gt(m_test=128)
    except TypeError:
        return
    assert False, "m_test kwarg should no longer be accepted by GraphTransformer"


def test_single_M_no_legacy_attrs():
    """self.M == num_samples; the legacy m_train/m_test attributes are gone."""
    m = _rpearl(num_samples=17)
    assert m.M == 17
    assert not hasattr(m, "m_train"), "m_train attribute should be removed"
    assert not hasattr(m, "m_test"), "m_test attribute should be removed"


def test_train_eval_use_same_probe_count():
    """Core of the removal: train() and eval() request the SAME probe count == M.

    Spy on _sample_probes to capture the requested m in each mode.
    """
    M = 9
    m = _rpearl(num_samples=M)
    seen = []
    orig = m._sample_probes

    def spy(num_nodes, mm, device, generator=None):
        seen.append(mm)
        return orig(num_nodes, mm, device, generator)

    m._sample_probes = spy
    g = _graph()
    m.train(); m(g)
    m.eval(); m(g)
    assert seen == [M, M], f"expected [{M},{M}] probes in train/eval, got {seen}"


def test_no_mtest_token_in_scope_files():
    """No `m_test`/`m_train` token survives in src/prism/models/ or experiments/."""
    roots = {
        "src/prism/models": (".py",),
        "experiments": (".yaml", ".yml"),
    }
    pat = re.compile(r"\bm_(test|train)\b")
    offenders = []
    for root, exts in roots.items():
        for dp, _, files in os.walk(root):
            for f in files:
                if f.endswith(exts):
                    path = os.path.join(dp, f)
                    with open(path) as fh:
                        for i, line in enumerate(fh, 1):
                            if pat.search(line):
                                offenders.append(f"{path}:{i}: {line.strip()}")
    assert not offenders, "stale m_test/m_train references:\n" + "\n".join(offenders)


# --------------------------------------------------------------------------- #
# Graph Transformer
# --------------------------------------------------------------------------- #
def test_gt_pe_generator_path():
    """token_embeddings=None ⇒ pure R-PEARL PE generator, [N, d_model] finite, NO gate."""
    g = _gt().eval()
    out = g(_graph(n=6))
    assert out.shape == (6, 16) and torch.isfinite(out).all()


def test_gt_fusion_mean_shape_and_gate():
    """pe_readout='mean' fusion: H0 = X_full + Ψ; output [N,d], gate=0 ⇒ exactly zero."""
    g = _gt(pe_readout="mean", fixed_seed_mode=True).eval()
    n, c = 6, 3
    tok = torch.randn(c, 16)
    is_token = torch.zeros(n, dtype=torch.bool)
    is_token[:c] = True
    out = g(_graph(n=n), token_embeddings=tok, is_token=is_token)
    assert out.shape == (n, 16) and torch.isfinite(out).all()
    with torch.no_grad():
        g.output_gain.fill_(0.0)
    out0 = g(_graph(n=n), token_embeddings=tok, is_token=is_token)
    assert torch.allclose(out0, torch.zeros_like(out0)), out0.abs().max().item()


def test_gt_fusion_second_moment_shape():
    """pe_readout='second_moment' fusion runs over the composite graph, [N,d] finite."""
    g = _gt(pe_readout="second_moment", fixed_seed_mode=True).eval()
    n, c = 6, 3
    tok = torch.randn(c, 16)
    is_token = torch.zeros(n, dtype=torch.bool)
    is_token[:c] = True
    out = g(_graph(n=n), token_embeddings=tok, is_token=is_token)
    assert out.shape == (n, 16) and torch.isfinite(out).all()


def test_gt_grad_flow():
    """Fusion forward/backward populates finite grads on blocks, pe_model, and the gate."""
    g = _gt(pe_readout="mean").train()
    n, c = 6, 3
    tok = torch.randn(c, 16, requires_grad=False)
    is_token = torch.zeros(n, dtype=torch.bool); is_token[:c] = True
    out = g(_graph(n=n), token_embeddings=tok, is_token=is_token)
    out.sum().backward()
    assert g.output_gain.grad is not None and torch.isfinite(g.output_gain.grad).all()
    block_grads = [p.grad for n_, p in g.blocks.named_parameters() if p.grad is not None]
    assert block_grads and all(torch.isfinite(x).all() for x in block_grads)
    assert any(x.abs().sum() > 0 for x in block_grads), "no grad reached the GT blocks"


def test_gt_pe_readout_validation():
    """Invalid pe_readout fails loud (ValueError)."""
    try:
        _gt(pe_readout="third_moment")
    except ValueError:
        return
    assert False, "expected ValueError for unknown pe_readout"


def test_expand_edge_index_khop():
    """_expand_edge_index ≤k-hop matches dense (A+I)^k > 0 on a path graph."""
    g = _gt(k_gt=2)
    n = 5
    # path 0-1-2-3-4 (undirected)
    e = [(i, i + 1) for i in range(n - 1)] + [(i + 1, i) for i in range(n - 1)]
    edge_index = torch.tensor(e, dtype=torch.long).t().contiguous()
    out = g._expand_edge_index(edge_index, n, k_hops=2)
    got = torch.zeros(n, n, dtype=torch.bool)
    got[out[0], out[1]] = True
    # dense oracle: (A+I)^2 > 0
    A = torch.eye(n)
    A[edge_index[0], edge_index[1]] = 1.0
    ref = (A @ A) > 0
    assert torch.equal(got, ref), f"\n{got.int()}\nvs\n{ref.int()}"


def test_sparse_attention_parity_dense_oracle():
    """_SafeBatchedSparseAttn == dense masked-softmax attention over A's pattern.

    Independent oracle: per head, scores=(Q Kᵀ)·scale masked to A's nonzeros, row-softmax,
    @V; heads concatenated. Self-loops ensure every row has a neighbor (no all-masked rows).
    """
    torch.manual_seed(0)
    H, N, Fh = 3, 5, 4
    scale = float(Fh) ** -0.5
    Q = torch.randn(H, N, Fh)
    K = torch.randn(H, N, Fh)
    V = torch.randn(H, N, Fh)
    # ring + self loops so every node has >=1 neighbor
    e = [(i, (i + 1) % N) for i in range(N)] + [(i, i) for i in range(N)]
    ei = torch.tensor(e, dtype=torch.long).t().contiguous()
    vals = torch.ones(ei.shape[1])
    A = torch.sparse_coo_tensor(ei, vals, (N, N)).coalesce().to_sparse_csr()
    scale_t = torch.tensor(scale)

    got = gtmod._SafeBatchedSparseAttn.apply(Q, K, V, A, scale_t, None, False)  # [N, H*Fh]

    # dense reference
    Ad = torch.zeros(N, N, dtype=torch.bool)
    coo = A.to_sparse_coo().coalesce()
    Ad[coo.indices()[0], coo.indices()[1]] = True
    heads_out = []
    for h in range(H):
        scores = (Q[h] @ K[h].t()) * scale
        scores = scores.masked_fill(~Ad, float("-inf"))
        alpha = torch.softmax(scores, dim=1)
        heads_out.append(alpha @ V[h])
    ref = torch.cat(heads_out, dim=1)  # [N, H*Fh]
    assert torch.allclose(got, ref, atol=1e-5), (got - ref).abs().max().item()


def test_sparse_attention_backward_finite():
    """_SafeBatchedSparseAttn backward gives finite, non-trivial grads on Q,K,V."""
    torch.manual_seed(0)
    H, N, Fh = 2, 4, 4
    Q = torch.randn(H, N, Fh, requires_grad=True)
    K = torch.randn(H, N, Fh, requires_grad=True)
    V = torch.randn(H, N, Fh, requires_grad=True)
    e = [(i, (i + 1) % N) for i in range(N)] + [(i, i) for i in range(N)]
    ei = torch.tensor(e, dtype=torch.long).t().contiguous()
    A = torch.sparse_coo_tensor(ei, torch.ones(ei.shape[1]), (N, N)).coalesce().to_sparse_csr()
    out = gtmod._SafeBatchedSparseAttn.apply(Q, K, V, A, torch.tensor(Fh ** -0.5), None, False)
    out.sum().backward()
    for name, t in [("Q", Q), ("K", K), ("V", V)]:
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
    assert V.grad.abs().sum() > 0


def test_block_normalize_flag():
    """SparseTransformerBlock holds 2 LayerNorms iff normalize=True; both run finite."""
    bn = gtmod.SparseTransformerBlock(16, heads=4, dropout=0.0, normalize=True)
    bf = gtmod.SparseTransformerBlock(16, heads=4, dropout=0.0, normalize=False)
    assert len(bn.norms) == 2 and len(bf.norms) == 0
    x = torch.randn(5, 16)
    e = [(i, (i + 1) % 5) for i in range(5)] + [(i, i) for i in range(5)]
    ei = torch.tensor(e, dtype=torch.long).t().contiguous()
    for b in (bn.eval(), bf.eval()):
        y = b(x, ei)
        assert y.shape == (5, 16) and torch.isfinite(y).all()


def test_semantic_gt_forward_and_guards():
    """SemanticGraphTransformer: [N,d] finite, gate=0 ⇒ zero, permutation forbidden."""
    nf = 5
    torch.manual_seed(0)
    m = gtmod.SemanticGraphTransformer(node_feature_dim=nf, d_model=16, num_layers=2,
                                       heads=4, dropout=0.0, k_gt=2).eval()
    g = Data(x=torch.randn(6, nf), edge_index=_graph().edge_index, num_nodes=6)
    out = m(g)
    assert out.shape == (6, 16) and torch.isfinite(out).all()
    with torch.no_grad():
        m.output_gain.fill_(0.0)
    assert torch.allclose(m(g), torch.zeros(6, 16))
    raised = False
    try:
        m(g, permutation=object())
    except NotImplementedError:
        raised = True
    assert raised, "SemanticGraphTransformer must reject permutation eval"


# --------------------------------------------------------------------------- #
# END-TO-END: fully-instantiated untrained GT over a REAL composite graph
# --------------------------------------------------------------------------- #
def _composite():
    """Real composite graph via the production builder (token cycle + scene + crosslinks)."""
    from prism.models.composite_graph import build_composite_graph
    c, n_scene = 16, 6
    inj = {0: [(2, 4)], 3: [(9, 11)]}
    scene_ei = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    scene_ew = torch.ones(scene_ei.shape[1])
    cg = build_composite_graph(c, scene_ei, scene_ew, n_scene, inj, cycle_directed=True)
    pe_data = Data(x=torch.zeros(cg.num_nodes, 1), edge_index=cg.edge_index)
    pe_data.edge_weight = cg.edge_weight
    return cg, pe_data, c


def test_e2e_full_gt_over_composite_graph():
    """Fully-instantiated untrained GT runs forward+backward on a real composite graph.

    CS-viability: both readouts (mean / second_moment), fp32 + bf16, finite outputs of
    the correct shape, and gradients reaching R-PEARL + the GT blocks + the gate with no
    NaNs. Output magnitude must be sane (non-zero, non-exploding) — NOT pinned to the
    embedding scale (the spectral/Lipschitz magnitude-pinning was removed).
    """
    cg, pe_data, c = _composite()
    d_model = 64
    X = torch.randn(c, d_model)
    X = X / X.norm(dim=-1, keepdim=True) * 24.0  # Llama-like embedding scale
    for mode in ("mean", "second_moment"):
        torch.manual_seed(0)
        gt = gtmod.GraphTransformer(num_layers=3, pe_hidden_channels=32, pe_num_layers=3,
                                    d_model=d_model, heads=4, num_samples=8, k_pe=2,
                                    k_gt=2, eps=1e-6, pe_readout=mode)
        gt.eval()
        with torch.no_grad():
            Y = gt(pe_data, token_embeddings=X, is_token=cg.is_token)
        assert Y.shape == (cg.num_nodes, d_model)
        assert torch.isfinite(Y).all(), f"{mode}: non-finite output"
        tok_norm = Y[cg.is_token].norm(dim=-1).mean().item()
        assert 0.0 < tok_norm < 1e3, f"{mode}: insane output magnitude {tok_norm}"
        # bf16 token-embedding path must run and stay finite
        with torch.no_grad():
            Yb = gt(pe_data, token_embeddings=X.to(torch.bfloat16), is_token=cg.is_token)
        assert Yb.dtype == torch.bfloat16 and torch.isfinite(Yb).all(), f"{mode}: bf16 path"
        # train backward: grads reach R-PEARL, blocks, gate; none NaN
        gt.train()
        gt.zero_grad()
        gt(pe_data, token_embeddings=X, is_token=cg.is_token).sum().backward()
        pe_grad = sum(1 for p in gt.pe_model.parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0)
        blk_grad = sum(1 for p in gt.blocks.parameters()
                       if p.grad is not None and p.grad.abs().sum() > 0)
        assert pe_grad > 0, f"{mode}: no grad reached R-PEARL"
        assert blk_grad > 0, f"{mode}: no grad reached GT blocks"
        assert gt.output_gain.grad is not None and torch.isfinite(gt.output_gain.grad).all()
        assert not any(torch.isnan(p.grad).any() for p in gt.parameters()
                       if p.grad is not None), f"{mode}: NaN gradient"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
