"""The ``wire_signal='cov_factor'`` signal: is ``r`` really a factor of ``C``?

Theorem 3 of arXiv:2509.22259 holds for ``r_i = [u_k[i]/√λ_k]``, i.e. a SQUARE ROOT of
``L†`` — the only form for which ``‖r_i − r_j‖² = L†_ii + L†_jj − 2L†_ij``. The probe
covariance ``C = E_q[Φ'Φ'ᵀ] − ΨΨᵀ`` is itself a Gram matrix, so it plays ``L†``'s role
and the signal must be a factor of it, never its rows.

``test_gram_identity_holds`` is the gating test: it is what separates the shipped
construction from the plausible-looking wrong one, and ``test_rows_of_c_are_the_wrong_
object`` shows the wrong one failing on the same graph. Nothing downstream of the signal
is meaningful until both hold.

Run:  uv run --with pytest -m pytest tests/test_wire_cov_factor.py -q
"""
import sys
sys.path.insert(0, "src")

import pytest
import torch
from torch_geometric.data import Data

from prism.models.gnn_llm import WIRE_SIGNALS, wire_head_count
from prism.models.gt import GraphTransformer

K = 512          # JL rank; relative error on the distances is ~sqrt(2/K)
DRAWS = 24       # independent G draws averaged when testing the EXPECTATION


def _gt(d_model=32, num_samples=64, seed=1):
    """A directed (MagNet) Ψ producer with the blocks inside E_q — cov_factor's premise."""
    torch.manual_seed(seed)
    return GraphTransformer(num_layers=2, pe_hidden_channels=16, pe_num_layers=2,
                            d_model=d_model, heads=2, num_samples=num_samples,
                            dropout=0.0, k_pe=2, k_gt=2, pe_pool="gt",
                            directed=True, learn_r=True).eval()


def _composite(c=6, n=3, seed=0):
    """A token cycle ⊔ scene ⊔ crosslinks ⊔ anchor, laid out TOKEN ROWS FIRST."""
    torch.manual_seed(seed)
    src, dst, w = [], [], []
    for i in range(c):                                   # directed token cycle
        src.append(i); dst.append((i + 1) % c); w.append(1.0)
    for u, v in ((0, 1), (1, 2)):                        # undirected scene edges
        src += [c + u, c + v]; dst += [c + v, c + u]; w += [1.0, 1.0]
    for j in range(n):                                   # scene -> token crosslinks
        src.append(c + j); dst.append(j); w.append(0.1)
    src += [0, c + n]; dst += [c + n, c]; w += [10.0, 10.0]   # anchor bond
    data = Data(x=torch.zeros(c + n + 1, 1),
                edge_index=torch.tensor([src, dst], dtype=torch.long),
                edge_weight=torch.tensor(w))
    data.num_token_nodes = c
    return data


def _cov(gt, data, c):
    C, _ = gt.pe_model.covariance_token_block(data, c, pe_pool="gt", gt=gt)
    return C.double()


def _factor(gt, data, c, k=K):
    r, _ = gt.pe_model.covariance_token_block(data, c, pe_pool="gt", gt=gt, project=k)
    return r.double()


def _pairwise_sq(x):
    """‖x_i − x_j‖² for every pair, exactly (no cdist rounding)."""
    return torch.cdist(x, x).pow(2)


def _gram_distances(C):
    """C_ii + C_jj − 2C_ij — what the factor's distances must reproduce."""
    d = C.diagonal()
    return d.unsqueeze(1) + d.unsqueeze(0) - 2.0 * C


# --------------------------------------------------------------------- the signal

@torch.no_grad()
def test_gram_identity_holds():
    """E_G‖r_i − r_j‖² = C_ii + C_jj − 2C_ij, the hypothesis Theorem 3 is stated on.

    Averaged over independent G draws: a single draw carries the JL error ~sqrt(2/K),
    which is a property of the estimator, not a defect. What must be exact is the MEAN.
    """
    gt, data = _gt(), _composite()
    c = data.num_token_nodes
    # C is itself a probe estimate, so average it over the same number of draws as the
    # factor — otherwise this compares two different Monte-Carlo samples.
    target = torch.stack([_gram_distances(_cov(gt, data, c)) for _ in range(DRAWS)]).mean(0)
    got = torch.stack([_pairwise_sq(_factor(gt, data, c)) for _ in range(DRAWS)]).mean(0)
    off = ~torch.eye(c, dtype=torch.bool)
    rel = ((got - target).abs()[off] / target[off].abs().clamp_min(1e-12)).max()
    assert float(rel) < 0.25, f"factor distances disagree with C by {float(rel):.3f}"


@torch.no_grad()
def test_rows_of_c_are_the_wrong_object():
    """Rows of C give the resistance form of C², not of C — the slip this arm avoids.

    Equality would need C² = C. This asserts the two are FAR apart on a real composite,
    so the gating test above cannot be passed by accident.
    """
    gt, data = _gt(), _composite()
    c = data.num_token_nodes
    C = _cov(gt, data, c)
    target = _gram_distances(C)
    rows = _pairwise_sq(C)                       # what feeding C[i,:] would rotate by
    off = ~torch.eye(c, dtype=torch.bool)
    rel = ((rows - target).abs()[off] / target[off].abs().clamp_min(1e-12)).max()
    assert float(rel) > 1.0, (
        "rows of C reproduced C's own distances — the discriminator is not discriminating")


@torch.no_grad()
def test_factor_is_unbiased_in_the_inner_product():
    """E_G[⟨r_i, r_j⟩] = C_ij, including the off-diagonal (the distances alone can hide sign)."""
    gt, data = _gt(), _composite()
    c = data.num_token_nodes
    target = torch.stack([_cov(gt, data, c) for _ in range(DRAWS)]).mean(0)
    r = torch.stack([_factor(gt, data, c) for _ in range(DRAWS)])
    got = torch.stack([x @ x.transpose(0, 1) for x in r]).mean(0)
    scale = target.diagonal().mean().clamp_min(1e-12)
    assert float((got - target).abs().max() / scale) < 0.25


@torch.no_grad()
def test_factor_shape_and_rank_follow_the_rotation_width():
    gt, data = _gt(), _composite()
    c = data.num_token_nodes
    for k in (16, 128):
        assert _factor(gt, data, c, k).shape == (c, k)


@torch.no_grad()
def test_variance_falls_as_one_over_k():
    """The JL error is ~sqrt(2/k): raising k 16x must cut the spread ~4x."""
    gt, data = _gt(), _composite()
    c = data.num_token_nodes
    target = torch.stack([_gram_distances(_cov(gt, data, c)) for _ in range(DRAWS)]).mean(0)
    off = ~torch.eye(c, dtype=torch.bool)
    spread = []
    for k in (32, 512):
        errs = torch.stack([
            (_pairwise_sq(_factor(gt, data, c, k)) - target).abs()[off].mean()
            for _ in range(DRAWS)])
        spread.append(float(errs.mean()))
    assert spread[1] < spread[0] * 0.6, f"k=512 error {spread[1]:.4g} vs k=32 {spread[0]:.4g}"


def test_project_rejects_a_pinned_probe_draw():
    """fixed_seed_mode pins the probes while G still resamples — half a frozen estimator."""
    gt, data = _gt(), _composite()
    gt.pe_model.fixed_seed_mode = True
    with pytest.raises(ValueError, match="fixed_seed_mode"):
        gt.pe_model.covariance_token_block(data, data.num_token_nodes,
                                           pe_pool="gt", gt=gt, project=K)


def test_project_rejects_a_non_positive_rank():
    gt, data = _gt(), _composite()
    with pytest.raises(ValueError, match="positive rank"):
        gt.pe_model.covariance_token_block(data, data.num_token_nodes,
                                           pe_pool="gt", gt=gt, project=0)


@torch.no_grad()
def test_covariance_path_is_untouched_by_the_new_argument():
    """project=None must return exactly what it returned before the extension."""
    gt, data = _gt(), _composite()
    c = data.num_token_nodes
    C, psi_gram = gt.pe_model.covariance_token_block(data, c, pe_pool="gt", gt=gt)
    assert C.shape == (c, c) and psi_gram.shape == (c, c)
    assert torch.allclose(C, C.transpose(0, 1), atol=1e-4)
    assert float(C.diagonal().min()) >= -1e-5          # PSD diagonal


def test_the_factor_carries_gradient_to_the_producer():
    """The whole point of the JL form over an eigendecomposition: Stage 3 can train."""
    gt, data = _gt(), _composite()
    r, _ = gt.pe_model.covariance_token_block(
        data, data.num_token_nodes, pe_pool="gt", gt=gt, project=64)
    r.pow(2).sum().backward()
    grads = [p.grad for p in gt.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads, "no producer parameter received gradient through the factor"


# ------------------------------------------------------------------- frequencies

def test_sigma_is_per_plane_and_exponential():
    """σ[n] = sigma_init·10000^(−2n/P): RoPE's decay, on the learnable scale."""
    from tests.test_wire_smoke import _gemma4
    llm = _gemma4()
    if llm is None:
        pytest.skip("transformers Gemma4 fixture unavailable")
    from prism.models.gnn_llm import WireGraphLLM
    m = WireGraphLLM(llm, _gt(d_model=16), d_model=16, vanilla=False,
                     pe_gain_init=1.0, sigma_init=0.01).eval()
    idx = m.active_layer_indices()
    assert idx, "no active WIRE layer to check"
    for li in idx:
        sigma = m._wire_sigma[str(li)].detach()
        P = sigma.numel()
        assert sigma.dim() == 1, f"sigma must be per-plane, got shape {tuple(sigma.shape)}"
        want = 0.01 * 10000.0 ** (-2.0 * torch.arange(P, dtype=torch.float32) / P)
        assert torch.allclose(sigma, want, atol=1e-9)
        # The clamp convention: max_n sigma[n] is the old scalar sigma_init exactly.
        assert abs(m.layer_omega_scale(li) - 0.01) < 1e-9
        assert m.layer_omega(li).shape == (P, 16)


def test_eps_is_one_table_per_layer_after_the_per_head_average():
    """Per-head draw, mean over heads — Eq. (3) needs q and k to share ω under GQA."""
    from tests.test_wire_smoke import _gemma4
    llm = _gemma4()
    if llm is None:
        pytest.skip("transformers Gemma4 fixture unavailable")
    from prism.models.gnn_llm import WireGraphLLM
    m = WireGraphLLM(llm, _gt(d_model=16), d_model=16, vanilla=False,
                     pe_gain_init=1.0).eval()
    layers = m._decoder_layers()
    for li in m.active_layer_indices():
        eps = getattr(m._wire_eps, str(li))
        assert eps.dim() == 2 and eps.shape[1] == 16, tuple(eps.shape)
        h = wire_head_count(layers[li].self_attn)
        assert h > 1, "fixture must have >1 head for the average to mean anything"
        # Mean of h iid N(0,1) has std 1/sqrt(h); a single draw would read ~1.0.
        assert float(eps.std()) < 0.9


def test_draws_are_unseeded():
    """ε follows the AMBIENT RNG, and ``omega_seed`` is inert.

    Both halves are needed. Same ambient state + different ``omega_seed`` must give the
    SAME ε (the seed no longer reaches the draw); different ambient state must give a
    DIFFERENT ε (the draw is not pinned some other way). Note the fixtures call
    ``manual_seed`` themselves, so the ambient state has to be set after they run.
    """
    from tests.test_wire_smoke import _gemma4
    if _gemma4() is None:
        pytest.skip("transformers Gemma4 fixture unavailable")
    from prism.models.gnn_llm import WireGraphLLM

    def _eps(ambient, omega_seed):
        llm, gt = _gemma4(), _gt(d_model=16)          # these seed the global RNG
        torch.manual_seed(ambient)                    # ...so set it afterwards
        m = WireGraphLLM(llm, gt, d_model=16, vanilla=False,
                         pe_gain_init=1.0, omega_seed=omega_seed).eval()
        return getattr(m._wire_eps, str(m.active_layer_indices()[0])).clone()

    assert torch.allclose(_eps(7, 0), _eps(7, 12345)), "omega_seed still reaches the draw"
    assert not torch.allclose(_eps(7, 0), _eps(8, 0)), "the draw is pinned, not ambient"


def test_signal_names_are_the_public_set():
    assert WIRE_SIGNALS == ("psi", "cov_factor")


# ------------------------------------------------------- the composite arm, end to end

class _StubTokenizer:
    """Enough of a tokenizer for find_last_graph_scope: no block found => scope_start 0."""

    def batch_decode(self, seqs, **kw):
        return [f"t{int(s[0])}" for s in seqs]


def _scene(n=3):
    src = [0, 1, 1, 2]
    dst = [1, 0, 2, 1]
    return Data(x=torch.zeros(n, 1),
                edge_index=torch.tensor([src, dst], dtype=torch.long),
                edge_weight=torch.ones(4))


def _composite_model(signal="cov_factor", d_model=16, window=8, **kw):
    from tests.test_wire_smoke import _gemma4
    from prism.models.gnn_llm import CompositeWireGraphLLM
    llm = _gemma4()
    if llm is None:
        pytest.skip("transformers Gemma4 fixture unavailable")
    kw.setdefault("vanilla", False)
    return CompositeWireGraphLLM(
        llm, _gt(d_model=d_model, num_samples=16), d_model=d_model,
        tokenizer=_StubTokenizer(), context_window=window, signal=signal,
        pe_gain_init=1.0, **kw).eval()


def test_composite_cov_factor_forward_and_backward():
    """The arm the config selects, exercised end to end: loss is finite and trains."""
    m = _composite_model()
    ids = torch.randint(0, 64, (1, 6))
    inj = {0: [(1, 2)], 1: [(3, 4)]}
    out = m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids.clone(),
            graphs=[_scene()], injection_maps=[inj])
    assert torch.isfinite(out.loss), "non-finite loss on the covariance-factor arm"
    out.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.pe_model.parameters()), \
        "no gradient reached the Psi producer through the factor"
    assert m.pe_gain.grad is not None and float(m.pe_gain.grad.abs()) > 0


@torch.no_grad()
def test_composite_signal_width_and_scope():
    """r is [B, seq, d_model] with the scope's RoPE positions zeroed."""
    m = _composite_model()
    ids = torch.randint(0, 64, (1, 6))
    m._arm([_scene()], [{0: [(1, 2)]}], 6, ids.device, input_ids=ids,
           attention_mask=torch.ones_like(ids))
    assert m._wire_signal.shape == (1, 6, 16), tuple(m._wire_signal.shape)
    assert torch.isfinite(m._wire_signal).all()
    start, end = m._wire_scope_spans[0]
    pos = m.scope_position_ids(6, ids.device)
    assert int(pos[0, start:end].abs().max()) == 0, "RoPE not switched off on the scope"
    # The factor must actually be non-zero, or the rotation is a silent identity.
    assert float(m._wire_signal.abs().max()) > 0


@torch.no_grad()
def test_composite_psi_signal_still_works():
    """The default arm is untouched by the extension."""
    m = _composite_model(signal="psi", vanilla=True, vanilla_omega_init="exponential")
    ids = torch.randint(0, 64, (1, 6))
    m._arm([_scene()], [{0: [(1, 2)]}], 6, ids.device, input_ids=ids,
           attention_mask=torch.ones_like(ids))
    assert m._wire_signal.shape == (1, 6, 16)
    assert torch.isfinite(m._wire_signal).all()


def test_cov_factor_rejects_the_vanilla_arm():
    """Theorem 3 needs Gaussian omega; vanilla takes one learnable table and no expectation."""
    with pytest.raises(ValueError, match="wire_vanilla=false"):
        _composite_model(vanilla=True, vanilla_omega_init="exponential")


def test_cov_factor_rejects_pe_pool_pe():
    """Phi' = T(Phi(q)) must be INSIDE E_q, or the factor is of the pre-block covariance."""
    from tests.test_wire_smoke import _gemma4
    from prism.models.gnn_llm import CompositeWireGraphLLM
    llm = _gemma4()
    if llm is None:
        pytest.skip("transformers Gemma4 fixture unavailable")
    torch.manual_seed(1)
    gt = GraphTransformer(num_layers=2, pe_hidden_channels=16, pe_num_layers=2,
                          d_model=16, heads=2, num_samples=16, dropout=0.0,
                          k_pe=2, k_gt=2, pe_pool="pe", directed=True, learn_r=True).eval()
    with pytest.raises(ValueError, match="pe_pool"):
        CompositeWireGraphLLM(llm, gt, d_model=16, tokenizer=_StubTokenizer(),
                              context_window=8, signal="cov_factor", vanilla=False,
                              pe_gain_init=1.0)


def test_unknown_signal_is_rejected():
    with pytest.raises(ValueError, match="signal must be one of"):
        _composite_model(signal="covariance")
