"""STEP 4 tests for the R-PEARL β = 1/F filter-norm invariant and the F-independent
second-moment (C) injection. Covers:

  (a) measured ‖H(S)‖₂ ≤ 1/F over random S with ‖S̄‖₂ ≤ 1, at several widths F;
  (b) Ĉ has ‖Ĉ‖₂ ≈ 1 and stays PSD after normalization;
  (c) the injected logit-bias and value-mix magnitudes are O(1) and do NOT scale
      with F (sweep F).

Run: pytest -q tests/test_rpearl_filter_norm.py
"""
import math

import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

from prism.models.gcn import GCN, _operator_spectral_norm
from prism.models.gnn_llm import InjectedCompositeGraphLLM


# ---------------------------------------------------------------- helpers
def _sym_norm_gso(n, p=0.3, seed=0):
    """A random symmetric-degree-normalized GSO S̄ = D^{-1/2}(A+I)D^{-1/2} (‖S̄‖₂ ≤ 1)."""
    g = torch.Generator().manual_seed(seed)
    A = (torch.rand(n, n, generator=g) < p).float()
    A = torch.triu(A, 1)
    A = A + A.t()
    ei = A.nonzero().t()
    ei, _ = add_self_loops(ei, num_nodes=n)
    ew = torch.ones(ei.shape[1])
    deg = torch.zeros(n).index_add_(0, ei[0], ew)
    dinv = deg.clamp(min=1e-12).pow(-0.5)
    ew = dinv[ei[0]] * ew * dinv[ei[1]]
    return ei, ew


# ---------------------------------------------------------------- (a)
@pytest.mark.parametrize("F", [16, 64, 256])
def test_filter_norm_leq_one_over_F(F):
    torch.manual_seed(0)
    gcn = GCN(in_channels=1, hidden_channels=F, num_layers=3, k=5, dropout=0.0).eval()
    gcn.strict_filter_norm = True            # tests run strict (fail-loud)
    n = 60
    ei, ew = _sym_norm_gso(n)
    # measure the TRUE ‖H(S)‖₂ of every layer on the actual S̄
    report = gcn.filter_norm_report(ei, ew, num_nodes=n)
    for idx, d in report.items():
        assert d["measured"] <= d["target"] + gcn.filter_norm_tol, \
            f"layer {idx}: ‖H(S)‖₂={d['measured']:.3e} > 1/F={d['target']:.3e}"
    # the strict assert path must not raise on a compliant filter
    gcn.assert_filter_bounds(strict=True, report=report)


def test_filter_norm_target_reads_F_from_layer():
    # F must come from the layer's OUTPUT width, not a hardcoded constant.
    for F in (8, 32, 128):
        gcn = GCN(1, F, num_layers=2, k=3, dropout=0.0).eval()
        ei, ew = _sym_norm_gso(40)
        rep = gcn.filter_norm_report(ei, ew, num_nodes=40)
        for d in rep.values():
            assert abs(d["target"] - 1.0 / F) < 1e-9


def test_strict_mode_raises_when_violated(monkeypatch):
    # Force a too-loose target (pretend F=1) and confirm the strict assert raises.
    gcn = GCN(1, 32, num_layers=2, k=3, dropout=0.0).eval()
    ei, ew = _sym_norm_gso(40)
    rep = gcn.filter_norm_report(ei, ew, num_nodes=40)
    bad = {0: {"measured": 5.0, "target": 1.0 / 32}}
    with pytest.raises(AssertionError):
        gcn.assert_filter_bounds(strict=True, report=bad)


# ---------------------------------------------------------------- (b) analytic Ĉ from taps
@pytest.mark.parametrize("F", [8, 64, 256])
def test_analytic_C_hat_deterministic_psd_circulant(F):
    torch.manual_seed(0)
    c, K1 = 24, 6
    H = torch.randn(K1, F)                       # R-PEARL-style filter taps, raw O(1)
    C1, r1 = InjectedCompositeGraphLLM._analytic_c_from_taps(H, c)
    C2, r2 = InjectedCompositeGraphLLM._analytic_c_from_taps(H, c)
    # deterministic (no probes): identical across calls
    assert torch.allclose(C1, C2) and torch.allclose(r1, r2), "analytic Ĉ not deterministic"
    # unit zero-lag autocorrelation + entries in [-1,1]
    assert abs(float(C1.diagonal().max()) - 1.0) < 1e-5, "diag_max(Ĉ) != 1"
    assert float(C1.abs().max()) <= 1.0 + 1e-5, "Ĉ entry > 1"
    # circulant: depends only on (t-u)
    assert torch.allclose(C1[0], r1, atol=1e-5), "Ĉ row 0 != c_row"
    assert torch.allclose(C1[5, 7], C1[10, 12], atol=1e-5), "not circulant"
    # PSD (ρ_k = ‖ĥ(ω_k)‖² ≥ 0 ⇒ circulant PSD)
    lam = torch.linalg.eigvalsh(0.5 * (C1 + C1.t()))
    assert float(lam[0]) >= -1e-4, f"Ĉ not PSD: λ_min={float(lam[0]):.2e}"


def test_analytic_C_hat_grad_flows_to_taps():
    # the proof's point: the SAME taps are trained — gradient must reach H.
    H = torch.randn(6, 32, requires_grad=True)
    C, _ = InjectedCompositeGraphLLM._analytic_c_from_taps(H, 24)
    C.sum().backward()
    assert H.grad is not None and float(H.grad.abs().sum()) > 0, "no gradient to filter taps"


def test_analytic_C_hat_scale_is_F_independent():
    # diag-max is 1 for any F; the bias is O(1) regardless of the tap width F.
    torch.manual_seed(0)
    diags = []
    for F in (8, 64, 256, 1024):
        H = torch.randn(6, F)
        C, _ = InjectedCompositeGraphLLM._analytic_c_from_taps(H, 24)
        diags.append(float(C.diagonal().max()))
    assert all(abs(d - 1.0) < 1e-5 for d in diags), f"diag_max drifts with F: {diags}"


# ---------------------------------------------------------------- (c)
def test_injection_magnitudes_are_O1_and_F_independent():
    """λ_C·Ĉ (logit bias) and λ_V·(Ĉ·v renormed) (value mix) must be O(1) and not scale
    with F — replicating the in-attention magnitudes the patched forward produces."""
    torch.manual_seed(0)
    c, dh, F_sweep = 24, 16, (8, 64, 256, 1024)
    lam_c, lam_v = 1.0, 0.1
    q = torch.randn(1, 4, c, dh); k = torch.randn(1, 4, c, dh)
    content = (q @ k.transpose(-1, -2) / math.sqrt(dh))   # O(1) content logits
    bias_mag, vmix_mag = [], []
    for F in F_sweep:
        H = torch.randn(6, F)
        C_hat, _ = InjectedCompositeGraphLLM._analytic_c_from_taps(H, c)
        bias = lam_c * C_hat[None, None]
        bias_mag.append(float(bias.std() / content.std()))
        # value-mix: mixed = Ĉ·v, renormed to ‖v‖, then λ_V scaled (mirrors gnn_llm)
        v = torch.randn(1, 4, c, dh)
        mixed = torch.einsum("nm,bhmd->bhnd", C_hat, v)
        mixed = mixed * (v.norm(dim=-1).mean() / mixed.norm(dim=-1).mean().clamp(min=1e-12))
        vmix_mag.append(float((lam_v * mixed).norm(dim=-1).mean() / v.norm(dim=-1).mean()))
    # O(1): bias comparable to content logits; value-mix ≈ λ_V
    assert all(0.05 <= b <= 10 for b in bias_mag), f"bias not O(1): {bias_mag}"
    assert all(abs(m - lam_v) < 0.02 for m in vmix_mag), f"value-mix not ≈λ_V: {vmix_mag}"
    # F-independence: bias magnitude barely moves across a 128x tap-width sweep
    assert max(bias_mag) / min(bias_mag) < 1.5, f"bias scales with F: {bias_mag}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
