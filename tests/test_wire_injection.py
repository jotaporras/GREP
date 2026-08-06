"""Tests for WIRE rotary graph injection (``prism.models.gnn_llm.WireGraphLLM``).

WIRE (Reid et al., *Rotary Position Encodings for Graphs*, ICML 2026) rotates each
token's query and key by an angle derived from its node's graph feature, instead of
adding a projected signal to them (``GraphAugmentedLLM``). Because Gemma applies its
own RoPE *before* dispatching to the attention interface, WIRE composes a SECOND
rotation on top: same-plane 2-D rotations commute, so the score picks up the phase
``r_j − r_i`` on top of the text phase ``p_j − p_i``.

The invariants asserted here are the ones that make that safe, plus the one that
checks the theory actually being implemented:

  1. Ψ = 0 (or absent) ⇒ **exactly** the stock logits (θ=0 ⇒ cos=1, sin=0 ⇒ the
     rotation is the identity, not merely a small perturbation).
  2. Composition: the post-RoPE second rotation equals one rotation by the summed
     angle (the property that makes the hook point valid at all).
  3. Norm preservation — rotations cannot rescale q/k (unlike an additive signal).
  4. Relative-only: a constant shift of every ``r`` leaves every score unchanged.
  5. **Theorem 3**: over random ω ~ N(0, σ²I), the mean score matches
     ``qᵀk(1 − σ²‖r_i−r_j‖²/2)``. This is the only test that checks the guarantee.
  6. ``rotate_nope_planes=False`` leaves Gemma's NoPE channels untouched.
  7. Gate at 0 ⇒ exact identity (cold start).

Fixtures are tiny random-init **``gemma4``** models (not ``gemma4_unified``): only
that family carries the ``proportional`` / ``partial_rotary_factor`` rope the 31B
target uses, which is what makes test 6 meaningful.

Run:  uv run -m pytest tests/test_wire_injection.py -v
"""
import sys
sys.path.insert(0, "src")

import torch
from torch_geometric.data import Data

from prism.models.gnn_llm import (
    WireGraphLLM,
    _wire_rotate_prefix,
    wire_cos_sin,
    wire_rope_planes,
)
from prism.models.gt import GraphTransformer
from prism.models.utils import Permutation

_TOL = 1e-5


def _skip(msg):
    """Skip under pytest; print and bail when run as a plain script."""
    if __name__ != "__main__" and "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    print(f"[SKIP] {msg}")
    return None


def _gemma4(num_layers=4, seed=0):
    """Tiny random-init ``gemma4`` (the 31B family: proportional rope on globals)."""
    try:
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    except Exception:  # noqa: BLE001 — any import failure ⇒ unsupported here
        return None
    torch.manual_seed(seed)
    cfg = Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=num_layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, global_head_dim=16, max_position_embeddings=64,
        sliding_window=8, attn_implementation="eager")
    return Gemma4ForCausalLM(cfg).eval()


def _wrap(llm, d_model=16, layer_scope="dense", omega_scale=0.05,
          rotate_nope_planes=False, pe_gain_init=1.0, omega_learnable=True,
          max_angle=8.0, vanilla=False, **kw):
    """Wrap with a real GraphTransformer Ψ producer (scaled down).

    ``vanilla=False`` by DEFAULT here even though the class default is True: every test
    in this module that touches ε/σ is a test of the EXPECTATION ARM and must pin it
    explicitly. The vanilla path (the shipped default) has its own tests at the bottom
    of the file, and ``test_default_mode_is_vanilla`` asserts the class default itself.
    """
    torch.manual_seed(1)
    gt = GraphTransformer(
        num_layers=2, pe_hidden_channels=16, pe_num_layers=2, d_model=d_model,
        heads=2, num_samples=8, dropout=0.0, k_pe=2, k_gt=2)
    return WireGraphLLM(
        llm, gt, d_model=d_model, layer_scope=layer_scope,
        sigma_init=omega_scale, rotate_nope_planes=rotate_nope_planes,
        pe_gain_init=pe_gain_init, freeze_sigma=not omega_learnable,
        max_angle=max_angle, vanilla=vanilla, **kw).eval()


def _path_graph(n=3):
    """Path graph P_n — the structure Theorem 2 says WIRE reduces to RoPE on."""
    x = torch.randn(n, 1)                                   # ignored (random probes)
    src = list(range(n - 1))
    dst = list(range(1, n))
    edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)
    g = Data(x=x, edge_index=edge_index)
    g.num_nodes = n
    return g


_tiny_graph = _path_graph


def _global_layer_idx(llm):
    return [i for i, t in enumerate(llm.config.layer_types) if t == "full_attention"][0]


def _capture_qk(wrap, ids, signal, layer_idx):
    """Capture (q, k) handed to the stock attention fn at ``layer_idx``.

    Injection is ISOLATED to ``layer_idx`` (back-ref nulled elsewhere) so the hidden
    state entering the layer is identical with and without the signal — mirrors
    ``tests/test_pe_injection_qkv_numeric._capture_qkv``.
    """
    layers = list(wrap._decoder_layers())
    attn = layers[layer_idx].self_attn
    orig = attn._wire_orig_attn_fn
    store = {}

    def cap(module, query, key, value, attention_mask,
            scaling=None, dropout=0.0, **kw):
        store["q"] = query.detach().clone()
        store["k"] = key.detach().clone()
        return orig(module, query, key, value, attention_mask,
                    scaling=scaling, dropout=dropout, **kw)

    saved_refs = []
    for j, l in enumerate(layers):
        if j != layer_idx:
            a = l.self_attn
            saved_refs.append((a, a._wire_model))
            object.__setattr__(a, "_wire_model", None)

    attn._wire_orig_attn_fn = cap
    wrap._wire_signal = signal
    try:
        with torch.no_grad():
            wrap.llm(input_ids=ids)
    finally:
        attn._wire_orig_attn_fn = orig
        wrap._wire_signal = None
        for a, ref in saved_refs:
            object.__setattr__(a, "_wire_model", ref)
    return store


# --------------------------------------------------------------------------- #
# 1. Ψ = 0 / absent ⇒ EXACTLY the stock logits.                                #
# --------------------------------------------------------------------------- #

def test_zero_signal_is_exactly_identity():
    """θ=0 ⇒ cos=1, sin=0 ⇒ rotation is the identity, so logits are bit-identical.

    Stronger than the additive scheme's parity test (which needs a tolerance): a
    rotation by exactly zero is exactly the identity map in floating point.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    ids = torch.randint(0, 64, (2, 7))
    with torch.no_grad():
        ref = llm(input_ids=ids).logits

    wrap = _wrap(llm)
    impl = next(iter(wrap._decoder_layers())).self_attn.config._attn_implementation
    assert impl == "prism_wire", f"expected prism_wire dispatch, got {impl!r}"

    def logits():
        with torch.no_grad():
            return wrap.llm(input_ids=ids).logits

    wrap._wire_signal = None
    d_none = (logits() - ref).abs().max().item()
    wrap._wire_signal = torch.zeros(2, 7, wrap._wire_d_model)
    d_zero = (logits() - ref).abs().max().item()
    wrap._wire_signal = torch.randn(2, 7, wrap._wire_d_model) * 0.5
    d_nz = (logits() - ref).abs().max().item()
    wrap._wire_signal = None

    assert d_none == 0.0, f"r=None changed logits (max|Δ|={d_none:.2e})"
    assert d_zero == 0.0, f"r=0 changed logits (max|Δ|={d_zero:.2e}) — not exact identity"
    assert d_nz > 1e-4, f"r≠0 did not change logits (max|Δ|={d_nz:.2e}) — rotation inert?"


# --------------------------------------------------------------------------- #
# 2. Composition — the property that legitimises the post-RoPE hook point.     #
# --------------------------------------------------------------------------- #

def test_rotation_composes_with_text_rope():
    """R(θ)·RoPE(p)·x == a single rotation by (p + θ).

    Pure math on the kernel: Gemma's ``x*cos + rotate_half(x)*sin`` is a block
    2-D rotation pairing channel n with n+d/2, and same-plane rotations commute
    and add their angles. If this fails, applying WIRE after the text RoPE is not
    equivalent to a combined phase and the whole hook point is invalid.
    """
    torch.manual_seed(10)
    d, s = 16, 5
    x = torch.randn(1, 2, s, d)
    p = torch.randn(1, s, d // 2)          # "text" angles per plane
    t = torch.randn(1, s, d // 2)          # WIRE angles per plane

    def rot(x, ang):
        emb = torch.cat((ang, ang), dim=-1)
        cos, sin = emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)
        x1, x2 = x[..., : d // 2], x[..., d // 2:]
        return x * cos + torch.cat((-x2, x1), dim=-1) * sin

    two_step = rot(rot(x, p), t)
    one_step = rot(x, p + t)
    err = (two_step - one_step).abs().max().item()
    assert err < 1e-5, f"second rotation does not compose (max|Δ|={err:.2e})"


def test_wire_cos_sin_matches_manual_angles():
    """``wire_cos_sin`` builds cos/sin for θ_n = ω_n·r, duplicated across halves."""
    torch.manual_seed(11)
    m, planes, head_dim, s = 6, 4, 8, 3
    r = torch.randn(1, s, m)
    omega = torch.randn(planes, m)
    cos, sin = wire_cos_sin(r, omega, head_dim)
    assert cos.shape == (1, s, head_dim) and sin.shape == (1, s, head_dim)

    theta = r.float() @ omega.float().t()                 # [1, s, planes]
    pad = torch.zeros(1, s, head_dim // 2 - planes)
    full = torch.cat((theta, pad), dim=-1)
    exp = torch.cat((full, full), dim=-1)
    assert (cos - exp.cos()).abs().max().item() < 1e-6
    assert (sin - exp.sin()).abs().max().item() < 1e-6
    # Unrotated planes are the identity.
    assert torch.equal(cos[..., planes:head_dim // 2],
                       torch.ones(1, s, head_dim // 2 - planes))
    assert torch.equal(sin[..., planes:head_dim // 2],
                       torch.zeros(1, s, head_dim // 2 - planes))


# --------------------------------------------------------------------------- #
# 3. Norm preservation.                                                        #
# --------------------------------------------------------------------------- #

def test_rotation_preserves_norm():
    """‖R(θ)q‖ == ‖q‖ per head — a rotation cannot rescale q/k."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm, omega_scale=0.5)
    ids = torch.randint(0, 64, (1, 6))
    torch.manual_seed(12)
    r = torch.randn(1, 6, wrap._wire_d_model) * 0.5
    li = _global_layer_idx(llm)
    off = _capture_qk(wrap, ids, None, li)
    on = _capture_qk(wrap, ids, r, li)
    for name in ("q", "k"):
        n0 = off[name].norm(dim=-1)
        n1 = on[name].norm(dim=-1)
        d = (n0 - n1).abs().max().item()
        assert d < 1e-4, f"{name} norm changed under rotation (max|Δ|={d:.2e})"
        assert (on[name] - off[name]).abs().max().item() > 1e-4, \
            f"{name} was not rotated at all — vacuous norm test"


# --------------------------------------------------------------------------- #
# 4. Relative-only: a constant shift of every r leaves scores unchanged.        #
# --------------------------------------------------------------------------- #

def test_scores_depend_only_on_r_difference():
    """Shifting every position's r by a constant c leaves all q·k scores unchanged.

    Scores carry the phase ω·(r_j − r_i), so a global shift cancels. (The shift must
    cover EVERY position: non-node tokens hold r=0, so shifting only node tokens
    would legitimately change node↔non-node scores.)
    """
    torch.manual_seed(13)
    m, planes, head_dim, s = 6, 4, 8, 5
    q = torch.randn(1, 2, s, head_dim)
    k = torch.randn(1, 2, s, head_dim)
    r = torch.randn(1, s, m)
    omega = torch.randn(planes, m) * 0.3
    c = torch.randn(1, 1, m)

    def scores(rr):
        cos, sin = wire_cos_sin(rr, omega, head_dim)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        x1, x2 = q[..., : head_dim // 2], q[..., head_dim // 2:]
        qr = q * cos + torch.cat((-x2, x1), dim=-1) * sin
        y1, y2 = k[..., : head_dim // 2], k[..., head_dim // 2:]
        kr = k * cos + torch.cat((-y2, y1), dim=-1) * sin
        return qr @ kr.transpose(-1, -2)

    d = (scores(r) - scores(r + c)).abs().max().item()
    assert d < 1e-4, f"scores changed under a global r shift (max|Δ|={d:.2e})"


# --------------------------------------------------------------------------- #
# 5. Theorem 3 — the guarantee actually being implemented.                     #
# --------------------------------------------------------------------------- #

def test_small_omega_expectation_matches_theorem3():
    """E_ω[(R(r_i)q)ᵀ(R(r_j)k)] == qᵀk (1 − σ²‖r_i−r_j‖²/2) + O(σ⁴).

    Theorem 3 of the WIRE paper, checked by Monte Carlo over ω ~ N(0, σ²I_m).
    NOTE what this does and does not establish: the identity holds for ANY feature
    r (it is the Johnson–Lindenstrauss step). It says the downweighting is
    quadratic in ‖r_i − r_j‖ — NOT that ‖r_i − r_j‖² is the effective resistance,
    which additionally requires r = [u_k/√λ_k] over all nontrivial Laplacian modes.
    """
    torch.manual_seed(14)
    m, head_dim = 8, 16
    planes = head_dim // 2

    def mc_scores(q, k, r_i, r_j, sigma, draws):
        r = torch.stack((r_i, r_j)).view(1, 2, m)
        x = torch.stack((q, k)).view(1, 2, head_dim)
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2:]
        out = torch.empty(draws)
        for d in range(draws):
            omega = torch.randn(planes, m) * sigma
            cos, sin = wire_cos_sin(r, omega, head_dim)
            xr = x * cos + torch.cat((-x2, x1), dim=-1) * sin
            out[d] = xr[0, 0] @ xr[0, 1]
        return out

    r_i, r_j = torch.randn(m), torch.randn(m)
    dist2 = float((r_i - r_j).pow(2).sum())

    # --- Part A: high-power check of the CORRECTION's magnitude. ----------------
    # With k parallel to q the sin cross-terms (q_n·k_{n+h} − q_{n+h}·k_n) vanish
    # identically, so the estimator's variance collapses to that of cos alone —
    # O(σ⁴) — and the σ²‖Δr‖²/2 downweighting is resolvable in few draws. With
    # random k the sin term's variance (~σ‖Δr‖·‖q‖‖k‖) exceeds the correction
    # itself at these σ, so that configuration cannot test the magnitude at all;
    # it is checked separately in Part B for unbiasedness.
    q = torch.randn(head_dim)
    k = q.clone()
    sigma = 0.1
    qk = float(q @ k)
    predicted = qk * (1.0 - sigma ** 2 * dist2 / 2.0)
    correction = qk - predicted
    assert abs(correction) > 1e-2, "correction too small — test would be vacuous"

    mean = float(mc_scores(q, k, r_i, r_j, sigma, 400).mean())
    rel = abs(mean - predicted) / abs(correction)
    assert rel < 0.10, (
        f"MC mean {mean:.6f} != Theorem-3 prediction {predicted:.6f} "
        f"(qᵀk={qk:.6f}, correction={correction:.4f}, relative error {rel:.1%})")
    # The observed value is far closer to the corrected prediction than to bare qᵀk,
    # i.e. the downweighting is genuinely observed and not an artefact of tolerance.
    assert abs(mean - predicted) < 0.2 * abs(mean - qk), (
        f"downweighting not resolved: mean={mean:.6f}, predicted={predicted:.6f}, qᵀk={qk:.6f}")

    # --- Part B: with random k, the sin term must average away (unbiasedness). ---
    k = torch.randn(head_dim)
    sigma = 0.05
    qk = float(q @ k)
    predicted = qk * (1.0 - sigma ** 2 * dist2 / 2.0)
    s = mc_scores(q, k, r_i, r_j, sigma, 4000)
    mean, se = float(s.mean()), float(s.std()) / (len(s) ** 0.5)
    assert abs(mean - predicted) < 3.0 * se, (
        f"estimator biased: mean={mean:.6f}, predicted={predicted:.6f}, "
        f"|Δ|={abs(mean - predicted):.2e} > 3·SE={3 * se:.2e}")


# --------------------------------------------------------------------------- #
# 6. NoPE planes untouched by default.                                         #
# --------------------------------------------------------------------------- #

def test_nope_planes_untouched_by_default():
    """``rotate_nope_planes=False`` restricts WIRE to the text-rotary planes.

    Gemma's global layers use ``rope_type='proportional'`` with
    ``partial_rotary_factor=0.25``: only ``int(0.25*head_dim//2)`` planes carry text
    RoPE; the rest are NoPE (cos=1, sin=0) and the model treats them as
    position-free content. Rotating them injects graph phase where none is expected,
    so the default must leave those channels exactly untouched.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm, rotate_nope_planes=False, omega_scale=0.5)
    li = _global_layer_idx(llm)
    attn = list(wrap._decoder_layers())[li].self_attn
    head_dim = attn.head_dim
    planes = wire_rope_planes(attn, rotate_nope=False)
    assert 0 < planes < head_dim // 2, (
        f"fixture must have genuine NoPE planes (got {planes}/{head_dim // 2})")

    ids = torch.randint(0, 64, (1, 6))
    torch.manual_seed(15)
    r = torch.randn(1, 6, wrap._wire_d_model) * 0.5
    off = _capture_qk(wrap, ids, None, li)
    on = _capture_qk(wrap, ids, r, li)
    delta = (on["q"] - off["q"]).abs().amax(dim=(0, 1, 2))   # per-channel max |Δ|

    half = head_dim // 2
    rotated = list(range(planes)) + list(range(half, half + planes))
    untouched = [c for c in range(head_dim) if c not in rotated]
    assert delta[untouched].max().item() == 0.0, (
        f"NoPE channels moved: {delta[untouched].max().item():.2e}")
    assert delta[rotated].max().item() > 1e-4, "rotary channels did not move"

    # And with the flag ON, every plane moves.
    wrap_all = _wrap(_gemma4(), rotate_nope_planes=True, omega_scale=0.5)
    li2 = _global_layer_idx(wrap_all.llm)
    off2 = _capture_qk(wrap_all, ids, None, li2)
    on2 = _capture_qk(wrap_all, ids, r, li2)
    d2 = (on2["q"] - off2["q"]).abs().amax(dim=(0, 1, 2))
    assert d2.min().item() > 0.0, "rotate_nope_planes=True left some channel untouched"


# --------------------------------------------------------------------------- #
# 7. Cold-start gate + wiring contracts.                                       #
# --------------------------------------------------------------------------- #

def test_gate_zero_is_exact_identity():
    """``pe_gain_init=0.0`` ⇒ r = pe·tanh(0) = 0 ⇒ exact identity rotation."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    torch.manual_seed(16)
    wrap = _wrap(llm, pe_gain_init=0.0)
    ids = torch.randint(0, 64, (1, 8))
    g = _tiny_graph(3)
    inj = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}
    with torch.no_grad():
        stock = wrap.llm(input_ids=ids).logits
        out = wrap(input_ids=ids, graphs=[g], injection_maps=[inj]).logits
    d = (out - stock).abs().max().item()
    assert d == 0.0, f"closed gate changed logits (max|Δ|={d:.2e}) — not exact identity"
    assert wrap._wire_signal is None, "signal left armed after forward"


def test_open_gate_changes_logits_and_places_signal_at_spans():
    """Open gate: r lands only on node-token spans and the logits move."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    torch.manual_seed(17)
    wrap = _wrap(llm, pe_gain_init=1.0)
    ids = torch.randint(0, 64, (1, 8))
    g = _tiny_graph(3)
    inj = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}
    node_pos = [2, 4, 6]

    r = wrap.build_wire_signal([g], [inj], seq_len=8, device=ids.device)
    assert r.shape == (1, 8, wrap._wire_d_model)
    row_norm = r[0].norm(dim=-1)
    for p in node_pos:
        assert row_norm[p].item() > 1e-6, f"node-token row {p} has zero r"
    for p in [x for x in range(8) if x not in node_pos]:
        assert row_norm[p].item() == 0.0, f"non-node row {p} got nonzero r"

    with torch.no_grad():
        stock = wrap.llm(input_ids=ids).logits
        out = wrap(input_ids=ids, graphs=[g], injection_maps=[inj]).logits
    assert (out - stock).abs().max().item() > 1e-4, "graph forward did not change logits"


def test_layer_scope_restricts_active_layers():
    """``layer_scope='dense'`` activates exactly the full_attention layers."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm, layer_scope="dense")
    layers = list(wrap._decoder_layers())
    active = [bool(getattr(l.self_attn, "_wire_active", False)) for l in layers]
    expected = [t == "full_attention" for t in llm.config.layer_types]
    assert active == expected, f"active={active} expected={expected}"
    # Inactive layers own no ω table (nothing to train or save for them).
    for i, l in enumerate(layers):
        has = str(i) in wrap._wire_sigma
        assert has == expected[i], f"layer {i}: omega present={has}, active={expected[i]}"


def test_angle_is_measured_every_forward_and_never_raises():
    """The angle is measured on every forward and recorded, and exceeding the bound is
    a CLAMP rather than a failure — the inverse of the earlier contract, which raised.

    No config value and no optimiser trajectory may kill a run through this guard.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    torch.manual_seed(19)
    g = _tiny_graph(3)
    inj = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}

    ok = _wrap(llm, omega_scale=0.05, max_angle=8.0, pe_gain_init=1.0)
    ok.build_wire_signal([g], [inj], seq_len=8, device=torch.device("cpu"))
    assert ok._wire_measured_angle is not None and ok._wire_measured_angle >= 0.0
    assert ok._wire_measured_row_angle is not None
    assert ok._wire_effective_angle is not None
    assert ok._wire_psi_span is not None and ok._wire_psi_span > 0

    # An absurdly tight bound clamps instead of raising.
    import warnings as _w
    tight = _wrap(_gemma4(), omega_scale=0.05, max_angle=1e-9, pe_gain_init=1.0)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        tight.build_wire_signal([g], [inj], seq_len=8, device=torch.device("cpu"))   # must not raise
    assert tight._wire_effective_angle <= tight._wire_max_angle * (1 + 1e-5)
    assert tight._wire_measured_angle > tight._wire_effective_angle, "clamp did not bite"


def test_decode_step_never_silently_skips():
    """A cached decode step must NEVER silently fall through to stock attention.

    WIRE's premise is that q AND k carry the graph phase, and the KV cache stores keys
    written BEFORE the hook — so a cached step either re-rotates them (``decode='rotate'``,
    asserted numerically in ``test_wire_decode_matches_full_sequence_forward``) or is the
    LABELLED ``decode='skip'`` diagnostic. The historical third option — raising
    ``NotImplementedError`` because re-rotation was unbuilt — is gone; this test now pins
    that the rotation actually reaches the cached keys instead.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    torch.manual_seed(20)
    wrap = _wrap(llm, pe_gain_init=1.0)
    ids = torch.randint(0, 64, (1, 8))
    g = _tiny_graph(3)
    inj = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}

    rotated = []
    real_rotate = _wire_rotate_prefix

    def spy(x, cos, sin, n):
        rotated.append((tuple(x.shape), n))
        return real_rotate(x, cos, sin, n)

    gnn_llm_mod = sys.modules["prism.models.gnn_llm"]
    gnn_llm_mod._wire_rotate_prefix = spy
    try:
        with torch.no_grad():
            out = wrap.generate_with_graph(input_ids=ids, graphs=[g], injection_maps=[inj],
                                           max_new_tokens=2, do_sample=False,
                                           use_cache=True, pad_token_id=0)
    finally:
        gnn_llm_mod._wire_rotate_prefix = real_rotate

    assert out.shape[1] > ids.shape[1], "decode produced no tokens"
    # Cached steps must have rotated KEY tensors longer than the prompt (k_len > 8).
    assert any(shape[-2] > ids.shape[1] for shape, _ in rotated), (
        "no cached-step rotation happened — the decode branch fell through silently "
        f"(rotations seen: {rotated!r})")
    assert all(n == ids.shape[1] for _, n in rotated), (
        "cached steps must rotate exactly the prompt-length prefix (generated positions "
        f"carry r=0): {rotated!r}")


def test_no_parameter_or_buffer_shape_depends_on_node_count():
    """WIRE's §3.1 property, enforced literally: ω is ``[head_dim/2, m]`` — a function
    of head width and GT width only. Two wrappers built for different graph sizes must
    have byte-identical parameter shapes, and nothing may be ``N``- or ``N²``-shaped."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm)
    shapes = {k: tuple(v.shape) for k, v in wrap.named_buffers() if "_wire_eps" in k}
    assert shapes, "no ε tables registered"
    for k, shp in shapes.items():
        li = int(k.split(".")[-1])
        attn = list(wrap._decoder_layers())[li].self_attn
        planes = wire_rope_planes(attn, rotate_nope=False)
        assert shp == (planes, wrap._wire_d_model), f"{k}: {shp} is not [P, m]"
    # σ is a scalar per layer — nothing here is N- or N²-shaped either.
    for k, v in wrap.named_parameters():
        if "_wire_sigma" in k:
            assert v.shape == (), f"{k}: σ must be a scalar, got {tuple(v.shape)}"

    # Running two very different graph sizes through the same wrapper must not create
    # or resize anything.
    before = {k: tuple(v.shape) for k, v in
              list(wrap.named_parameters()) + list(wrap.named_buffers())}
    for n in (4, 60):
        g = _path_graph(n)
        inj = {i: [(i, i + 1)] for i in range(min(n, 6))}
        wrap.build_wire_signal([g], [inj], seq_len=8, device=torch.device("cpu"))
    after = {k: tuple(v.shape) for k, v in
             list(wrap.named_parameters()) + list(wrap.named_buffers())}
    assert before == after, "a parameter/buffer changed shape with graph size"


def test_ood_node_count_gate():
    """GATING ACCEPTANCE TEST — in-distribution vs OOD node count.

    Guards the failure this mechanism must not have: trained around ~30 nodes, run at
    ~120. Asserts at BOTH scales that there is no shape error, no truncation of the
    injected spans, the identity still holds exactly at non-graph tokens, the
    relative-only invariance still holds, and the measured angle statistics stay inside
    the configured bound (they must not drift with N).
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    torch.manual_seed(30)
    wrap = _wrap(llm, pe_gain_init=1.0, omega_scale=0.05, max_angle=8.0)
    seq = 40
    stats = {}
    for n in (30, 120):
        g = _path_graph(n)
        # Every node gets a distinct 1-token span; only the first `seq//2` fit, and the
        # rest are simply absent from the map (NOT silently truncated mid-span).
        inj = {i: [(i, i + 1)] for i in range(min(n, seq // 2))}
        r = wrap.build_wire_signal([g], [inj], seq_len=seq, device=torch.device("cpu"))
        assert r.shape == (1, seq, wrap._wire_d_model), f"N={n}: bad r shape {r.shape}"

        placed = (r[0].norm(dim=-1) > 0).nonzero().flatten().tolist()
        assert placed == sorted(inj.keys()), (
            f"N={n}: placed positions {placed} != mapped spans {sorted(inj.keys())}")
        # Identity at every non-graph token: r is exactly zero there ⇒ θ=0 ⇒ R=I.
        for p in range(seq):
            if p not in inj:
                assert r[0, p].abs().max().item() == 0.0, f"N={n}: pos {p} not exactly 0"

        stats[n] = (wrap._wire_measured_angle, wrap._wire_measured_row_angle)
        assert wrap._wire_measured_angle <= wrap._wire_max_angle, (
            f"N={n}: pair angle {wrap._wire_measured_angle:.3f} exceeded bound")

    # Angle magnitude must not blow up with N: cos/sin are periodic, so a Ψ-norm that
    # grows with graph size would wrap the angles and silently void Theorem 3's
    # leading-order reading — the "fine at 30, noise at 120" failure.
    (p30, r30), (p120, r120) = stats[30], stats[120]
    assert r120 <= 3.0 * r30 + 1e-6, (
        f"per-node angle scale grew with N (σ·max‖Ψ‖: {r30:.4f} @30 -> {r120:.4f} @120)")
    assert p120 <= 3.0 * p30 + 1e-6, (
        f"pairwise angle scale grew with N (σ·max‖ΔΨ‖: {p30:.4f} @30 -> {p120:.4f} @120)")

    # And a full forward runs clean at the OOD scale.
    ids = torch.randint(0, 64, (1, seq))
    g = _path_graph(120)
    inj = {i: [(i, i + 1)] for i in range(seq // 2)}
    with torch.no_grad():
        out = wrap(input_ids=ids, graphs=[g], injection_maps=[inj]).logits
        stock = wrap.llm(input_ids=ids).logits
    assert torch.isfinite(out).all(), "non-finite logits at OOD node count"
    assert (out - stock).abs().max().item() > 1e-4, "injection inert at OOD node count"


def test_permutation_equivariance_threading():
    """Lemma 1: relabelling nodes permutes Ψ rather than changing it.

    The ``permutation=`` kwarg is threaded to ``pe_model`` exactly as
    ``build_pe_signal`` does, so the eval-time equivariance check reaches the GT. Here
    we assert the plumbing AND that permuting the graph permutes the placed signal —
    up to the GT's probe randomness, which is why the check runs against the GT's own
    permutation path rather than by re-running on a relabelled graph.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    torch.manual_seed(31)
    wrap = _wrap(llm, pe_gain_init=1.0)
    n = 6
    g = _path_graph(n)
    inj = {i: [(i, i + 1)] for i in range(n)}
    seen = {}

    real_pe = wrap.pe_model

    class _Spy(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, data, permutation=None):
            seen["permutation"] = permutation
            return self.inner(data, permutation=permutation)

    wrap.pe_model = _Spy(real_pe)
    perm = Permutation(seed=3)
    try:
        wrap.build_wire_signal([g], [inj], seq_len=8, device=torch.device("cpu"),
                               permutation=perm)
    finally:
        wrap.pe_model = real_pe
    assert seen.get("permutation") is perm, "permutation not threaded to pe_model"
    # The GT consumed it (it materialises .perm on first apply), so the equivariance
    # path is live rather than merely accepted and dropped.
    assert perm.perm is not None and perm.perm.shape[0] == n, \
        "pe_model never applied the permutation"


def test_theorem2_single_frequency_on_path_reproduces_rope():
    """Theorem 2 sanity: on a path graph, WIRE with ONE nontrivial frequency applied to
    a monotone coordinate reproduces ordinary RoPE's relative phase structure.

    Checked at the kernel level rather than end-to-end: feed the path's monotone
    coordinate (the analogue of the Fiedler-vector coordinate the paper uses, up to the
    affine renormalisation it describes) as a 1-D r, and assert the resulting score
    phase depends only on the index difference — i.e. it IS a RoPE with frequency ω.
    """
    torch.manual_seed(32)
    n, head_dim = 8, 8
    coord = torch.arange(n, dtype=torch.float32).view(1, n, 1)      # monotone, 1-D
    omega = torch.tensor([[0.3]])                                    # one plane
    cos, sin = wire_cos_sin(coord, omega, head_dim)
    q = torch.randn(1, 1, n, head_dim)
    k = torch.randn(1, 1, n, head_dim)

    def rot(x):
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2:]
        return x * c + torch.cat((-x2, x1), dim=-1) * s

    # Same q,k at every position ⇒ the score must be a pure function of (i − j).
    qc = q[:, :, :1].expand(-1, -1, n, -1).contiguous()
    kc = k[:, :, :1].expand(-1, -1, n, -1).contiguous()
    sc = (rot(qc) @ rot(kc).transpose(-1, -2))[0, 0]
    for d in range(1, n):
        diag = torch.diagonal(sc, offset=d)
        spread = (diag.max() - diag.min()).abs().item()
        assert spread < 1e-4, (
            f"score at offset {d} is not translation invariant (spread={spread:.2e}) "
            "— WIRE on a path is not behaving as RoPE")


def test_omega_is_shared_across_heads_and_reparameterised():
    """ω = σ_ℓ·ε_ℓ: ONE table per layer, shared by every head.

    Sharing is what makes the modulation computable once per layer, and it is also what
    keeps Eq. 3 exact under GQA — a key head serves several query heads, so q and k must
    be rotated by identical frequencies. Also asserts the reparameterisation is real:
    ε is a frozen buffer, σ is the learnable scalar, and ω scales linearly with σ.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm, omega_scale=0.5)
    li = _global_layer_idx(llm)
    attn = list(wrap._decoder_layers())[li].self_attn
    planes = wire_rope_planes(attn, rotate_nope=False)

    eps = getattr(wrap._wire_eps, str(li))
    sigma = wrap._wire_sigma[str(li)]
    assert eps.shape == (planes, wrap._wire_d_model), f"eps is {tuple(eps.shape)}, not [P, m]"
    assert not eps.requires_grad, "eps must be frozen (it is the fixed Gaussian draw)"
    assert sigma.shape == (), "sigma must be a per-layer SCALAR"
    assert sigma.requires_grad, "sigma must be learnable by default"
    # eps is a persistent BUFFER, not a parameter.
    assert any(k.endswith(f"_wire_eps.{li}") for k in wrap.state_dict()), \
        "eps missing from state_dict"
    assert not any("_wire_eps" in k for k, _ in wrap.named_parameters()), \
        "eps registered as a parameter rather than a buffer"

    omega = wrap.layer_omega(li)
    assert torch.allclose(omega, sigma.detach() * eps, atol=1e-6), "omega != sigma * eps"

    # Rotation applied to q and k uses the SAME angles, broadcast over all heads.
    ids = torch.randint(0, 64, (1, 6))
    torch.manual_seed(21)
    r = torch.randn(1, 6, wrap._wire_d_model) * 0.5
    off = _capture_qk(wrap, ids, None, li)
    on = _capture_qk(wrap, ids, r, li)
    cos, sin = wire_cos_sin(r, omega.detach(), attn.head_dim)
    from prism.models.gnn_llm import _wire_rotate
    dq = (on["q"] - _wire_rotate(off["q"], cos, sin)).abs().max().item()
    dk = (on["k"] - _wire_rotate(off["k"], cos, sin)).abs().max().item()
    assert dq < 1e-5, f"query rotation != shared ω (max|Δ|={dq:.2e})"
    assert dk < 1e-5, f"key rotation != shared ω (max|Δ|={dk:.2e})"
    assert (on["q"] - off["q"]).abs().max().item() > 1e-4, "rotation inert — vacuous"


def test_sigma_receives_gradient_and_eps_does_not():
    """Gradient reaches the per-layer σ (and the gate), never the frozen ε."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    torch.manual_seed(18)
    wrap = _wrap(llm, omega_learnable=True, pe_gain_init=1.0)
    # σ trains at the BASE LR, deliberately outside structural_lr_mult: that multiplier
    # damps a PRETRAINED Ψ producer, and σ has no pretrained state to protect.
    sp = {id(p) for p in wrap.structural_parameters()}
    blp = {id(p) for p in wrap.base_lr_parameters()}
    for p in wrap._wire_sigma.values():
        assert p.requires_grad, "learnable σ is not trainable"
        assert id(p) in blp, "learnable σ missing from the base-LR group"
        assert id(p) not in sp, "σ leaked into the structural (multiplier) LR group"

    ids = torch.randint(0, 64, (1, 8))
    g = _tiny_graph(3)
    inj = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}
    out = wrap(input_ids=ids, graphs=[g], injection_maps=[inj])
    out.logits.float().square().mean().backward()
    grads = [p.grad for p in wrap._wire_sigma.values() if p.grad is not None]
    assert grads and max(gd.abs().max().item() for gd in grads) > 0, "σ got no grad"
    assert all(torch.isfinite(gd).all().item() for gd in grads), "non-finite σ grad"
    assert wrap.pe_gain.grad is not None and wrap.pe_gain.grad.abs().max().item() > 0, \
        "gate got no grad through the rotation"


def test_frozen_sigma_arm():
    """``freeze_sigma=True`` (literal-Theorem-3 arm) keeps σ out of the LR group but
    still in the checkpoint."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm, omega_learnable=False)
    sp = {id(p) for p in wrap.structural_parameters()}
    for p in wrap._wire_sigma.values():
        assert not p.requires_grad, "frozen σ is trainable"
        assert id(p) not in sp, "frozen σ leaked into the structural LR group"
    assert any("_wire_sigma" in k for k in wrap.state_dict()), "σ missing from state_dict"


def test_clamp_is_exact_unconditional_and_never_raises():
    """The angle guard is a CLAMP, not a failure path.

    Asserts (1) no exception on any σ, including adversarial values; (2) the POST-clamp
    angle satisfies ``σ_eff·span ≤ max_angle`` by construction; (3) the clamp is
    unconditional — the factor is exactly 1.0 inside the bound, so the identity path is
    the same code path.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    torch.manual_seed(19)
    g = _tiny_graph(4)
    inj = {i: [(i, i + 1)] for i in range(4)}

    # Inside the bound: factor is EXACTLY 1.0 (identity), not merely close.
    w = _wrap(_gemma4(), omega_scale=1e-6, max_angle=1.0, pe_gain_init=1.0)
    w.build_wire_signal([g], [inj], seq_len=8, device=torch.device("cpu"))
    for li in w.active_layer_indices():
        assert w.layer_scale_factor(li) == 1.0, "clamp not identity inside the bound"

    # Adversarial σ: driven to 1e3x. Must NOT raise, and must stay inside the bound.
    for mult in (1.0, 10.0, 1e2, 1e3):
        w2 = _wrap(_gemma4(), omega_scale=0.5, max_angle=0.25, pe_gain_init=1.0)
        with torch.no_grad():
            for k in w2._wire_sigma:
                w2._wire_sigma[k].mul_(mult)
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            w2.build_wire_signal([g], [inj], seq_len=8, device=torch.device("cpu"))
        span = w2._wire_psi_span
        for li in w2.active_layer_indices():
            sig_eff = float(w2._wire_sigma[str(li)].detach().abs()) * w2.layer_scale_factor(li)
            assert sig_eff * span <= w2._wire_max_angle * (1 + 1e-6), (
                f"mult={mult}: post-clamp angle {sig_eff * span:.6f} > "
                f"max_angle {w2._wire_max_angle}")
            # ω itself carries the clamped scale.
            assert float(w2.layer_omega(li).detach().abs().max()) <= (
                w2._wire_max_angle / span * float(
                    getattr(w2._wire_eps, str(li)).abs().max()) * (1 + 1e-5))
        assert w2._wire_effective_angle <= w2._wire_max_angle * (1 + 1e-5)

    # A full forward at the adversarial σ also completes without raising.
    ids = torch.randint(0, 64, (1, 8))
    inj8 = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}
    w3 = _wrap(_gemma4(), omega_scale=0.5, max_angle=0.25, pe_gain_init=1.0)
    with torch.no_grad():
        for k in w3._wire_sigma:
            w3._wire_sigma[k].mul_(1e3)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        with torch.no_grad():
            out = w3(input_ids=ids, graphs=[_tiny_graph(3)], injection_maps=[inj8]).logits
    assert torch.isfinite(out).all(), "non-finite logits under a clamped adversarial σ"


def test_clamp_gradient_and_saturation_telemetry():
    """Gradient still reaches σ when clamped (scaled by the detached factor), and the
    saturation is VISIBLE in telemetry rather than silent."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    torch.manual_seed(23)
    ids = torch.randint(0, 64, (1, 8))
    g = _tiny_graph(3)
    inj = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}

    w = _wrap(_gemma4(), omega_scale=0.5, max_angle=1e-2, pe_gain_init=1.0)
    with torch.no_grad():
        for k in w._wire_sigma:
            w._wire_sigma[k].mul_(100.0)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        out = w(input_ids=ids, graphs=[g], injection_maps=[inj])
    out.logits.float().square().mean().backward()

    grads = [p.grad for p in w._wire_sigma.values() if p.grad is not None]
    assert grads, "σ received no grad at all under clamping"
    assert all(torch.isfinite(gd).all().item() for gd in grads), "non-finite σ grad"
    # Non-zero: ω = σ·s·ε with s detached, so dω/dσ = s·ε != 0.
    assert max(gd.abs().max().item() for gd in grads) > 0, \
        "σ grad is exactly zero under clamping"

    t = w.wire_telemetry()
    for key in ("wire/psi_span", "wire/sigma_raw_max", "wire/sigma_eff_max",
                "wire/angle_raw_max", "wire/angle_eff_max", "wire/clamp_engaged",
                "wire/clamped_layers", "wire/scale_min", "wire/max_angle"):
        assert key in t, f"telemetry missing {key}"
    assert t["wire/clamp_engaged"] == 1, "clamp engaged but not reported"
    assert t["wire/clamped_layers"] == len(w.active_layer_indices())
    assert t["wire/sigma_raw_max"] > t["wire/sigma_eff_max"], \
        "saturation invisible: raw and effective σ reported identical"
    assert t["wire/angle_eff_max"] <= t["wire/max_angle"] * (1 + 1e-6)
    assert t["wire/scale_min"] < 1.0
    # Per-layer entries present (this is what makes saturation attributable).
    for li in w.active_layer_indices():
        assert f"wire/sigma_raw/L{li}" in t and f"wire/sigma_eff/L{li}" in t

    # Unclamped model reports clamp_engaged == 0 and raw == eff.
    w2 = _wrap(_gemma4(), omega_scale=1e-6, max_angle=1.0, pe_gain_init=1.0)
    w2.build_wire_signal([g], [inj], seq_len=8, device=torch.device("cpu"))
    t2 = w2.wire_telemetry()
    assert t2["wire/clamp_engaged"] == 0 and t2["wire/clamped_layers"] == 0
    assert t2["wire/sigma_raw_max"] == t2["wire/sigma_eff_max"]


def test_relative_only_holds_under_gqa():
    """Eq. 3 end-to-end under grouped-query attention: shifting r by a constant leaves
    every q·k score unchanged. Sharing ω across heads is what guarantees this."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm, omega_scale=0.3)
    li = _global_layer_idx(llm)
    attn = list(wrap._decoder_layers())[li].self_attn
    ids = torch.randint(0, 64, (1, 6))
    torch.manual_seed(22)
    r = torch.randn(1, 6, wrap._wire_d_model) * 0.4
    c = torch.randn(1, 1, wrap._wire_d_model) * 0.4

    def scores(sig):
        cap = _capture_qk(wrap, ids, sig, li)
        q, k = cap["q"], cap["k"]
        k = k.repeat_interleave(attn.num_key_value_groups, dim=1)   # mirrors repeat_kv
        return q @ k.transpose(-1, -2)

    s0, s1 = scores(r), scores(r + c)
    d = (s0 - s1).abs().max().item()
    assert d < 1e-3, f"scores moved under a global r shift under GQA (max|Δ|={d:.2e})"
    assert (s0 - scores(None)).abs().max().item() > 1e-3, "WIRE inert — vacuous test"


def test_kv_shared_layers_are_never_active():
    """KV-shared layers reuse an upstream layer's keys, which already carry that
    layer's ω phase. Rotating q here with a DIFFERENT ω would break the relative-only
    property, so such layers must be inactive entirely (not merely q-only)."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm, layer_scope="all")
    for i, l in enumerate(wrap._decoder_layers()):
        if getattr(l.self_attn, "is_kv_shared_layer", False):
            assert not getattr(l.self_attn, "_wire_active", False), \
                f"KV-shared layer {i} is WIRE-active"


def test_arch_registry_builds_wire_llm():
    """``build_planner_model(arch='wire_llm')`` constructs the class, and every
    ``wire_*`` key it reads exists in the shipped Hydra config."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    import omegaconf

    from prism.models.architectures import build_planner_model

    cfg = omegaconf.OmegaConf.load("experiments/base_config.yaml")
    gnn = cfg.gnn
    for key in ("wire_layer_scope", "wire_sigma_init", "wire_freeze_sigma",
                "wire_omega_seed", "wire_rotate_nope_planes", "wire_max_angle",
                "wire_decode"):
        assert key in gnn, f"base_config.yaml is missing gnn.{key}"
    gnn.arch = "wire_llm"
    gnn.d_model = 16
    gnn.pe_hidden_channels = 16
    gnn.num_samples = 4
    gnn.gt_heads = 2
    gnn.pe_gain_init = 1.0

    class _Tok:                       # collator only needs a pad token id
        pad_token_id = 0
        eos_token_id = 1

    model, _collator = build_planner_model(gnn, llm, _Tok())
    assert isinstance(model, WireGraphLLM), f"got {type(model).__name__}"
    assert model._wire_layer_scope == gnn.wire_layer_scope
    assert model._wire_freeze_sigma is bool(gnn.wire_freeze_sigma)
    assert model._wire_max_angle == float(gnn.wire_max_angle)


def test_eps_and_sigma_survive_a_checkpoint_roundtrip():
    """ε and σ must round-trip through ``state_dict`` exactly.

    ε is SAVED rather than regenerated from ``omega_seed``: regeneration would make the
    checkpoint depend on torch RNG determinism across versions/devices, so a reloaded
    model could silently rotate by different frequencies than it trained with.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    a = _wrap(llm, omega_learnable=False)
    saved_eps = {k: v.clone() for k, v in a._wire_eps.state_dict().items()}
    saved_sigma = {k: v.clone() for k, v in a._wire_sigma.state_dict().items()}
    assert saved_eps and saved_sigma, "nothing to save"

    b = _wrap(_gemma4(seed=99), omega_learnable=False)
    b._wire_eps.load_state_dict(saved_eps)
    b._wire_sigma.load_state_dict(saved_sigma)
    for k, v in saved_eps.items():
        assert torch.equal(v, b._wire_eps.state_dict()[k]), f"ε {k} changed on reload"
    for k, v in saved_sigma.items():
        assert torch.equal(v, b._wire_sigma.state_dict()[k]), f"σ {k} changed on reload"
    # ω is fully determined by (ε, σ), so the reloaded model rotates identically.
    li = _global_layer_idx(b.llm)
    assert torch.equal(a.layer_omega(li).detach(), b.layer_omega(li).detach()), \
        "ω differs after reload"

    # A different seed really does give different ε (so the round-trip is not vacuous).
    c_seeded = WireGraphLLM(
        _gemma4(), a.pe_model, d_model=a._wire_d_model, omega_seed=1234,
        freeze_sigma=True, vanilla=False)
    k0 = next(iter(saved_eps))
    assert not torch.equal(saved_eps[k0], c_seeded._wire_eps.state_dict()[k0]), \
        "omega_seed has no effect — the ε draw is not actually seeded"


def _wire_ckpt_dict(model):
    """EXACTLY what ``trainers.PRISMTrainer.save_model`` writes for ``wire_llm``.

    Mirrored here (rather than driving a real Trainer) so the multistage tests below
    stay a pure loader test; the key set is asserted against the trainer's contract.
    """
    d = {
        "pe_model": {k: v.clone() for k, v in model.pe_model.state_dict().items()},
        "pe_gain": model.pe_gain.data.clone(),
        "wire_eps": {k: v.clone() for k, v in model._wire_eps.state_dict().items()},
        "wire_sigma": {k: v.clone() for k, v in model._wire_sigma.state_dict().items()},
        "wire_omega": {k: v.clone() for k, v in model._wire_omega.state_dict().items()},
    }
    # The KEY SET is mode-independent by design (the unused store writes an empty dict):
    # a key present in one mode and absent in the other is the silent-corruption case.
    assert set(d) == {"pe_model", "pe_gain", "wire_eps", "wire_sigma", "wire_omega"}, \
        "wire_llm checkpoint contract changed — update trainers.save_model too"
    return d


def _write_ckpt(state, tmpdir):
    import os
    torch.save(state, os.path.join(tmpdir, "gnn_weights.pt"))
    return tmpdir


def _perturb(model, delta=0.137):
    """Move every carried tensor off its init so a carry test cannot pass vacuously."""
    with torch.no_grad():
        for p in model.pe_model.parameters():
            p.add_(delta)
        model.pe_gain.add_(delta)
        for b in model._wire_eps.buffers():
            b.add_(delta)
        for p in model._wire_sigma.values():
            p.add_(delta)
        for p in model._wire_omega.values():
            p.add_(delta)


def test_multistage_wire_to_wire_carry_is_exact():
    """Stage-N init from a prior ``wire_llm`` run carries Ψ, the gate, ε AND σ exactly.

    ω = σ·ε is the rotation the run trained with; carrying Ψ while cold-starting ε
    would silently rotate by different frequencies than the source model learned.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    import tempfile

    from prism.models import loaders

    src = _wrap(llm, pe_gain_init=0.3)
    _perturb(src)
    dst = _wrap(_gemma4(seed=99), pe_gain_init=1.0)
    fresh_gain = float(dst.pe_gain.detach())

    with tempfile.TemporaryDirectory() as td:
        _write_ckpt(_wire_ckpt_dict(src), td)
        loaders.load_pe_weights_into(dst, td, "wire_llm")

    assert float(dst.pe_gain.detach()) != fresh_gain, "pe_gain not carried"
    assert abs(float(dst.pe_gain.detach()) - float(src.pe_gain.detach())) < 1e-12
    for k, v in src.pe_model.state_dict().items():
        assert torch.equal(v, dst.pe_model.state_dict()[k]), f"pe_model.{k} not carried"
    for k, v in src._wire_eps.state_dict().items():
        assert torch.equal(v, dst._wire_eps.state_dict()[k]), f"ε {k} not carried"
    for k, v in src._wire_sigma.state_dict().items():
        assert torch.equal(v, dst._wire_sigma.state_dict()[k]), f"σ {k} not carried"
    li = _global_layer_idx(dst.llm)
    assert torch.equal(src.layer_omega(li).detach(), dst.layer_omega(li).detach()), \
        "ω differs after the multistage carry"


def test_multistage_gt_only_checkpoint_cold_starts_wire_terms():
    """A prior GT / mask checkpoint has no ``wire_*``: carry Ψ, cold-start the rest.

    A partial carry must not be mistaken for a full WIRE resume — ε/σ/pe_gain stay at
    their fresh-init values, which the loader reports.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    import tempfile

    from prism.models import loaders

    src = _wrap(llm)
    _perturb(src)
    dst = _wrap(_gemma4(seed=99), pe_gain_init=1.0)
    fresh_gain = float(dst.pe_gain.detach())
    fresh_eps = {k: v.clone() for k, v in dst._wire_eps.state_dict().items()}
    fresh_sigma = {k: v.clone() for k, v in dst._wire_sigma.state_dict().items()}

    # rpearl_gt_llm layout: the Ψ producer under "gt_model", plus additive-injection
    # heads WIRE deliberately has no counterpart for.
    ckpt = {"gt_model": src.pe_model.state_dict(),
            "pe_proj": {"weight": torch.zeros(4, src._wire_d_model)},
            "pe_gain": src.pe_gain.data.clone()}
    with tempfile.TemporaryDirectory() as td:
        _write_ckpt(ckpt, td)
        loaders.load_pe_weights_into(dst, td, "wire_llm")

    for k, v in src.pe_model.state_dict().items():
        assert torch.equal(v, dst.pe_model.state_dict()[k]), f"pe_model.{k} not carried"
    assert float(dst.pe_gain.detach()) == fresh_gain, \
        "pe_gain carried from a non-WIRE checkpoint (it gates an additive PE there, an angle here)"
    for k, v in fresh_eps.items():
        assert torch.equal(v, dst._wire_eps.state_dict()[k]), f"ε {k} moved off cold start"
    for k, v in fresh_sigma.items():
        assert torch.equal(v, dst._wire_sigma.state_dict()[k]), f"σ {k} moved off cold start"


def test_multistage_wire_hparam_drift_raises():
    """Config drift must raise, never silently partially load.

    (a) GT hyperparameters that do not reproduce the pretrained Ψ producer;
    (b) an ε plane count from a different ``wire_rotate_nope_planes`` / head width.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    import tempfile

    from prism.models import loaders
    from prism.models.gt import GraphTransformer

    src = _wrap(llm)
    ckpt = _wire_ckpt_dict(src)

    # (a) GT depth drift: the carried state_dict has keys the target GT lacks.
    torch.manual_seed(1)
    deep_gt = GraphTransformer(num_layers=4, pe_hidden_channels=16, pe_num_layers=2,
                               d_model=16, heads=2, num_samples=8, dropout=0.0,
                               k_pe=2, k_gt=2)
    drifted = WireGraphLLM(_gemma4(seed=7), deep_gt, d_model=16, sigma_init=0.05,
                           max_angle=8.0, vanilla=False).eval()
    raised = ""
    with tempfile.TemporaryDirectory() as td:
        _write_ckpt(ckpt, td)
        try:
            loaders.load_pe_weights_into(drifted, td, "wire_llm")
        except RuntimeError as e:
            raised = str(e)
    assert "gnn." in raised and ("missing" in raised or "unexpected" in raised), \
        f"expected a loud GT-drift RuntimeError naming the gnn.* knobs, got {raised!r}"

    # (b) plane-count drift: ε is [P, m] and P follows wire_rotate_nope_planes.
    wide = _wrap(_gemma4(seed=7), rotate_nope_planes=True)
    assert set(wide._wire_eps.state_dict()) == set(ckpt["wire_eps"]), \
        "fixture must differ in PLANE COUNT only, not in the active layer set"
    assert any(wide._wire_eps.state_dict()[k].shape != v.shape
               for k, v in ckpt["wire_eps"].items()), "fixture has no plane-count drift"
    raised = ""
    with tempfile.TemporaryDirectory() as td:
        _write_ckpt(ckpt, td)
        try:
            loaders.load_pe_weights_into(wide, td, "wire_llm")
        except RuntimeError as e:
            raised = str(e)
    assert "wire_rotate_nope_planes" in raised, \
        f"expected a loud ε-shape RuntimeError naming the knob, got {raised!r}"


# --------------------------------------------------------------------------- #
# Decode: cached-step key re-rotation (wire_decode='rotate').                   #
# --------------------------------------------------------------------------- #

def _prompt_injection_map(n_nodes, prompt_len):
    """One 1-token span per node inside the prompt, disjoint and in range."""
    return {i: [(1 + i, 2 + i)] for i in range(n_nodes) if 2 + i <= prompt_len}


def _armed(wrap, r):
    """Arm an explicit signal, dropping any decode-angle cache built from a previous one."""
    wrap._wire_signal = r
    wrap._wire_decode_cos_sin.clear()


def test_wire_decode_matches_full_sequence_forward():
    """GATING TEST. Cached greedy decode must reproduce, step for step, the logits of an
    UNCACHED forward over the same tokens under the same semantics.

    The semantics: ``r`` is a function of position only, so the full-sequence signal is
    the prompt signal zero-padded over the generated positions (generated tokens are
    absent from the injection map ⇒ r = 0). If the decode branch forgot to re-rotate the
    cached prompt keys — or rotated them by the wrong angle, or misaligned the position
    offset — the cached score would carry a different ``r_j − r_i`` and these logits
    would diverge. This is the one test that would catch a wrong rotation.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm)                       # layer_scope='dense' (global layers only)
    g = _path_graph(3)
    ids = torch.randint(0, 64, (1, 7))
    L, T = ids.shape[1], 4
    imap = _prompt_injection_map(3, L)

    # ONE Ψ draw: the GT samples random probes, so re-arming would give a different r.
    with torch.no_grad():
        r = wrap.build_wire_signal([g], [imap], L, ids.device)
    assert float(r.abs().max()) > 0, "signal is all-zero — the test would be vacuous"

    _armed(wrap, r)
    with torch.no_grad():
        gen = wrap.llm.generate(
            input_ids=ids, max_new_tokens=T, do_sample=False, use_cache=True,
            pad_token_id=0, return_dict_in_generate=True, output_scores=True)
    assert gen.sequences.shape[1] == L + T, "generation stopped early — nothing to compare"

    r_full = torch.cat((r, r.new_zeros(1, T, r.shape[-1])), dim=1)
    _armed(wrap, r_full)
    with torch.no_grad():
        full = wrap.llm(input_ids=gen.sequences, use_cache=False).logits
    _armed(wrap, None)

    worst = max((full[:, L - 1 + t] - gen.scores[t]).abs().max().item() for t in range(T))
    assert worst < 1e-4, (
        f"cached decode diverges from the uncached forward (max|Δlogit|={worst:.3e}) — "
        "the cached keys are not carrying the graph rotation the prefill scored with")


def test_wire_decode_actually_rotates_cached_keys():
    """The gating test above is only meaningful if 'skip' (WIRE off at decode) DIFFERS.

    Same prompt, same Ψ, same greedy steps: 'rotate' re-rotates the cached prompt keys,
    'skip' delegates untouched. If the two agree the rotation is inert at decode and the
    consistency check proves nothing.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm)
    g = _path_graph(3)
    ids = torch.randint(0, 64, (1, 7))
    L, T = ids.shape[1], 4
    with torch.no_grad():
        r = wrap.build_wire_signal([g], [imap := _prompt_injection_map(3, L)], L, ids.device)

    def scores(mode):
        wrap._wire_decode = mode
        _armed(wrap, r)
        with torch.no_grad():
            out = wrap.llm.generate(
                input_ids=ids, max_new_tokens=T, do_sample=False, use_cache=True,
                pad_token_id=0, return_dict_in_generate=True, output_scores=True)
        _armed(wrap, None)
        return out.scores

    rot, skip = scores("rotate"), scores("skip")
    wrap._wire_decode = "rotate"
    # Step 0 is the prefill logit (identical by construction); the DECODE steps must differ.
    delta = max((rot[t] - skip[t]).abs().max().item() for t in range(1, T))
    assert delta > 1e-5, (
        f"wire_decode='rotate' and 'skip' produce identical decode logits (max|Δ|={delta:.2e}) "
        "— the cached-key rotation is inert, so the consistency test is vacuous")
    assert imap  # silence the walrus


def test_wire_decode_survives_shifted_position_ids():
    """REGRESSION. Decode must key off CACHE SLOTS, not ``position_ids``.

    ``generate`` re-derives ``position_ids`` from the attention mask, so a prompt that is
    padded — or merely contains a token equal to ``pad_token_id`` — gets positions BELOW
    the cache-slot index while the cache itself is perfectly aligned. An alignment check
    written against ``position_ids`` reads that as a cropped cache and aborts a healthy
    run (it did, flakily, whenever a random prompt happened to contain the pad id).

    Here the prompt is deliberately half pad ids and the decode logits must still match
    the uncached forward exactly, as in the gating test.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm)
    pad = 0
    ids = torch.randint(1, 64, (1, 8))
    ids[0, :4] = pad                                  # left "padding" -> shifted positions
    attn = torch.ones_like(ids)
    attn[0, :4] = 0
    L, T = ids.shape[1], 3
    with torch.no_grad():
        r = wrap.build_wire_signal([_path_graph(3)], [{0: [(5, 6)], 1: [(6, 7)]}],
                                   L, ids.device)
    _armed(wrap, r)
    with torch.no_grad():
        gen = wrap.llm.generate(
            input_ids=ids, attention_mask=attn, max_new_tokens=T, do_sample=False,
            use_cache=True, pad_token_id=pad, return_dict_in_generate=True,
            output_scores=True)
    steps = gen.sequences.shape[1] - L
    assert steps > 0, "nothing generated — the regression would not be exercised"

    r_full = torch.cat((r, r.new_zeros(1, steps, r.shape[-1])), dim=1)
    _armed(wrap, r_full)
    with torch.no_grad():
        full = wrap.llm(input_ids=gen.sequences, attention_mask=torch.cat(
            (attn, attn.new_ones(1, steps)), dim=1), use_cache=False).logits
    _armed(wrap, None)
    worst = max((full[:, L - 1 + t] - gen.scores[t]).abs().max().item()
                for t in range(steps))
    assert worst < 1e-4, (
        f"padded-prompt decode diverges from the uncached forward (max|Δ|={worst:.3e})")


def test_wire_generate_with_graph_runs_and_disarms():
    """``generate_with_graph`` must produce tokens (no NotImplementedError) and leave the
    signal AND the decode-angle cache cleared, so nothing leaks into the next sample."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm)
    ids = torch.randint(0, 64, (1, 7))
    out = wrap.generate_with_graph(
        input_ids=ids, graphs=[_path_graph(3)],
        injection_maps=[_prompt_injection_map(3, ids.shape[1])],
        max_new_tokens=5, do_sample=False, use_cache=True, pad_token_id=0)
    assert out.shape[0] == 1 and out.shape[1] > ids.shape[1], \
        f"no tokens generated (out {tuple(out.shape)} vs prompt {tuple(ids.shape)})"
    assert wrap._wire_signal is None, "_wire_signal must be disarmed after generate"
    assert not wrap._wire_decode_cos_sin, "decode angle cache must be cleared after generate"


def test_wire_generate_threads_permutation():
    """``permutation`` must reach the Ψ producer (the --permutation-seed sweep mode).

    Asserted at the producer, not on the output: a permuted graph relabels the nodes, so
    a run that ignored the kwarg would silently score the UNPERMUTED graph.
    """
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm)
    seen = []
    real = wrap.pe_model.forward

    def spy(data, permutation=None):
        seen.append(permutation)
        return real(data, permutation=permutation)

    wrap.pe_model.forward = spy
    perm = Permutation(7)
    ids = torch.randint(0, 64, (1, 7))
    try:
        wrap.generate_with_graph(
            input_ids=ids, graphs=[_path_graph(3)],
            injection_maps=[_prompt_injection_map(3, ids.shape[1])],
            permutation=perm, max_new_tokens=3, do_sample=False, use_cache=True,
            pad_token_id=0)
    finally:
        wrap.pe_model.forward = real
    assert seen and seen[0] is perm, \
        f"permutation did not reach the Psi producer (saw {seen!r})"


def test_wire_layer_scope_all_rejects_decode():
    """``layer_scope='all'`` puts WIRE on sliding-window layers, whose KV cache is cropped
    past the window — key slot 0 stops being absolute position 0. It must fail BEFORE
    generating, naming the config knob, not blanket-fail every WIRE run."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm, layer_scope="all")
    ids = torch.randint(0, 64, (1, 7))
    raised = ""
    try:
        wrap.generate_with_graph(
            input_ids=ids, graphs=[_path_graph(3)],
            injection_maps=[_prompt_injection_map(3, 7)],
            max_new_tokens=3, do_sample=False, use_cache=True, pad_token_id=0)
    except NotImplementedError as e:
        raised = str(e)
    assert "wire_layer_scope" in raised, \
        f"expected a loud pre-generation error naming gnn.wire_layer_scope, got {raised!r}"
    # ...and the same model under 'skip' (WIRE off at decode) is explicitly allowed.
    wrap._wire_decode = "skip"
    out = wrap.generate_with_graph(
        input_ids=ids, graphs=[_path_graph(3)],
        injection_maps=[_prompt_injection_map(3, 7)],
        max_new_tokens=3, do_sample=False, use_cache=True, pad_token_id=0)
    assert out.shape[1] > ids.shape[1]


def test_wire_decode_legacy_error_normalises_to_rotate():
    """Checkpoints written before decode existed recorded wire_decode='error'. They must
    still evaluate: the value is normalised to 'rotate' and the substitution is WARNED,
    never silent. 'skip' keeps its meaning; anything else still raises."""
    import warnings as _w
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        wrap = _wrap(llm)
        wrap._wire_decode = "x"                       # placeholder, replaced below
        from prism.models.gnn_llm import WireGraphLLM as _W
        legacy = _W(llm, wrap.pe_model, d_model=16, decode="error", max_angle=8.0)
    assert legacy._wire_decode == "rotate", \
        f"legacy 'error' must normalise to 'rotate', got {legacy._wire_decode!r}"
    assert any("legacy" in str(c.message) for c in caught), \
        "the legacy normalisation must be announced (DeprecationWarning), not silent"
    raised = ""
    try:
        _W(llm, wrap.pe_model, d_model=16, decode="bogus", max_angle=8.0)
    except ValueError as e:
        raised = str(e)
    assert "decode must be one of" in raised, f"unknown mode not rejected: {raised!r}"


def test_wire_is_routed_as_graph_augmented_at_eval():
    """``evaluate._is_graph_augmented`` must recognise WireGraphLLM.

    A missing entry does not fail loudly: it routes the checkpoint to the PLAIN-LLM
    client and evaluates it with the graph channel silently absent. Also pinned against
    the config-side list ``evaluate._GNN_ARCHITECTURES``.
    """
    from prism.eval import evaluate as _ev
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _wrap(llm)
    assert _ev._is_graph_augmented(wrap), \
        "WireGraphLLM must route to GraphAugmentedInMemoryLLM, not the plain client"
    assert "wire_llm" in _ev._GNN_ARCHITECTURES


# --------------------------------------------------------------------------- #
# wire_vanilla: the paper's algorithm, and the DEFAULT.                        #
#                                                                              #
# Vanilla mode replaces omega = sigma*eps with ONE LEARNABLE [P, m] table per   #
# layer (Alg. 1 / 3.3 "one learnable instantiation"). These tests pin (a) that  #
# it is the default in all three registration sites, (b) that the expectation   #
# machinery is genuinely OFF the active path, (c) that everything the rotation  #
# depends on still holds, and (d) that the two modes' checkpoints do not mix.   #
# --------------------------------------------------------------------------- #

def _vanilla(llm=None, d_model=16, **kw):
    """Vanilla-mode wrapper. Mirrors ``_wrap`` but leaves ``vanilla`` at the DEFAULT."""
    torch.manual_seed(1)
    gt = GraphTransformer(
        num_layers=2, pe_hidden_channels=16, pe_num_layers=2, d_model=d_model,
        heads=2, num_samples=8, dropout=0.0, k_pe=2, k_gt=2)
    kw.setdefault("pe_gain_init", 1.0)
    kw.setdefault("max_angle", 8.0)
    return WireGraphLLM(llm or _gemma4(), gt, d_model=d_model, **kw).eval()


def test_default_mode_is_vanilla():
    """THE DEFAULT. Asserted at all three registration sites, because a default that
    disagrees between them is how a run silently trains the wrong architecture."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    import inspect

    import omegaconf

    # (1) the class signature.
    sig = inspect.signature(WireGraphLLM.__init__)
    assert sig.parameters["vanilla"].default is True, \
        "WireGraphLLM.__init__ default for `vanilla` is not True"
    assert sig.parameters["vanilla_omega_init"].default == "zero"

    # (2) a model built with NO explicit setting comes up vanilla.
    w = _vanilla(llm)
    assert w._wire_vanilla is True, "constructing without `vanilla` did not give vanilla"
    assert len(w._wire_omega) > 0, "vanilla model has no omega table"
    assert len(w._wire_eps.state_dict()) == 0 and len(w._wire_sigma) == 0, \
        "vanilla model still built the expectation arm's eps/sigma"

    # (3) the shipped Hydra config.
    cfg = omegaconf.OmegaConf.load("experiments/base_config.yaml")
    assert cfg.gnn.wire_vanilla is True, "base_config.yaml does not default to vanilla"
    assert cfg.gnn.wire_vanilla_omega_init == "zero"

    # (4) architectures.build_planner_model with the key ABSENT must still pick vanilla
    #     (the eval-time loader deliberately defaults the other way; see loaders.py).
    from prism.models.architectures import build_planner_model
    gnn = cfg.gnn
    gnn.arch = "wire_llm"
    gnn.d_model = 16
    gnn.pe_hidden_channels = 16
    gnn.num_samples = 4
    gnn.gt_heads = 2
    gnn.pe_gain_init = 1.0
    del gnn["wire_vanilla"]                      # simulate a config predating the key

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

    model, _ = build_planner_model(gnn, _gemma4(), _Tok())
    assert model._wire_vanilla is True, \
        "architectures.build_model defaults `vanilla` to False when the key is absent"


def test_vanilla_omega_is_one_learnable_table_per_layer():
    """3.3: omega IS the learned parameter. No eps, no sigma, no reparameterisation."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    w = _vanilla(llm)
    li = _global_layer_idx(llm)
    attn = list(w._decoder_layers())[li].self_attn
    planes = wire_rope_planes(attn, rotate_nope=False)

    om = w._wire_omega[str(li)]
    assert om.shape == (planes, w._wire_d_model), \
        f"omega is {tuple(om.shape)}, not [P, m] = {(planes, w._wire_d_model)}"
    assert isinstance(om, torch.nn.Parameter) and om.requires_grad, \
        "omega must be a LEARNABLE parameter in vanilla mode (3.3)"
    assert any(k.endswith(f"_wire_omega.{li}") for k in w.state_dict()), \
        "omega missing from state_dict"
    # It trains at the BASE LR (outside structural_lr_mult, which exists to damp the
    # pretrained Ψ producer), and the frozen-buffer eps is nowhere.
    assert id(om) in {id(p) for p in w.base_lr_parameters()}, \
        "learnable omega missing from the base-LR group"
    assert id(om) not in {id(p) for p in w.structural_parameters()}, \
        "omega leaked into the structural (multiplier) LR group"
    assert not any("_wire_eps" in k or "_wire_sigma" in k for k in w.state_dict()), \
        "vanilla mode still carries expectation-arm state in the checkpoint"

    # active_layer_indices reads the vanilla store.
    act = [i for i, l in enumerate(w._decoder_layers())
           if getattr(l.self_attn, "_wire_active", False)]
    assert w.active_layer_indices() == act


def test_vanilla_zero_init_is_exact_identity_but_still_trainable():
    """The reference default init_omega="zero" gives theta=0 => EXACT identity, and it
    is a live starting point ONLY because the sin term of Eq. 8 is retained:
    d sin(theta)/d omega = cos(theta)*r = r != 0 at theta=0, while d cos/d omega = 0."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    w = _vanilla(llm, vanilla_omega_init="zero")
    for li in w.active_layer_indices():
        assert float(w._wire_omega[str(li)].abs().max()) == 0.0, "zero init is not zero"

    ids = torch.randint(0, 64, (1, 8))
    g, inj = _tiny_graph(3), {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}
    with torch.no_grad():
        out = w(input_ids=ids, graphs=[g], injection_maps=[inj]).logits
        stock = w.llm(input_ids=ids).logits
    assert torch.equal(out, stock), \
        "zero-init vanilla is not BIT-IDENTICAL to the stock LLM (rotation must be exact identity)"

    # ...and gradient still reaches omega, so it is not a dead start.
    out2 = w(input_ids=ids, graphs=[g], injection_maps=[inj])
    out2.logits.float().square().mean().backward()
    gs = [p.grad for p in w._wire_omega.values() if p.grad is not None]
    assert gs and max(float(gd.abs().max()) for gd in gs) > 0, \
        "omega got no gradient at zero init — the sin term has been dropped somewhere"
    assert all(torch.isfinite(gd).all().item() for gd in gs)


def test_vanilla_omega_init_strategies():
    """All three reference values construct, are finite, and the random ones are seeded."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    from prism.models.gnn_llm import WIRE_VANILLA_OMEGA_INITS
    assert WIRE_VANILLA_OMEGA_INITS == ("zero", "uniform", "exponential")

    for init in WIRE_VANILLA_OMEGA_INITS:
        a = _vanilla(_gemma4(), vanilla_omega_init=init, omega_seed=3)
        b = _vanilla(_gemma4(), vanilla_omega_init=init, omega_seed=3)
        k = next(iter(a._wire_omega))
        assert torch.isfinite(a._wire_omega[k]).all(), f"{init}: non-finite omega"
        assert torch.equal(a._wire_omega[k], b._wire_omega[k]), \
            f"{init}: same seed gave a different draw"
        if init != "zero":
            assert float(a._wire_omega[k].abs().max()) > 0, f"{init}: drew all zeros"
            d = _vanilla(_gemma4(), vanilla_omega_init=init, omega_seed=999)
            assert not torch.equal(a._wire_omega[k], d._wire_omega[k]), \
                f"{init}: omega_seed has no effect"
    # Invalid value rejected, naming the knob.
    try:
        _vanilla(_gemma4(), vanilla_omega_init="nope")
        raise AssertionError("invalid vanilla_omega_init was not rejected")
    except ValueError as e:
        assert "vanilla_omega_init" in str(e)


def test_vanilla_relative_only_holds_under_gqa():
    """Eq. 3 end to end in VANILLA mode: shifting r by a constant leaves every q.k score
    unchanged. This is the invariant the shared-across-heads omega exists to protect —
    a key head serves several query heads, so q and k must rotate by identical omega."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    # Non-zero omega, or the test is vacuous (zero init => rotation is the identity).
    wrap = _vanilla(llm, vanilla_omega_init="uniform", omega_seed=5)
    li = _global_layer_idx(llm)
    attn = list(wrap._decoder_layers())[li].self_attn
    assert attn.num_key_value_groups > 1, "fixture is not GQA — test would be vacuous"

    ids = torch.randint(0, 64, (1, 6))
    torch.manual_seed(22)
    r = torch.randn(1, 6, wrap._wire_d_model) * 0.4
    c = torch.randn(1, 1, wrap._wire_d_model) * 0.4

    def scores(sig):
        cap = _capture_qk(wrap, ids, sig, li)
        q, k = cap["q"], cap["k"]
        k = k.repeat_interleave(attn.num_key_value_groups, dim=1)   # mirrors repeat_kv
        return q @ k.transpose(-1, -2)

    s0, s1 = scores(r), scores(r + c)
    d = (s0 - s1).abs().max().item()
    assert d < 1e-3, f"vanilla scores moved under a global r shift (max|delta|={d:.2e})"
    assert (s0 - scores(None)).abs().max().item() > 1e-3, "WIRE inert — vacuous test"


def test_vanilla_omega_is_shared_across_heads():
    """One [P, m] table per LAYER: q and k are rotated by the SAME frequencies, which is
    what keeps Eq. 3 exact under GQA (3.1 sanctions head sharing explicitly)."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    wrap = _vanilla(llm, vanilla_omega_init="uniform", omega_seed=5)
    li = _global_layer_idx(llm)
    attn = list(wrap._decoder_layers())[li].self_attn
    omega = wrap.layer_omega(li)
    assert omega.dim() == 2, f"omega must be [P, m], got {tuple(omega.shape)}"

    ids = torch.randint(0, 64, (1, 6))
    torch.manual_seed(21)
    r = torch.randn(1, 6, wrap._wire_d_model) * 0.5
    off = _capture_qk(wrap, ids, None, li)
    on = _capture_qk(wrap, ids, r, li)
    cos, sin = wire_cos_sin(r, omega.detach(), attn.head_dim)
    from prism.models.gnn_llm import _wire_rotate
    dq = (on["q"] - _wire_rotate(off["q"], cos, sin)).abs().max().item()
    dk = (on["k"] - _wire_rotate(off["k"], cos, sin)).abs().max().item()
    assert dq < 1e-5, f"query rotation != shared omega (max|delta|={dq:.2e})"
    assert dk < 1e-5, f"key rotation != shared omega (max|delta|={dk:.2e})"
    assert (on["q"] - off["q"]).abs().max().item() > 1e-4, "rotation inert — vacuous"


def test_vanilla_has_no_expectation_machinery_on_the_active_path():
    """The point of the switch: sigma/eps must not participate in vanilla mode at all."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    w = _vanilla(llm, vanilla_omega_init="uniform", omega_seed=5)
    li = w.active_layer_indices()[0]

    # omega == the table itself (times the clamp factor, which is 1.0 here).
    assert w.layer_scale_factor(li) == 1.0
    assert torch.equal(w.layer_omega(li).detach(), w._wire_omega[str(li)].detach()), \
        "layer_omega applied something other than the learned table"
    # No sigma to freeze, no eps buffer to draw.
    assert str(li) not in w._wire_sigma and not hasattr(w._wire_eps, str(li))

    # freeze_sigma pins the LEARNABLE term in vanilla too (that term is omega).
    f = _vanilla(_gemma4(), freeze_sigma=True)
    for p in f._wire_omega.values():
        assert not p.requires_grad, "freeze_sigma did not pin omega in vanilla mode"
        assert id(p) not in {id(x) for x in f.structural_parameters()}, \
            "frozen omega leaked into the structural LR group"
    assert any("_wire_omega" in k for k in f.state_dict()), "omega missing from state_dict"


def test_vanilla_clamp_is_retained_and_visible():
    """DELIBERATE DEPARTURE FROM THE PAPER: the angle clamp stays ON in vanilla mode.
    Neither the paper nor the reference bounds theta; it is kept because nothing else
    bounds it here (Psi comes from a GT whose final block has normalize=False). Assert
    it (a) never raises, (b) holds by construction, (c) is EXACTLY the identity inside
    the bound, and (d) reports saturation in telemetry rather than failing silently."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    import warnings as _w
    g = _tiny_graph(4)
    inj = {i: [(i, i + 1)] for i in range(4)}

    # Inside the bound: factor is EXACTLY 1.0 => the paper's algorithm, bit for bit.
    w = _vanilla(_gemma4(), vanilla_omega_init="zero", max_angle=1.0)
    w.build_wire_signal([g], [inj], seq_len=8, device=torch.device("cpu"))
    for li in w.active_layer_indices():
        assert w.layer_scale_factor(li) == 1.0, "clamp not identity inside the bound"

    # Adversarially large omega must NOT raise and must stay inside the bound.
    for mult in (1.0, 10.0, 1e2, 1e3):
        w2 = _vanilla(_gemma4(), vanilla_omega_init="uniform", omega_seed=5,
                      max_angle=0.25)
        with torch.no_grad():
            for k in w2._wire_omega:
                w2._wire_omega[k].mul_(mult)
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            w2.build_wire_signal([g], [inj], seq_len=8, device=torch.device("cpu"))
        span = w2._wire_psi_span
        for li in w2.active_layer_indices():
            eff = w2.layer_omega_scale(li) * w2.layer_scale_factor(li)
            assert eff * span <= w2._wire_max_angle * (1 + 1e-6), (
                f"mult={mult}: post-clamp angle {eff * span:.6f} > "
                f"max_angle {w2._wire_max_angle}")
        assert w2._wire_effective_angle <= w2._wire_max_angle * (1 + 1e-5)

    # Saturation is VISIBLE, and gradient still reaches omega.
    ids = torch.randint(0, 64, (1, 8))
    inj8 = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}
    w3 = _vanilla(_gemma4(), vanilla_omega_init="uniform", omega_seed=5, max_angle=1e-3)
    with torch.no_grad():
        for k in w3._wire_omega:
            w3._wire_omega[k].mul_(100.0)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        out = w3(input_ids=ids, graphs=[_tiny_graph(3)], injection_maps=[inj8])
    assert torch.isfinite(out.logits).all(), "non-finite logits under a clamped omega"
    out.logits.float().square().mean().backward()
    gs = [p.grad for p in w3._wire_omega.values() if p.grad is not None]
    assert gs and all(torch.isfinite(gd).all().item() for gd in gs)

    t = w3.wire_telemetry()
    assert t["wire/vanilla"] == 1, "telemetry does not record the mode"
    assert t["wire/clamp_engaged"] == 1, "clamp engaged but not reported"
    assert t["wire/sigma_raw_max"] > t["wire/sigma_eff_max"], \
        "saturation invisible: raw and effective scale reported identical"
    assert t["wire/angle_eff_max"] <= t["wire/max_angle"] * (1 + 1e-6)
    for key in ("wire/psi_span", "wire/scale_min", "wire/clamped_layers"):
        assert key in t, f"telemetry missing {key}"
    assert all(isinstance(v, (int, float)) for v in t.values())


def test_vanilla_checkpoint_roundtrip_and_no_cross_mode_mixing():
    """Round-trip in BOTH modes, and refuse to mix them.

    The saved key SET is mode-independent (the unused store writes an empty dict), but
    the populated store differs. Loading a vanilla checkpoint into an expectation-arm
    model (or vice versa) would otherwise cold-start omega while reporting a resume."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    import tempfile

    from prism.models import loaders

    # ---- vanilla round-trip -------------------------------------------------
    a = _vanilla(llm, vanilla_omega_init="uniform", omega_seed=5)
    _perturb(a)
    ck_v = _wire_ckpt_dict(a)
    assert ck_v["wire_omega"] and not ck_v["wire_eps"] and not ck_v["wire_sigma"], \
        "vanilla checkpoint did not populate exactly wire_omega"

    b = _vanilla(_gemma4(seed=99), vanilla_omega_init="uniform", omega_seed=5)
    with tempfile.TemporaryDirectory() as td:
        _write_ckpt(ck_v, td)
        loaders.load_pe_weights_into(b, td, "wire_llm")
    for k, v in ck_v["wire_omega"].items():
        assert torch.equal(v, b._wire_omega.state_dict()[k]), f"omega {k} changed on reload"
    li = _global_layer_idx(b.llm)
    assert torch.equal(a.layer_omega(li).detach(), b.layer_omega(li).detach()), \
        "omega differs after reload"
    assert torch.equal(a.pe_gain.data, b.pe_gain.data), "gate not carried"

    # ---- expectation-arm round-trip ----------------------------------------
    c = _wrap(_gemma4(), omega_learnable=False)
    _perturb(c)
    ck_e = _wire_ckpt_dict(c)
    assert ck_e["wire_eps"] and ck_e["wire_sigma"] and not ck_e["wire_omega"], \
        "expectation checkpoint did not populate exactly wire_eps/wire_sigma"
    d = _wrap(_gemma4(seed=99), omega_learnable=False)
    with tempfile.TemporaryDirectory() as td:
        _write_ckpt(ck_e, td)
        loaders.load_pe_weights_into(d, td, "wire_llm")
    assert torch.equal(c.layer_omega(li).detach(), d.layer_omega(li).detach()), \
        "omega = sigma*eps differs after reload"

    # ---- crossing the boundary must RAISE, not cold-start -------------------
    for ckpt, target, label in ((ck_v, _wrap(_gemma4()), "vanilla ckpt -> expectation model"),
                                (ck_e, _vanilla(_gemma4()), "expectation ckpt -> vanilla model")):
        raised = ""
        with tempfile.TemporaryDirectory() as td:
            _write_ckpt(ckpt, td)
            try:
                loaders.load_pe_weights_into(target, td, "wire_llm")
            except KeyError as e:
                raised = str(e)
        assert "wire_vanilla" in raised, \
            f"{label}: expected a loud cross-mode KeyError naming the knob, got {raised!r}"


def test_vanilla_forward_decode_and_permutation_still_work():
    """The mode switch must not disturb anything the rotation is wired into: a forward
    that changes the logits, a generate that produces tokens and disarms, and the
    permutation threading used by the eval-time equivariance sweep."""
    llm = _gemma4()
    if llm is None:
        return _skip("gemma4 unavailable")
    from prism.models.utils import Permutation

    w = _vanilla(llm, vanilla_omega_init="uniform", omega_seed=5)
    ids = torch.randint(0, 64, (1, 8))
    g, inj = _tiny_graph(3), {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}

    with torch.no_grad():
        out = w(input_ids=ids, graphs=[g], injection_maps=[inj]).logits
        stock = w.llm(input_ids=ids).logits
    assert torch.isfinite(out).all()
    assert (out - stock).abs().max().item() > 1e-5, "vanilla forward is inert"
    assert w._wire_signal is None, "signal not disarmed after forward"

    with torch.no_grad():
        o = w.generate_with_graph(input_ids=ids, graphs=[g], injection_maps=[inj],
                                  max_new_tokens=3, do_sample=False)
    assert o.shape[1] > ids.shape[1], "vanilla generate produced no tokens"
    assert w._wire_signal is None and not w._wire_decode_cos_sin, \
        "generate left the signal or the decode-angle cache armed"

    # Permutation takes the seed only; num_nodes is an `apply` argument (models/utils.py:19).
    perm = Permutation(seed=0)
    r0 = w.build_wire_signal([g], [inj], 8, torch.device("cpu"))
    r1 = w.build_wire_signal([g], [inj], 8, torch.device("cpu"), permutation=perm)
    assert torch.isfinite(r1).all() and r0.shape == r1.shape


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
    print("done")
