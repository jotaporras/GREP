"""Objective-agnostic numerical smoke test of the whole ``WireGraphLLM`` surface.

This file is deliberately NOT written to validate the WIRE approach or to support any
conclusion about it. It enumerates every public method, every module-level function,
every ``wire_*`` config switch (including the non-default side of each), and the
degenerate-input paths, exercises each one numerically on the tiny CPU ``gemma4``
fixture, and records what it measured.

Every row is (path, what was checked, measured value, status). A row is only PASS if
the numeric assertion for that path actually held; paths that could not be exercised
are recorded as UNEXERCISED with a reason rather than being asserted more weakly than
the code actually does.

Run:  uv run --with pytest -m pytest tests/test_wire_smoke.py -q -s
"""
import sys
sys.path.insert(0, "src")

import warnings

import torch
from torch_geometric.data import Data

from prism.models.gnn_llm import (
    MASK_LAYER_SCOPES,
    WIRE_DECODE_LEGACY,
    WIRE_DECODE_MODES,
    WireGraphLLM,
    _wire_resolve_orig_attn_fn,
    _wire_rotate,
    wire_cos_sin,
    wire_place_at_node_spans,
    wire_rope_planes,
)
from prism.models.gt import GraphTransformer

ROWS = []


def rec(path, checked, measured, status):
    ROWS.append((path, checked, str(measured), status))


def _gemma4(num_layers=4, seed=0):
    try:
        from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    except Exception:
        return None
    torch.manual_seed(seed)
    cfg = Gemma4TextConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=num_layers, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, global_head_dim=16, max_position_embeddings=64,
        sliding_window=8, attn_implementation="eager")
    return Gemma4ForCausalLM(cfg).eval()


def _gt(d_model=16, seed=1):
    torch.manual_seed(seed)
    return GraphTransformer(num_layers=2, pe_hidden_channels=16, pe_num_layers=2,
                            d_model=d_model, heads=2, num_samples=8, dropout=0.0,
                            k_pe=2, k_gt=2)


def _wrap(llm=None, d_model=16, **kw):
    # vanilla=False by default: this module's assertions are about the EXPECTATION arm
    # (eps buffers + learnable sigma). The vanilla path is covered in
    # test_wire_injection.py, which also asserts the class default is vanilla=True.
    kw.setdefault("pe_gain_init", 1.0)
    kw.setdefault("vanilla", False)
    return WireGraphLLM(llm or _gemma4(), _gt(d_model), d_model=d_model, **kw).eval()


def _path_graph(n):
    if n == 1:
        g = Data(x=torch.randn(1, 1),
                 edge_index=torch.zeros(2, 0, dtype=torch.long))
        g.num_nodes = 1
        return g
    src, dst = list(range(n - 1)), list(range(1, n))
    g = Data(x=torch.randn(n, 1),
             edge_index=torch.tensor([src + dst, dst + src], dtype=torch.long))
    g.num_nodes = n
    return g


def _isolated_graph(n=4):
    """n nodes, ONE edge — leaves n-2 isolated (degree-0) nodes."""
    g = Data(x=torch.randn(n, 1),
             edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    g.num_nodes = n
    return g


CPU = torch.device("cpu")
IDS = torch.randint(0, 64, (1, 8))
INJ = {0: [(2, 3)], 1: [(4, 5)], 2: [(6, 7)]}


def _skipall(msg):
    import pytest
    pytest.skip(msg)


# ===================================================================== module fns

def test_module_level_functions():
    llm = _gemma4()
    if llm is None:
        return _skipall("gemma4 unavailable")
    w = _wrap(llm)
    layers = list(w._decoder_layers())
    glob = [i for i, t in enumerate(llm.config.layer_types) if t == "full_attention"][0]
    slid = [i for i, t in enumerate(llm.config.layer_types) if t == "sliding_attention"][0]

    # --- wire_rope_planes: both flags x both layer types -----------------------
    ga, gs = layers[glob].self_attn, layers[slid].self_attn
    p_g_f = wire_rope_planes(ga, rotate_nope=False)
    p_g_t = wire_rope_planes(ga, rotate_nope=True)
    p_s_f = wire_rope_planes(gs, rotate_nope=False)
    p_s_t = wire_rope_planes(gs, rotate_nope=True)
    ok = (p_g_f == int(0.25 * ga.head_dim // 2) and p_g_t == ga.head_dim // 2
          and p_s_f == gs.head_dim // 2 and p_s_t == gs.head_dim // 2)
    rec("wire_rope_planes", "global/sliding x rotate_nope F/T",
        f"global {p_g_f}/{p_g_t} of {ga.head_dim // 2}, sliding {p_s_f}/{p_s_t} "
        f"of {gs.head_dim // 2}", "PASS" if ok else "FAIL")
    assert ok

    # --- wire_cos_sin ---------------------------------------------------------
    r = torch.randn(2, 5, 16)
    om = torch.randn(3, 16)
    cos, sin = wire_cos_sin(r, om, 8)
    shape_ok = cos.shape == (2, 5, 8) and sin.shape == (2, 5, 8)
    dt_ok = cos.dtype == torch.float32
    pad_ok = (torch.equal(cos[..., 3:4], torch.ones(2, 5, 1))
              and torch.equal(sin[..., 3:4], torch.zeros(2, 5, 1)))
    ident = wire_cos_sin(torch.zeros(1, 3, 16), om, 8)
    id_ok = torch.equal(ident[0], torch.ones(1, 3, 8)) and torch.equal(
        ident[1], torch.zeros(1, 3, 8))
    # fp32 promotion from a bf16 input
    cb, _ = wire_cos_sin(r.bfloat16(), om.bfloat16(), 8)
    prom_ok = cb.dtype == torch.float32
    rec("wire_cos_sin", "shape/dtype/zero-pad/r=0 identity/bf16->fp32",
        f"shape {tuple(cos.shape)} dtype {cos.dtype} pad_identity={pad_ok} "
        f"r0_identity={id_ok} bf16_promoted={prom_ok}",
        "PASS" if (shape_ok and dt_ok and pad_ok and id_ok and prom_ok) else "FAIL")
    assert shape_ok and dt_ok and pad_ok and id_ok and prom_ok

    raised = ""
    try:
        wire_cos_sin(r, torch.randn(9, 16), 8)   # P > head_dim//2
    except ValueError as e:
        raised = str(e)
    rec("wire_cos_sin", "P > head_dim//2 rejected",
        f"ValueError: {raised[:48]}", "PASS" if "planes" in raised else "FAIL")
    assert "planes" in raised

    # --- _wire_rotate ---------------------------------------------------------
    x = torch.randn(1, 4, 5, 8)
    c, s = wire_cos_sin(torch.randn(1, 5, 16), om, 8)
    y = _wire_rotate(x, c, s)
    n_err = (x.norm(dim=-1) - y.norm(dim=-1)).abs().max().item()
    oop = y.data_ptr() != x.data_ptr()
    ident2 = _wire_rotate(x, torch.ones_like(c), torch.zeros_like(s))
    rec("_wire_rotate", "norm preserved / out-of-place / identity at theta=0",
        f"max|dnorm|={n_err:.2e} out_of_place={oop} identity_exact="
        f"{torch.equal(ident2, x)}",
        "PASS" if (n_err < 1e-5 and oop and torch.equal(ident2, x)) else "FAIL")
    assert n_err < 1e-5 and oop and torch.equal(ident2, x)

    # --- wire_place_at_node_spans --------------------------------------------
    dest = torch.zeros(1, 6, 4)
    rows = torch.arange(12, dtype=torch.float32).view(3, 4)
    wire_place_at_node_spans(dest, 0, rows, {0: [(0, 2)], 2: [(4, 99)]}, 6)
    placed = (dest[0].norm(dim=-1) > 0).tolist()
    exp = [True, True, False, False, True, True]      # span 2 truncated to seq_len
    d2 = torch.zeros(1, 6, 4)
    wire_place_at_node_spans(d2, 0, rows, {}, 6)      # empty map
    rec("wire_place_at_node_spans", "placement / span truncation / empty map",
        f"placed={placed} empty_map_allzero={float(d2.abs().max()) == 0.0}",
        "PASS" if (placed == exp and float(d2.abs().max()) == 0.0) else "FAIL")
    assert placed == exp and float(d2.abs().max()) == 0.0

    # --- _wire_resolve_orig_attn_fn ------------------------------------------
    import transformers.masking_utils as mu
    mod, fn = _wire_resolve_orig_attn_fn(w, layers[0].self_attn)
    reg = "prism_wire" in getattr(mu, "ALL_MASK_ATTENTION_FUNCTIONS")._global_mapping
    rec("_wire_resolve_orig_attn_fn", "returns modeling module + delegate; mask mirrored",
        f"module={mod.__name__.split('.')[-1]} callable={callable(fn)} mask_registered={reg}",
        "PASS" if (callable(fn) and reg) else "FAIL")
    assert callable(fn) and reg


# ===================================================================== config axes

def test_config_switch_matrix():
    if _gemma4() is None:
        return _skipall("gemma4 unavailable")
    g, inj = _path_graph(3), INJ

    # --- wire_layer_scope: every value in MASK_LAYER_SCOPES -------------------
    for scope in MASK_LAYER_SCOPES:
        w = _wrap(layer_scope=scope)
        act = [i for i, l in enumerate(w._decoder_layers())
               if getattr(l.self_attn, "_wire_active", False)]
        sig = sorted(int(k) for k in w._wire_sigma)
        with torch.no_grad():
            out = w(input_ids=IDS, graphs=[g], injection_maps=[inj]).logits
        ok = act == sig and torch.isfinite(out).all()
        rec(f"wire_layer_scope={scope}", "active layers == sigma keys; forward finite",
            f"active={act} finite={bool(torch.isfinite(out).all())}",
            "PASS" if ok else "FAIL")
        assert ok

    # --- wire_freeze_sigma: both sides ---------------------------------------
    for frz in (False, True):
        w = _wrap(freeze_sigma=frz)
        req = [p.requires_grad for p in w._wire_sigma.values()]
        insp = {id(p) for p in w.structural_parameters()}
        in_group = [id(p) in insp for p in w._wire_sigma.values()]
        ok = all(r == (not frz) for r in req) and all(m == (not frz) for m in in_group)
        rec(f"wire_freeze_sigma={frz}", "requires_grad and LR-group membership",
            f"requires_grad={req[0]} in_structural_group={in_group[0]}",
            "PASS" if ok else "FAIL")
        assert ok

    # --- wire_rotate_nope_planes: both sides ---------------------------------
    for rn in (False, True):
        w = _wrap(rotate_nope_planes=rn)
        li = [i for i, l in enumerate(w._decoder_layers())
              if not getattr(l.self_attn, "is_sliding", False)][0]
        attn = list(w._decoder_layers())[li].self_attn
        planes = getattr(w._wire_eps, str(li)).shape[0]
        exp = attn.head_dim // 2 if rn else int(0.25 * attn.head_dim // 2)
        rec(f"wire_rotate_nope_planes={rn}", "eps plane count on a global layer",
            f"planes={planes} expected={exp} (head_dim={attn.head_dim})",
            "PASS" if planes == exp else "FAIL")
        assert planes == exp

    # --- wire_sigma_init / wire_omega_seed -----------------------------------
    a, b = _wrap(sigma_init=0.02), _wrap(sigma_init=0.02)
    c = _wrap(sigma_init=0.07)
    k = next(iter(a._wire_sigma))
    same_eps = torch.equal(getattr(a._wire_eps, k), getattr(b._wire_eps, k))
    d = _wrap(sigma_init=0.02, omega_seed=999)
    diff_eps = not torch.equal(getattr(a._wire_eps, k), getattr(d._wire_eps, k))
    # σ is stored as an fp32 Parameter, so a config float does NOT round-trip exactly
    # (0.02 -> 0.019999999552965164). Compare to fp32 tolerance, not with ==.
    sig_ok = (abs(float(a._wire_sigma[k]) - 0.02) < 1e-7
              and abs(float(c._wire_sigma[k]) - 0.07) < 1e-7)
    rec("wire_sigma_init / wire_omega_seed",
        "sigma set (fp32 tol); eps reproducible by seed",
        f"sigma={float(a._wire_sigma[k]):.9f}/{float(c._wire_sigma[k]):.9f} "
        f"(fp32, not exact) same_seed_same_eps={same_eps} diff_seed_diff_eps={diff_eps}",
        "PASS" if (sig_ok and same_eps and diff_eps) else "FAIL")
    assert sig_ok and same_eps and diff_eps

    # --- wire_decode: every accepted value generates --------------------------
    # Both live modes decode ('rotate' re-rotates the cached prompt keys, 'skip' runs
    # WIRE off at decode), and the LEGACY 'error' — recorded by checkpoints written
    # before the rotation existed — must normalise to 'rotate' rather than raise.
    for mode in WIRE_DECODE_MODES + tuple(WIRE_DECODE_LEGACY):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            w = _wrap(decode=mode)
        expected = WIRE_DECODE_LEGACY.get(mode, mode)
        err = ""
        gen_ok = False
        try:
            with torch.no_grad():
                o = w.generate_with_graph(input_ids=IDS, graphs=[g],
                                          injection_maps=[inj], max_new_tokens=3,
                                          do_sample=False)
            gen_ok = o.shape[1] > IDS.shape[1]
        except NotImplementedError as e:
            err = str(e)[:60]
        ok = gen_ok and w._wire_decode == expected
        meas = (f"resolved={w._wire_decode!r} generated ok, out_len={o.shape[1]}"
                if gen_ok else f"raised: {err}")
        rec(f"wire_decode={mode}", f"cached decode behaviour (-> {expected})", meas,
            "PASS" if ok else "FAIL")
        assert ok

    # --- invalid config rejected ---------------------------------------------
    bad = []
    for kw, frag in (({"layer_scope": "nope"}, "layer_scope"),
                     ({"decode": "nope"}, "decode"),
                     ({"sigma_init": 0.0}, "sigma_init"),
                     ({"max_angle": 0.0}, "max_angle"),
                     ({"pe_node_features": "word_embeddings"}, "pe_node_features")):
        try:
            _wrap(**kw)
            bad.append(f"{frag}: NOT rejected")
        except ValueError as e:
            if frag not in str(e):
                bad.append(f"{frag}: wrong message")
    rec("__init__ validation", "5 invalid configs rejected with a naming message",
        "all rejected" if not bad else "; ".join(bad), "PASS" if not bad else "FAIL")
    assert not bad


# ===================================================================== methods

def test_public_methods_numeric():
    if _gemma4() is None:
        return _skipall("gemma4 unavailable")
    w = _wrap()
    g, inj = _path_graph(3), INJ

    # active_layer_indices / _decoder_layers
    idxs = w.active_layer_indices()
    rec("active_layer_indices / _decoder_layers", "sorted, matches eps buffers",
        f"{idxs}, n_layers={len(list(w._decoder_layers()))}",
        "PASS" if idxs == sorted(idxs) and idxs else "FAIL")
    assert idxs == sorted(idxs) and idxs

    # build_wire_signal
    r = w.build_wire_signal([g], [inj], seq_len=8, device=CPU)
    nz = (r[0].norm(dim=-1) > 0).nonzero().flatten().tolist()
    rec("build_wire_signal", "shape/dtype/finite/placed-only-at-spans",
        f"shape={tuple(r.shape)} dtype={r.dtype} finite={bool(torch.isfinite(r).all())} "
        f"placed={nz}", "PASS" if (r.shape == (1, 8, 16) and torch.isfinite(r).all()
                                   and nz == [2, 4, 6]) else "FAIL")
    assert r.shape == (1, 8, 16) and torch.isfinite(r).all() and nz == [2, 4, 6]

    # layer_scale_factor / layer_omega
    li = idxs[0]
    s = w.layer_scale_factor(li)
    om = w.layer_omega(li)
    expect = w._wire_sigma[str(li)].detach() * s * getattr(w._wire_eps, str(li))
    rec("layer_scale_factor / layer_omega", "factor in (0,1]; omega == sigma*s*eps",
        f"s={s:.6g} shape={tuple(om.shape)} max|omega-expected|="
        f"{float((om.detach() - expect).abs().max()):.2e}",
        "PASS" if (0 < s <= 1.0 and float((om.detach() - expect).abs().max()) < 1e-7)
        else "FAIL")
    assert 0 < s <= 1.0 and float((om.detach() - expect).abs().max()) < 1e-7

    # wire_telemetry
    t = w.wire_telemetry()
    need = {"wire/psi_span", "wire/max_angle", "wire/sigma_raw_max", "wire/sigma_eff_max",
            "wire/angle_raw_max", "wire/angle_eff_max", "wire/clamp_engaged",
            "wire/clamped_layers", "wire/scale_min"}
    miss = need - set(t)
    allnum = all(isinstance(v, (int, float)) for v in t.values())
    rec("wire_telemetry", "required keys present, all values scalar numbers",
        f"{len(t)} keys, missing={sorted(miss) or 'none'}, all_scalar={allnum}",
        "PASS" if (not miss and allnum) else "FAIL")
    assert not miss and allnum

    # structural_parameters
    sp = w.structural_parameters()
    has_sig = any(id(p) in {id(x) for x in sp} for p in w._wire_sigma.values())
    has_eps = any(id(getattr(w._wire_eps, str(i))) in {id(x) for x in sp} for i in idxs)
    rec("structural_parameters", "includes pe_model+pe_gain+sigma, EXCLUDES eps",
        f"n={len(sp)} sigma_in={has_sig} eps_in={has_eps}",
        "PASS" if (has_sig and not has_eps) else "FAIL")
    assert has_sig and not has_eps

    # __getattr__ passthrough
    ok_attr = w.config is not None and hasattr(w, "generate")
    rec("__getattr__", "falls through to self.llm", f"config+generate reachable={ok_attr}",
        "PASS" if ok_attr else "FAIL")
    assert ok_attr

    # gradient checkpointing enable/disable
    gc_err = ""
    try:
        w.gradient_checkpointing_enable()
        on = bool(getattr(w.llm, "is_gradient_checkpointing", False))
        w.gradient_checkpointing_disable()
        off = bool(getattr(w.llm, "is_gradient_checkpointing", False))
    except Exception as e:
        gc_err, on, off = str(e)[:40], None, None
    rec("gradient_checkpointing_enable/disable", "toggles llm flag",
        f"on={on} off={off} err={gc_err or 'none'}",
        "PASS" if (gc_err == "" and on and not off) else "FAIL")
    assert gc_err == "" and on and not off

    # forward: signal armed during, disarmed after
    with torch.no_grad():
        out = w(input_ids=IDS, graphs=[g], injection_maps=[inj]).logits
        stock = w.llm(input_ids=IDS).logits
    rec("forward", "finite logits, differ from stock, signal disarmed after",
        f"finite={bool(torch.isfinite(out).all())} "
        f"max|d|={float((out - stock).abs().max()):.3e} disarmed={w._wire_signal is None}",
        "PASS" if (torch.isfinite(out).all() and w._wire_signal is None
                   and float((out - stock).abs().max()) > 1e-5) else "FAIL")
    assert torch.isfinite(out).all() and w._wire_signal is None

    # forward with NO graph == stock
    with torch.no_grad():
        nog = w(input_ids=IDS).logits
    rec("forward (no graph)", "identical to stock LLM",
        f"max|d|={float((nog - stock).abs().max()):.3e}",
        "PASS" if float((nog - stock).abs().max()) == 0.0 else "FAIL")
    assert float((nog - stock).abs().max()) == 0.0

    # _arm
    w._arm([g], [inj], 8, CPU)
    armed = w._wire_signal is not None
    w._wire_signal = None
    rec("_arm", "sets _wire_signal", f"armed={armed}", "PASS" if armed else "FAIL")
    assert armed


# ===================================================================== gradients

def test_gradient_flow_targets():
    if _gemma4() is None:
        return _skipall("gemma4 unavailable")
    torch.manual_seed(5)
    w = _wrap()
    g, inj = _path_graph(3), INJ
    out = w(input_ids=IDS, graphs=[g], injection_maps=[inj])
    out.logits.float().square().mean().backward()

    def gn(p):
        return 0.0 if p.grad is None else float(p.grad.abs().max())

    gain = gn(w.pe_gain)
    sig = max(gn(p) for p in w._wire_sigma.values())
    pe = max((gn(p) for p in w.pe_model.parameters()), default=0.0)
    eps_grads = [getattr(w._wire_eps, str(i)).grad for i in w.active_layer_indices()]
    eps_none = all(x is None for x in eps_grads)
    eps_rg = all(not getattr(w._wire_eps, str(i)).requires_grad
                 for i in w.active_layer_indices())
    finite = all(torch.isfinite(p.grad).all() for p in w.parameters() if p.grad is not None)
    ok = gain > 0 and sig > 0 and pe > 0 and eps_none and eps_rg and finite
    rec("gradient flow", "grad reaches pe_model/pe_gain/sigma; eps receives none",
        f"pe_gain={gain:.3e} sigma={sig:.3e} pe_model={pe:.3e} "
        f"eps_grad_is_None={eps_none} eps_requires_grad={not eps_rg} all_finite={finite}",
        "PASS" if ok else "FAIL")
    assert ok


def test_determinism():
    if _gemma4() is None:
        return _skipall("gemma4 unavailable")
    g, inj = _path_graph(3), INJ
    outs = []
    for _ in range(2):
        torch.manual_seed(11)
        w = _wrap(_gemma4(seed=3))
        torch.manual_seed(12)
        with torch.no_grad():
            outs.append(w(input_ids=IDS, graphs=[g], injection_maps=[inj]).logits)
    same = torch.equal(outs[0], outs[1])
    rec("determinism", "same seed -> bit-identical logits",
        f"bit_identical={same} max|d|={float((outs[0] - outs[1]).abs().max()):.2e}",
        "PASS" if same else "FAIL")
    assert same


# ===================================================================== degenerate

def test_degenerate_inputs():
    if _gemma4() is None:
        return _skipall("gemma4 unavailable")
    w = _wrap()

    cases = []
    # empty injection map
    try:
        r = w.build_wire_signal([_path_graph(3)], [{}], seq_len=8, device=CPU)
        cases.append(("empty injection map",
                      f"r allzero={float(r.abs().max()) == 0.0} finite={bool(torch.isfinite(r).all())}",
                      float(r.abs().max()) == 0.0 and bool(torch.isfinite(r).all())))
    except Exception as e:
        cases.append(("empty injection map", f"RAISED {type(e).__name__}: {e}", False))

    # single-node graph, N=1 (cdist branch skipped: span must be 0)
    try:
        r = w.build_wire_signal([_path_graph(1)], [{0: [(1, 2)]}], seq_len=8, device=CPU)
        span = w._wire_psi_span
        s = w.layer_scale_factor(w.active_layer_indices()[0])
        cases.append(("N=1 single-node graph",
                      f"span={span} scale={s} finite={bool(torch.isfinite(r).all())}",
                      span == 0.0 and s == 1.0 and bool(torch.isfinite(r).all())))
    except Exception as e:
        cases.append(("N=1 single-node graph", f"RAISED {type(e).__name__}: {e}", False))

    # node with no mention span (node 2 absent from the map)
    try:
        r = w.build_wire_signal([_path_graph(3)], [{0: [(1, 2)]}], seq_len=8, device=CPU)
        nz = (r[0].norm(dim=-1) > 0).nonzero().flatten().tolist()
        cases.append(("node with no mention span", f"placed={nz}", nz == [1]))
    except Exception as e:
        cases.append(("node with no mention span", f"RAISED {type(e).__name__}: {e}", False))

    # isolated (degree-0) nodes
    try:
        r = w.build_wire_signal([_isolated_graph(4)], [{i: [(i, i + 1)] for i in range(4)}],
                                seq_len=8, device=CPU)
        cases.append(("isolated degree-0 nodes",
                      f"finite={bool(torch.isfinite(r).all())} "
                      f"nan={bool(torch.isnan(r).any())}",
                      bool(torch.isfinite(r).all())))
    except Exception as e:
        cases.append(("isolated degree-0 nodes", f"RAISED {type(e).__name__}: {e}", False))

    # all-zero Psi via closed gate
    try:
        w0 = _wrap(pe_gain_init=0.0)
        r = w0.build_wire_signal([_path_graph(3)], [INJ], seq_len=8, device=CPU)
        with torch.no_grad():
            a = w0(input_ids=IDS, graphs=[_path_graph(3)], injection_maps=[INJ]).logits
            b = w0.llm(input_ids=IDS).logits
        cases.append(("all-zero Psi (gate=0)",
                      f"r_allzero={float(r.abs().max()) == 0.0} span={w0._wire_psi_span} "
                      f"logits_identical={float((a - b).abs().max()) == 0.0}",
                      float(r.abs().max()) == 0.0 and float((a - b).abs().max()) == 0.0))
    except Exception as e:
        cases.append(("all-zero Psi (gate=0)", f"RAISED {type(e).__name__}: {e}", False))

    # spans past seq_len
    try:
        r = w.build_wire_signal([_path_graph(3)], [{0: [(6, 99)]}], seq_len=8, device=CPU)
        nz = (r[0].norm(dim=-1) > 0).nonzero().flatten().tolist()
        cases.append(("span extending past seq_len", f"placed={nz}", nz == [6, 7]))
    except Exception as e:
        cases.append(("span extending past seq_len", f"RAISED {type(e).__name__}: {e}", False))

    for name, meas, ok in cases:
        rec(f"degenerate: {name}", "no crash / finite / expected placement", meas,
            "PASS" if ok else "FAIL")
    bad = [c for c in cases if not c[2]]
    assert not bad, f"degenerate-input failures: {[(n, m) for n, m, _ in bad]}"


def test_zzz_report():
    """Prints the surface table. Named to sort last under pytest's file order."""
    if not ROWS:
        return _skipall("no rows collected (fixture unavailable)")
    wid = [max(len(r[i]) for r in ROWS) for i in range(4)]
    hdr = ("PATH", "CHECKED", "MEASURED", "STATUS")
    wid = [max(wid[i], len(hdr[i])) for i in range(4)]
    line = "  ".join("-" * w for w in wid)
    print("\n" + line)
    print("  ".join(h.ljust(wid[i]) for i, h in enumerate(hdr)))
    print(line)
    for r in ROWS:
        print("  ".join(r[i].ljust(wid[i]) for i in range(4)))
    print(line)
    n_fail = sum(1 for r in ROWS if r[3] == "FAIL")
    print(f"{len(ROWS)} paths: {sum(1 for r in ROWS if r[3] == 'PASS')} PASS, "
          f"{n_fail} FAIL, {sum(1 for r in ROWS if r[3] == 'UNEXERCISED')} UNEXERCISED")
    assert n_fail == 0, f"{n_fail} surface paths FAILED"


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
