"""End-to-end smoke test of ``WireGraphLLM`` on Apple silicon (torch + MPS).

WHY torch/MPS AND NOT MLX
------------------------
``WireGraphLLM`` cannot be hosted by MLX without a reimplementation. It is a
``transformers.PreTrainedModel`` subclass (``gnn_llm.py:1166``) whose entire injection
mechanism is HF/PyTorch machinery: ``AttentionInterface.register(_WIRE_IMPL, ...)``
(``gnn_llm.py:294``), per-layer ``attn.config._attn_implementation = _WIRE_IMPL``
(``gnn_llm.py:1579``), delegation through ``transformers.masking_utils`` (``:324``) and
``mod.ALL_ATTENTION_FUNCTIONS`` (``:317``), plus ``nn.Parameter``/``register_buffer``
state and a PyTorch-Geometric ``GraphTransformer`` Ψ producer. MLX has no
``AttentionInterface``, no ``_attn_implementation`` dispatch, and no PyG. So MPS it is —
the supported Apple-silicon path here.

THE FIXTURE
-----------
The real ``google/gemma-4-31B-it`` weights are not on disk (config + tokenizer only) and
57 GiB of bf16 would not fit the 37 GiB MPS budget anyway. This builds a ~29 M-parameter
random-init ``Gemma4ForCausalLM`` whose config MIRRORS the 31B text config in every
structural feature the WIRE code branches on — read from
``~/.cache/huggingface/hub/models--google--gemma-4-31B-it/.../config.json``:

    feature                      31B                       fixture
    num_hidden_layers            60                        24
    layer_types                  full @ 5,11,...,59        full @ 5,11,17,23  (same 1:6)
    globals / slidings           10 / 50                   4 / 20
    head_dim (sliding)           256                       16
    global_head_dim              512                       32
    num_attention_heads          32                        8
    num_key_value_heads          16   (GQA 2x)             4   (GQA 2x)
    num_global_key_value_heads   4    (GQA 8x)             1   (GQA 8x)
    attention_k_eq_v             true (v_proj None)        true
    num_kv_shared_layers         0                         0
    rope full_attention          proportional, prf 0.25    same
    rope sliding_attention       default, theta 1e4        same
    hidden / intermediate        5376 / 21504 (4x)         96 / 384 (4x)
    vocab_size                   262144                    262144 (real tokenizer)
    final_logit_softcapping      30.0                      30.0
    sliding_window               1024                      64
    tie_word_embeddings          true                      true

FOUR global layers is the point: the shipped 4-layer fixture in ``test_wire_smoke.py``
has exactly ONE, so ``dense`` / ``dense_top_half`` / ``dense_first`` all collapse to the
same active set there and are exercised but not DISTINGUISHED. Here they resolve to
{5,11,17,23} / {17,23} / {5}.

Run: uv run --with pytest -m pytest tests/test_wire_mps_smoke.py -q -s
"""
import os
import sys
import time
import warnings

sys.path.insert(0, "src")

import pytest
import torch
from torch import nn

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")

# ------------------------------------------------------------------ constants

BASE_MODEL = "google/gemma-4-31B-it"          # config + tokenizer only, no weights
TRAIN_FILES = "data/n_30/gen/nav100_n30_gemma_data/split/formatted_all_new__train.json"
BATCH = 2
MAX_LEN = 1536            # full n_30 sequences are ~1.4k; no truncation in practice
STEPS = 3
LR = 2e-4                 # experiments/base_config.yaml trainer.learning_rate

ROWS: list[tuple[str, str, str, str]] = []


def rec(path, checked, measured, status):
    ROWS.append((path, checked, str(measured), status))


# ------------------------------------------------------------------- device

def _device():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable — refusing to substitute device=cpu silently")
    return torch.device("mps")


def _mem_gb():
    """(live tensors, driver total) in GiB.

    torch.mps has no ``max_memory_allocated``, so 'peak' is not directly available:
    ``current_allocated_memory`` is live tensor bytes at the call, and
    ``driver_allocated_memory`` is the total the driver holds INCLUDING the cached
    allocator pool — an upper bound on the true peak, not the peak itself. Both are
    reported rather than one being passed off as the peak.
    """
    cur = getattr(torch.mps, "current_allocated_memory", lambda: 0)()
    drv = getattr(torch.mps, "driver_allocated_memory", lambda: 0)()
    return cur / 2**30, drv / 2**30


# ------------------------------------------------- MPS sparse-CSR workaround

_SPARSE_CSR_ON_MPS: bool | None = None
_SPARSE_CSR_ERR = ""


def sparse_csr_available_on_mps() -> bool:
    """Whether ``gt.py``'s CSR sparse attention can run natively on MPS.

    ``SparseGraphAttention.forward`` (``src/prism/models/gt.py:54-56``) builds
    ``torch.sparse_coo_tensor(...).coalesce().to_sparse_csr()`` on ``x.device`` and
    ``_SafeBatchedSparseAttn`` then calls ``torch.sparse.sampled_addmm`` /
    ``torch.sparse_csr_tensor`` on it. torch 2.10 registers NO CSR kernels for MPS.
    Probed at runtime so this test starts passing natively the day torch adds them.
    """
    global _SPARSE_CSR_ON_MPS, _SPARSE_CSR_ERR
    if _SPARSE_CSR_ON_MPS is None:
        try:
            ei = torch.tensor([[0, 1], [1, 0]], device="mps")
            torch.sparse_coo_tensor(
                ei, torch.ones(2, device="mps"), (2, 2)).coalesce().to_sparse_csr()
            _SPARSE_CSR_ON_MPS = True
        except Exception as e:                        # noqa: BLE001 — any failure = unsupported
            _SPARSE_CSR_ON_MPS = False
            _SPARSE_CSR_ERR = str(e).split(".")[0]
    return _SPARSE_CSR_ON_MPS


class CpuHostedPsi(nn.Module):
    """Runs the Ψ producer on CPU and hands Ψ back on the LLM's device.

    NOT a workaround the repo ships — it exists only so the WIRE/LLM half of the
    pipeline (rotation, attention, LLM forward/backward, AdamW) can be measured on MPS
    while ``gt.py``'s CSR sparse attention is unrunnable there. Installed by the test,
    never by ``build_planner_model``; every row produced under it is labelled.
    """

    def __init__(self, pe_model: nn.Module, out_device):
        super().__init__()
        self.pe_model = pe_model.to("cpu")
        self.out_device = out_device

    def forward(self, graph, permutation=None):
        return self.pe_model(graph.to("cpu"), permutation=permutation).to(self.out_device)


# ------------------------------------------------------------------ fixture

def gemma4_31b_shaped(seed: int = 0, dtype=torch.float32):
    """Tiny random-init ``Gemma4ForCausalLM`` mirroring the 31B text config.

    Same idiom as ``tests/test_wire_smoke.py::_gemma4`` and
    ``tests/test_pe_injection_parity.py`` — a ``Gemma4TextConfig`` built by hand — only
    scaled to keep the 31B structure instead of the minimum that constructs.
    """
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    torch.manual_seed(seed)
    cfg = Gemma4TextConfig(
        vocab_size=262_144,                # real tokenizer
        hidden_size=96,
        intermediate_size=384,             # 4x hidden, as 21504 = 4 x 5376
        num_hidden_layers=24,              # 4 globals under the default 1:6 pattern
        num_attention_heads=8,
        num_key_value_heads=4,             # GQA 2x, as 32/16
        num_global_key_value_heads=1,      # GQA 8x, as 32/4
        head_dim=16,                       # sliding
        global_head_dim=32,                # global: 2x sliding, as 512 vs 256
        attention_k_eq_v=True,             # => v_proj is None on global layers
        num_kv_shared_layers=0,
        max_position_embeddings=4096,
        sliding_window=64,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=262_144,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        rms_norm_eps=1e-6,
        attention_bias=False,
        attn_implementation="eager",
        # rope_parameters left to the class default, which IS the 31B value:
        #   full_attention  -> proportional, partial_rotary_factor 0.25, theta 1e6
        #   sliding_attention -> default, theta 1e4
    )
    return Gemma4ForCausalLM(cfg).to(dtype)


def _tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    tok.padding_side = "right"
    return tok


def _gnn_cfg(**overrides):
    """``experiments/base_config.yaml`` gnn section, arch=wire_llm, scaled where noted."""
    import omegaconf
    cfg = omegaconf.OmegaConf.load("experiments/base_config.yaml")
    gnn = cfg.gnn
    gnn.arch = "wire_llm"
    # Only the Ψ-producer width is shrunk (1024 -> 128) to keep the GT cheap; every
    # wire_* switch keeps its shipped value unless a test overrides it.
    gnn.d_model = 128
    gnn.pe_hidden_channels = 64
    gnn.num_samples = 8
    for k, v in overrides.items():
        gnn[k] = v
    return gnn


def _dataset(tokenizer, n=BATCH):
    """Real populated n_30 graphs through the repo's own preprocessing pipeline."""
    import omegaconf
    from prism.data import data
    cfg = omegaconf.OmegaConf.load("experiments/base_config.yaml").data
    cfg.train_files = TRAIN_FILES
    cfg.val_files = None
    cfg.val_frac = 0.0
    cfg.debug = True
    cfg.dataset_proportion = 0.01
    train, _ = data.load_and_split_dataset(cfg, tokenizer)
    return [train[i] for i in range(min(n, len(train)))]


def _batch(collator, examples, device):
    """Collate through ``SpineDataCollator``, then split model kwargs from index columns."""
    feats = []
    for e in examples:
        f = dict(e)
        f["input_ids"] = f["input_ids"][:MAX_LEN]
        f["attention_mask"] = f["attention_mask"][:MAX_LEN]
        feats.append(f)
    batch = collator(feats)
    idx_cols = {k: batch.pop(k) for k in list(batch) if k.endswith("_idx")}
    graphs = batch.pop("graphs").to(device)
    inj = batch.pop("injection_maps")
    tensors = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    return tensors, graphs, inj, idx_cols


def _mask_labels(inputs, idx_cols):
    """loss_target='responses' (base_config default) — REUSES the trainer's masker."""
    from prism.training.trainers import LossTargetMixin
    LossTargetMixin._mask_labels_to_positions(
        object.__new__(LossTargetMixin), inputs, idx_cols["assistant_idx"], "responses")


def _build(gnn, tokenizer, device, seed=0):
    from prism.models.architectures import build_planner_model
    llm = gemma4_31b_shaped(seed=seed).to(device)
    model, collator = build_planner_model(gnn, llm, tokenizer)
    model = model.to(device)
    if device.type == "mps" and not sparse_csr_available_on_mps():
        model.pe_model = CpuHostedPsi(model.pe_model, device)
    return model, collator


# =========================================================== 1. fixture gate

def test_fixture_mirrors_31b_structure():
    """Gating acceptance test: every 31B feature WIRE branches on is really present.

    Run first — if the fixture silently lost (say) the head_dim asymmetry or the
    partial_rotary_factor, every downstream row would be measuring the wrong model.
    """
    import json
    import glob

    from prism.models.gnn_llm import wire_rope_planes

    real_path = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/*/config.json"))
    real = json.load(open(real_path[0]))["text_config"] if real_path else None

    llm = gemma4_31b_shaped()
    cfg = llm.config
    layers = llm.model.layers
    glob_idx = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    slid_idx = [i for i, t in enumerate(cfg.layer_types) if t == "sliding_attention"]
    ga, sa = layers[glob_idx[0]].self_attn, layers[slid_idx[0]].self_attn

    checks = {
        "both layer kinds present": (len(glob_idx) > 0 and len(slid_idx) > 0, True),
        ">=4 global layers": (len(glob_idx), 4),
        "global every 6th (1:6 as 31B)": (glob_idx, [5, 11, 17, 23]),
        "head_dim asymmetry global>sliding": ((ga.head_dim, sa.head_dim), (32, 16)),
        "GQA groups sliding": (sa.num_key_value_groups, 2),
        "GQA groups global": (ga.num_key_value_groups, 8),
        "attention_k_eq_v => v_proj None on global": (ga.v_proj is None, True),
        "sliding keeps a v_proj": (sa.v_proj is not None, True),
        "num_kv_shared_layers=0 => no kv-shared layer": (
            any(l.self_attn.is_kv_shared_layer for l in layers), False),
        "rope full=proportional prf=0.25": (
            (cfg.rope_parameters["full_attention"]["rope_type"],
             cfg.rope_parameters["full_attention"]["partial_rotary_factor"]),
            ("proportional", 0.25)),
        "rope sliding=default": (
            cfg.rope_parameters["sliding_attention"]["rope_type"], "default"),
        "MC planes differ by layer type (rotate_nope=False)": (
            (wire_rope_planes(ga, False), wire_rope_planes(sa, False)), (4, 8)),
        "rotate_nope=True widens ONLY the global layer": (
            (wire_rope_planes(ga, True), wire_rope_planes(sa, True)), (16, 8)),
        "attention_bias False": (cfg.attention_bias, False),
    }
    bad = [f"{k}: got {g!r} want {w!r}" for k, (g, w) in checks.items() if g != w]
    for k, (g, w) in checks.items():
        rec(f"fixture: {k}", "matches scaled 31B", f"{g!r}",
            "PASS" if g == w else "FAIL")
    if real is not None:
        same_shape = (
            real["attention_k_eq_v"] is True
            and real["num_kv_shared_layers"] == 0
            and real["global_head_dim"] == 2 * real["head_dim"]
            and real["num_attention_heads"] // real["num_key_value_heads"] == 2
            and real["num_attention_heads"] // real["num_global_key_value_heads"] == 8
            and real["rope_parameters"]["full_attention"]["partial_rotary_factor"] == 0.25
            and real["layer_types"].count("full_attention") * 6 == len(real["layer_types"]))
        rec("fixture: cross-checked against the REAL 31B config.json",
            "ratios/flags read from disk, not memory",
            f"31B: {real['num_hidden_layers']}L "
            f"{real['layer_types'].count('full_attention')} global, hd "
            f"{real['head_dim']}/{real['global_head_dim']}, k_eq_v="
            f"{real['attention_k_eq_v']}, prf="
            f"{real['rope_parameters']['full_attention']['partial_rotary_factor']}",
            "PASS" if same_shape else "FAIL")
        assert same_shape, "the ratios this fixture mirrors are not the ones on disk"
    else:
        rec("fixture: cross-check vs real config.json", "config.json on disk",
            "not found in the HF cache", "UNEXERCISED")
    assert not bad, "; ".join(bad)


def test_layer_scopes_are_distinguished():
    """The gap this fixture exists to close: dense / dense_top_half / dense_first must
    resolve to three DIFFERENT active sets (they collapse to one on a 4-layer fixture)."""
    from prism.models.gnn_llm import resolve_mask_active_flags

    layers = gemma4_31b_shaped().model.layers
    sets = {}
    for scope in ("all", "dense", "dense_top_half", "dense_first"):
        flags = resolve_mask_active_flags(layers, scope)
        sets[scope] = [i for i, f in enumerate(flags) if f]
    distinct = len({tuple(v) for v in sets.values()}) == 4
    rec("wire_layer_scope resolution", "4 scopes -> 4 DISTINCT active sets",
        f"all={len(sets['all'])} layers, dense={sets['dense']}, "
        f"top_half={sets['dense_top_half']}, first={sets['dense_first']}",
        "PASS" if distinct else "FAIL")
    assert sets["dense"] == [5, 11, 17, 23]
    assert sets["dense_top_half"] == [17, 23]
    assert sets["dense_first"] == [5]
    assert distinct, f"scopes collapsed: {sets}"


def test_gt_psi_producer_on_mps():
    """DEFECT PROBE: can the shipped Ψ producer run on MPS at all?

    ``build_planner_model(arch='wire_llm')`` builds a ``gt.GraphTransformer``, whose
    ``SparseGraphAttention.forward`` calls ``.to_sparse_csr()`` on ``x.device``. If MPS
    has no CSR kernels this fails for EVERY graph architecture (wire_llm,
    rpearl_gt_llm, learnable_graph_mask), not just WIRE. Recorded, not asserted, so the
    row flips to PASS by itself when torch grows the kernels.
    """
    device = _device()
    from prism.models import gt as gt_module
    from torch_geometric.data import Data

    torch.manual_seed(0)
    g = Data(x=torch.randn(6, 1),
             edge_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]]))
    g.num_nodes = 6
    net = gt_module.GraphTransformer(
        num_layers=2, pe_hidden_channels=16, pe_num_layers=2, d_model=32, heads=2,
        num_samples=4, dropout=0.0, k_pe=2, k_gt=2).to(device)
    err = ""
    try:
        out = net(g.to(device))
        ok = torch.isfinite(out).all().item()
    except Exception as e:                            # noqa: BLE001
        ok, err = False, f"{type(e).__name__}: {str(e).split('.')[0]}"
    rec("gt.GraphTransformer (Ψ producer) native on MPS",
        "forward on device=mps without a CPU detour",
        "finite Ψ" if ok else err,
        "PASS" if ok else "BLOCKED")
    if not ok:
        rec("=> WIRE runs on MPS with Ψ hosted on CPU (test-local CpuHostedPsi)",
            "rotation + LLM fwd/bwd + AdamW on MPS; only the sparse GT on CPU",
            "workaround active for every MPS row below", "PASS")


# ================================================ 2. the training smoke loop

def _train_steps(model, collator, examples, device, steps=STEPS, label="default"):
    """Hand-rolled loop: forward -> loss -> backward -> AdamW.step -> zero_grad."""
    model.train()
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR)
    losses, times, telem = [], [], None
    grads = {}
    for step in range(steps):
        inputs, graphs, inj, idx_cols = _batch(collator, examples, device)
        inputs["labels"] = inputs["input_ids"].clone()
        _mask_labels(inputs, idx_cols)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        out = model(graphs=graphs, injection_maps=inj, **inputs)
        loss = out.loss
        loss.backward()
        if step == 0:
            grads = _grad_report(model)
            telem = model.wire_telemetry()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        times.append(time.perf_counter() - t0)
        losses.append(float(loss.detach()))
    live_gb, drv_gb = _mem_gb()
    return {"losses": losses, "times": times, "grads": grads, "telemetry": telem,
            "live_gb": live_gb, "drv_gb": drv_gb, "label": label}


def _grad_report(model):
    """max|grad| per graph-side component, plus the ε-gets-nothing invariant."""
    def m(p):
        return None if p.grad is None else float(p.grad.abs().max())

    pe = [m(p) for p in model.pe_model.parameters()]
    pe = [x for x in pe if x is not None]
    sigmas = {int(k): m(p) for k, p in model._wire_sigma.items()}
    eps_grad = {i: getattr(model._wire_eps, str(i)).grad
                for i in model.active_layer_indices()}
    eps_rg = {i: getattr(model._wire_eps, str(i)).requires_grad
              for i in model.active_layer_indices()}
    all_finite = all(torch.isfinite(p.grad).all()
                     for p in model.parameters() if p.grad is not None)
    return {
        "pe_model_max": max(pe) if pe else None,
        "pe_model_none": sum(1 for p in model.pe_model.parameters() if p.grad is None),
        "pe_gain": m(model.pe_gain),
        "sigma": sigmas,
        "eps_all_none": all(v is None for v in eps_grad.values()),
        "eps_requires_grad": any(eps_rg.values()),
        "all_finite": bool(all_finite),
    }


def test_train_loop_default_config():
    """The headline run: base_config wire_* defaults, real n_30 batch, 3 AdamW steps."""
    device = _device()
    tok = _tokenizer()
    examples = _dataset(tok)
    # pe_gain_init: base_config ships 0.0 (gate closed => WIRE is the identity rotation
    # at step 0, and the whole channel is numerically inert). Opened to 1.0 here so the
    # rotation is actually live; the closed-gate side is measured separately below.
    gnn = _gnn_cfg(pe_gain_init=1.0)
    model, collator = _build(gnn, tok, device)

    scopes = [i for i, l in enumerate(model._decoder_layers())
              if getattr(l.self_attn, "_wire_active", False)]
    rec("build_planner_model(arch=wire_llm)", "constructs WireGraphLLM on MPS",
        f"{type(model).__name__}, active layers {scopes}, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params",
        "PASS" if scopes == [5, 11, 17, 23] else "FAIL")
    assert scopes == [5, 11, 17, 23]

    inputs, graphs, inj, idx_cols = _batch(collator, examples, device)
    n_inj = sum(len(v) for m in inj for v in m.values())
    rec("SpineDataCollator over real data/n_30", "graphs + injection maps reach the model",
        f"input_ids {tuple(inputs['input_ids'].shape)}, graph N={graphs.num_nodes} "
        f"E={graphs.num_edges}, {n_inj} node-mention spans",
        "PASS" if n_inj > 0 and graphs.num_nodes > 0 else "FAIL")
    assert n_inj > 0

    # NON-VACUITY: with only 4 rotated planes at ~0.09 rad the channel could be
    # numerically inert, which would make every row below pass for the wrong reason.
    model.eval()
    with torch.no_grad():
        on = model(graphs=graphs, injection_maps=inj, **inputs).logits
        off = model(**inputs).logits                       # no graph => stock LLM
        stock = model.llm(**inputs).logits
    d_on = float((on - stock).abs().max())
    d_off = float((off - stock).abs().max())
    rec("WIRE is not inert on this fixture",
        "graph-armed logits differ from stock; no-graph logits identical to stock",
        f"max|Δ| armed={d_on:.3e}, unarmed={d_off:.3e}",
        "PASS" if (d_on > 1e-4 and d_off == 0.0) else "FAIL")
    assert d_on > 1e-4, f"WIRE inert (max|Δ|={d_on:.2e}) — the smoke test would be vacuous"
    assert d_off == 0.0, f"no-graph forward is not the stock LLM (max|Δ|={d_off:.2e})"

    r = _train_steps(model, collator, examples, device)

    finite = all(torch.isfinite(torch.tensor(l)) for l in r["losses"])
    changing = len(set(f"{l:.6f}" for l in r["losses"])) == len(r["losses"])
    rec("training loop: loss", "finite AND changing across 3 optimizer steps",
        f"{['%.5f' % l for l in r['losses']]}",
        "PASS" if finite and changing else "FAIL")

    g = r["grads"]
    sig_vals = list(g["sigma"].values())
    sig_ok = all(v is not None and v == v for v in sig_vals)
    grad_ok = (g["pe_model_max"] is not None and g["pe_gain"] is not None
               and sig_ok and g["eps_all_none"] and not g["eps_requires_grad"]
               and g["all_finite"])
    rec("gradients", "pe_model / pe_gain / per-layer sigma non-None+finite; eps none",
        f"pe_model max={g['pe_model_max']:.3e} ({g['pe_model_none']} params with no grad), "
        f"pe_gain={g['pe_gain']:.3e}, sigma="
        f"{ {k: '%.3e' % v for k, v in g['sigma'].items()} }, "
        f"eps_grad_all_None={g['eps_all_none']} eps_requires_grad={g['eps_requires_grad']} "
        f"all_finite={g['all_finite']}",
        "PASS" if grad_ok else "FAIL")

    t = r["telemetry"]
    rec("wire_telemetry", "raw vs effective sigma, measured angle, clamp state",
        f"psi_span={t['wire/psi_span']:.4f} sigma_raw_max={t['wire/sigma_raw_max']:.4g} "
        f"sigma_eff_max={t['wire/sigma_eff_max']:.4g} angle_raw={t['wire/angle_raw_max']:.4f} "
        f"angle_eff={t['wire/angle_eff_max']:.4f} max_angle={t['wire/max_angle']} "
        f"clamp_engaged={t['wire/clamp_engaged']} clamped_layers={t['wire/clamped_layers']} "
        f"scale_min={t['wire/scale_min']:.4g}",
        "PASS")
    rec("cost", "wall clock per step / peak MPS memory",
        f"steps {['%.2fs' % x for x in r['times']]}, live tensors "
        f"{r['live_gb']:.2f} GiB, driver-held (incl. cache, upper bound) "
        f"{r['drv_gb']:.2f} GiB",
        "PASS")

    assert finite, f"non-finite loss: {r['losses']}"
    assert changing, f"loss did not move across steps: {r['losses']}"
    assert grad_ok, f"gradient invariants violated: {g}"


# ============================================= 3. the non-default config sides

_SIDES = [
    ("wire_layer_scope=all", {"wire_layer_scope": "all"}),
    ("wire_layer_scope=dense", {"wire_layer_scope": "dense"}),
    ("wire_layer_scope=dense_top_half", {"wire_layer_scope": "dense_top_half"}),
    ("wire_layer_scope=dense_first", {"wire_layer_scope": "dense_first"}),
    ("wire_rotate_nope_planes=True", {"wire_rotate_nope_planes": True}),
    ("wire_freeze_sigma=True", {"wire_freeze_sigma": True}),
    ("pe_gain_init=0.0 (shipped default: gate closed)", {"pe_gain_init": 0.0}),
    ("wire_sigma_init=1.0 (drives the angle clamp)", {"wire_sigma_init": 1.0}),
]

_EXPECT_ACTIVE = {"all": list(range(24)), "dense": [5, 11, 17, 23],
                  "dense_top_half": [17, 23], "dense_first": [5]}


@pytest.mark.parametrize("label,over", _SIDES, ids=[s[0] for s in _SIDES])
def test_config_sides(label, over):
    """Each non-default config side: 2 real steps, same invariants, on MPS."""
    device = _device()
    tok = _tokenizer()
    examples = _dataset(tok)
    over = dict(over)
    over.setdefault("pe_gain_init", 1.0)
    gnn = _gnn_cfg(**over)
    model, collator = _build(gnn, tok, device)

    active = [i for i, l in enumerate(model._decoder_layers())
              if getattr(l.self_attn, "_wire_active", False)]
    scope = gnn.wire_layer_scope
    assert active == _EXPECT_ACTIVE[scope], f"{scope}: active {active}"
    assert active == model.active_layer_indices() or scope == "all"

    # rotate_nope widens ONLY the global layers' eps tables.
    planes = {i: getattr(model._wire_eps, str(i)).shape[0]
              for i in model.active_layer_indices()}
    frozen = [p.requires_grad for p in model._wire_sigma.values()]
    in_group = {id(p) for p in model.structural_parameters()}
    sigma_in_group = [id(p) in in_group for p in model._wire_sigma.values()]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = _train_steps(model, collator, examples, device, steps=2, label=label)
    clamp_warn = any("clamp engaged" in str(w.message) for w in caught)

    g = r["grads"]
    t = r["telemetry"]
    finite = all(l == l and abs(l) != float("inf") for l in r["losses"])
    changing = r["losses"][0] != r["losses"][1]
    sigma_grads = [v for v in g["sigma"].values()]
    if gnn.wire_freeze_sigma:
        sigma_ok = all(v is None for v in sigma_grads) and not any(frozen) \
            and not any(sigma_in_group)
    else:
        sigma_ok = all(v is not None for v in sigma_grads) and all(frozen) \
            and all(sigma_in_group)
    ok = (finite and changing and sigma_ok and g["eps_all_none"]
          and g["all_finite"] and g["pe_gain"] is not None)

    rec(f"config side: {label}",
        "active layers / eps planes / sigma grad+LR-group / loss / clamp",
        f"active={active if len(active) < 8 else str(len(active)) + ' layers'} "
        f"planes={sorted(set(planes.values()))} "
        f"sigma_requires_grad={frozen[0]} sigma_in_structural_group={sigma_in_group[0]} "
        f"loss={['%.5f' % x for x in r['losses']]} "
        f"angle_raw={t['wire/angle_raw_max']:.4f} angle_eff={t['wire/angle_eff_max']:.4f} "
        f"clamp={t['wire/clamp_engaged']}({t['wire/clamped_layers']}) warn={clamp_warn} "
        f"pe_gain_grad={g['pe_gain']:.3e} {r['times'][0]:.2f}s/step "
        f"live={r['live_gb']:.2f}GiB drv<={r['drv_gb']:.2f}GiB",
        "PASS" if ok else "FAIL")

    assert finite, f"{label}: non-finite loss {r['losses']}"
    assert changing, f"{label}: loss frozen across steps {r['losses']}"
    assert sigma_ok, f"{label}: sigma grad/LR-group wrong (frozen={gnn.wire_freeze_sigma})"
    assert g["eps_all_none"], f"{label}: eps received a gradient"
    assert g["all_finite"], f"{label}: non-finite gradient"

    # Clamp arithmetic: effective angle never exceeds max_angle, and it engages
    # exactly when the raw angle would have exceeded it.
    assert t["wire/angle_eff_max"] <= t["wire/max_angle"] * (1 + 1e-5)
    should_clamp = t["wire/angle_raw_max"] > t["wire/max_angle"] * (1 + 1e-9)
    assert bool(t["wire/clamp_engaged"]) == should_clamp, (
        f"{label}: clamp_engaged={t['wire/clamp_engaged']} but raw angle "
        f"{t['wire/angle_raw_max']} vs max {t['wire/max_angle']}")


# ------------------------------------------------------------------- report

def test_zzz_report():
    if not ROWS:
        pytest.skip("no rows collected")
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
    print(f"{len(ROWS)} rows: {sum(1 for r in ROWS if r[3] == 'PASS')} PASS, "
          f"{n_fail} FAIL, {sum(1 for r in ROWS if r[3] == 'UNEXERCISED')} UNEXERCISED")
    assert n_fail == 0, f"{n_fail} rows FAILED"
