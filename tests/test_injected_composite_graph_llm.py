"""Verification battery for ``InjectedCompositeGraphLLM`` (q/k/v in-attention injection).

Target: ``prism.models.composite_graph_llm.InjectedCompositeGraphLLM`` — the
composite-graph wrapper that injects the Graph-Transformer code into attention at
every layer, in three modes:

  - **additive** (``pe_qk_injection``): ``q += W_q·S``, ``k += W_k·S``, ``v += W_v·S``;
  - **c_per_layer**: post-RoPE ``q ← C_tok·q``, ``k ← C_tok·k``;
  - **c_bias**: additive ``λ_C·Ĉ + λ_ψ·Ψ̃`` logit bias + residual ``λ_V·Ĉ`` value mix.

Dependency closure exercised for real (NOT stubbed): ``GraphTransformer`` (gt.py) →
``RandomGNNPositionalEncodings`` (r_pearl.py), ``build_composite_graph``
(composite_graph.py), ``GatedInjection`` + ``CompositeGraphLLM`` (parent), and the
RoPE-disable in ``prism.models.llama.disable_rope``. Boundaries stubbed/minimised:
the base LLM is a tiny random-init CPU model (no real Gemma-4 12B / Llama weights);
no SPINE message plumbing, no TRL/Hydra, no checkpoints.

Base LLM: a tiny random-init **Llama** — the architecture the composite-graph code
actually targets (the e7 configs pin ``llm=llama31_8b``); its Llama-native attention
interface is what the per-instance injection patch and ``disable_rope`` assume.

(NB: this wrapper is Llama-specific. With a Gemma-4 base it does not even
instantiate — ``disable_rope`` dereferences ``rotary_emb.inv_freq`` (absent on
``Gemma4UnifiedTextRotaryEmbedding``), and the patch unpacks ``cos, sin =
position_embeddings`` which Gemma-4 ships as a single tensor. ``GraphAugmentedLLM``
was migrated to an architecture-agnostic registered-attention path for exactly this;
``InjectedCompositeGraphLLM`` was not.)

Run:    python tests/test_injected_composite_graph_llm.py
Pytest: pytest tests/test_injected_composite_graph_llm.py -v
"""
import sys
sys.path.insert(0, "src")

import torch
from torch_geometric.data import Data

from prism.models.gt import GraphTransformer
from prism.models.composite_graph_llm import InjectedCompositeGraphLLM
from prism.models.llama import disable_rope, _IdentityRotaryEmbedding

_D = 32          # hidden / d_model
_C = 8           # prompt length == token-cycle length
_VOCAB = 64
_N_SCENE = 4


# --------------------------------------------------------------------------- #
# Fixtures — tiny, random-init, CPU.                                          #
# --------------------------------------------------------------------------- #
def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(
        vocab_size=_VOCAB, hidden_size=_D, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(cfg)


def _tiny_gt():
    return GraphTransformer(
        num_layers=2, pe_hidden_channels=16, pe_num_layers=2, d_model=_D,
        heads=2, num_samples=4, k_pe=2, k_gt=2, eps=1e-6,
        pe_readout="second_moment",
    )


def _llama_model(seed=0, **kw):
    """Build an ``InjectedCompositeGraphLLM`` over a tiny Llama with fixed seed.

    The base LLM is constructed FIRST (args evaluate before the call), so a sibling
    ``torch.manual_seed(seed); _tiny_llama()`` yields byte-identical base weights —
    used by the faithfulness test.
    """
    torch.manual_seed(seed)
    return InjectedCompositeGraphLLM(_tiny_llama(), _tiny_gt(), d_model=_D, **kw)


def _scene_and_inputs(batch=1):
    scene = Data(x=torch.zeros(_N_SCENE, 1),
                 edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
    scene.edge_weight = torch.ones(3)
    scene.num_nodes = _N_SCENE
    graphs = [scene for _ in range(batch)]
    inj = [{0: [(1, 3)], 2: [(4, 6)]} for _ in range(batch)]
    torch.manual_seed(123)
    input_ids = torch.randint(0, _VOCAB, (batch, _C))
    return input_ids, graphs, inj


# --------------------------------------------------------------------------- #
# Llama (native base) — the working contract.                                  #
# --------------------------------------------------------------------------- #
def test_llama_construct_default_rope_off():
    """Default (RoPE-off) construction succeeds on a Llama base — the architecture
    the composite-graph code actually targets (e7 configs pin ``llm=llama31_8b``)."""
    model = _llama_model(injection_mode="interpolate", gate_init=0.5)
    assert isinstance(model, InjectedCompositeGraphLLM)
    assert isinstance(model.llm.model.rotary_emb, _IdentityRotaryEmbedding)


def test_llama_attention_patch_forward_rope_on():
    """With RoPE left on (``disable_llm_rope=False``, isolating the attention patch
    from ``disable_rope``), a forward pass runs finite on a Llama base — the patch
    adds the GT code post-RoPE without disturbing the rotary path."""
    torch.manual_seed(0)
    model = InjectedCompositeGraphLLM(_tiny_llama(), _tiny_gt(), d_model=_D,
                                      injection_mode="interpolate", gate_init=0.5,
                                      disable_llm_rope=False)
    assert not isinstance(model.llm.model.rotary_emb, _IdentityRotaryEmbedding)
    ids, graphs, inj = _scene_and_inputs()
    with torch.no_grad():
        out = model(input_ids=ids, graphs=graphs, injection_maps=inj)
    assert torch.isfinite(out.logits).all()
def test_llama_construct_and_projection_shapes():
    """RoPE is disabled (identity rotary) and the dedicated code projections have
    the head-partitioned shapes the patch assumes."""
    model = _llama_model(injection_mode="interpolate", gate_init=0.5).eval()
    assert isinstance(model.llm.model.rotary_emb, _IdentityRotaryEmbedding)
    # head_dim = hidden // n_heads = 16; q: n_heads*head_dim, k/v: n_kv*head_dim.
    assert model.pe_q_proj.weight.shape == (2 * 16, _D)
    assert model.pe_k_proj.weight.shape == (1 * 16, _D)
    assert model.pe_v_proj.weight.shape == (1 * 16, _D)


def test_llama_signal_off_is_faithful_to_unpatched():
    """IMPL/parity: with ``_pe_signal=None`` the patched (RoPE-off) attention is
    byte-identical to an independent un-patched RoPE-off Llama with the same weights.
    Catches a patch that silently alters the base attention (e.g. dropped causal
    mask, wrong scaling)."""
    ids = torch.randint(0, _VOCAB, (1, _C))
    torch.manual_seed(0)
    ref_llm = disable_rope(_tiny_llama()).eval()
    X = ref_llm.get_input_embeddings()(ids)
    with torch.no_grad():
        ref = ref_llm(inputs_embeds=X).logits
    model = _llama_model(injection_mode="interpolate", gate_init=0.5).eval()
    model._pe_signal = None
    with torch.no_grad():
        got = model.llm(inputs_embeds=X).logits
    assert torch.allclose(ref, got, atol=1e-5), \
        f"patched signal-off forward diverges: max|Δ|={(ref - got).abs().max():.2e}"


def test_llama_additive_forward_shape_finite():
    """IFACE/RUNTIME: additive mode returns finite logits [B, C, vocab]."""
    model = _llama_model(injection_mode="interpolate", gate_init=0.5).eval()
    ids, graphs, inj = _scene_and_inputs()
    with torch.no_grad():
        out = model(input_ids=ids, graphs=graphs, injection_maps=inj)
    assert out.logits.shape == (1, _C, _VOCAB)
    assert torch.isfinite(out.logits).all()


def test_llama_injection_changes_logits():
    """IMPL/ablation: zeroing the dedicated W_q/W_k/W_v projections (no injected
    code) must change the logits vs. the live projections — the injection is not a
    no-op."""
    model = _llama_model(injection_mode="interpolate", gate_init=0.5).eval()
    ids, graphs, inj = _scene_and_inputs()
    with torch.no_grad():
        on = model(input_ids=ids, graphs=graphs, injection_maps=inj).logits
        saved = {n: getattr(model, n).weight.clone()
                 for n in ("pe_q_proj", "pe_k_proj", "pe_v_proj")}
        for n in saved:
            getattr(model, n).weight.zero_()
        off = model(input_ids=ids, graphs=graphs, injection_maps=inj).logits
        for n, w in saved.items():
            getattr(model, n).weight.copy_(w)
    assert not torch.allclose(on, off, atol=1e-5)


def test_llama_inject_v_toggle():
    """IFACE: ``inject_v=False`` omits ``pe_v_proj``; value injection toggles logits."""
    no_v = _llama_model(injection_mode="interpolate", gate_init=0.5, inject_v=False)
    assert getattr(no_v, "pe_v_proj", None) is None
    model = _llama_model(injection_mode="interpolate", gate_init=0.5).eval()
    ids, graphs, inj = _scene_and_inputs()
    with torch.no_grad():
        model._pe_inject_value = True
        a = model(input_ids=ids, graphs=graphs, injection_maps=inj).logits
        model._pe_inject_value = False
        b = model(input_ids=ids, graphs=graphs, injection_maps=inj).logits
        model._pe_inject_value = True
    assert not torch.allclose(a, b, atol=1e-5)


def test_llama_additive_backward_gradflow():
    """RUNTIME/grad: loss backprops finite, nonzero grads to the GT (R-PEARL) and to
    every dedicated code projection."""
    model = _llama_model(injection_mode="interpolate", gate_init=0.5).train()
    ids, graphs, inj = _scene_and_inputs(batch=2)
    out = model(input_ids=ids, graphs=graphs, injection_maps=inj, labels=ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    for n in ("pe_q_proj", "pe_k_proj", "pe_v_proj"):
        g = getattr(model, n).weight.grad
        assert g is not None and g.abs().sum() > 0, f"{n} got no grad"
    gt_grad = sum(1 for p in model.gt_model.parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0)
    assert gt_grad > 0, "Graph Transformer received no grad through the injection"
    assert not any(torch.isnan(p.grad).any()
                   for p in model.parameters() if p.grad is not None)


def test_llama_c_per_layer_forward_backward():
    """c_per_layer mode (q←C_tok·q): finite forward, GT grad, no dedicated projections."""
    model = _llama_model(injection_mode="none", c_per_layer=True, gate_init=0.3)
    assert getattr(model, "pe_q_proj", None) is None  # no projections in this mode
    model.train()
    ids, graphs, inj = _scene_and_inputs()
    out = model(input_ids=ids, graphs=graphs, injection_maps=inj, labels=ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    gt_grad = sum(1 for p in model.gt_model.parameters()
                  if p.grad is not None and p.grad.abs().sum() > 0)
    assert gt_grad > 0


def test_llama_c_bias_forward_backward():
    """c_bias mode: finite forward, switches base attn to SDPA, grads to λ_C/λ_ψ/λ_V."""
    model = _llama_model(injection_mode="interpolate", c_bias=True,
                         c_kernel="sampled", gate_init=0.3)
    assert model.llm.config._attn_implementation == "sdpa"
    model.train()
    ids, graphs, inj = _scene_and_inputs()
    out = model(input_ids=ids, graphs=graphs, injection_maps=inj, labels=ids)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    for n in ("lam_c", "lam_psi", "lam_v"):
        g = getattr(model, n).grad
        assert g is not None and torch.isfinite(g).all(), f"{n} grad missing/nonfinite"


def test_llama_c_bias_decode_extension():
    """c_bias decode path used by inference.py: ``decode_setup`` arms state and each
    ``decode_extend`` produces a finite bias row over the growing key sequence."""
    model = _llama_model(injection_mode="interpolate", c_bias=True,
                         c_kernel="sampled", gate_init=0.3).eval()
    ids, graphs, inj = _scene_and_inputs()
    aug = model._composite_graph(graphs[0], inj[0], _C, ids.device)
    model.decode_setup(aug, [(0, (11, 12)), (2, (13,))], _C, _C + 5, device=ids.device)
    for step, tok in enumerate([11, 12, 13], start=1):
        model.decode_extend(tok)
        row = model._pe_decode_row
        assert row.shape[0] == _C + step, f"row len {row.shape[0]} != {_C + step}"
        assert torch.isfinite(row).all()
    model.decode_disarm()
    assert model._decode_state is None and model._pe_decode_row is None


def test_llama_structural_parameters_cover_trained_groups():
    """train_v3 contract: ``GraphSFTTrainer`` re-enables grad on every tensor
    ``structural_parameters()`` reports. It must include the GT params and the
    mode-specific code params (additive: W_q/W_k/W_v; c_bias: λ gains)."""
    add = _llama_model(injection_mode="interpolate", gate_init=0.5)
    sp = {id(p) for p in add.structural_parameters()}
    assert all(id(p) in sp for p in add.gt_model.parameters())
    assert id(add.pe_q_proj.weight) in sp and id(add.pe_k_proj.weight) in sp

    cb = _llama_model(injection_mode="interpolate", c_bias=True, c_kernel="sampled",
                      gate_init=0.3)
    sp_cb = {id(p) for p in cb.structural_parameters()}
    assert id(cb.lam_c) in sp_cb and id(cb.lam_psi) in sp_cb and id(cb.lam_v) in sp_cb


def test_llama_generate_decode_loop():
    """Eval/inference contract: ``prepare_generation`` arms the signal and returns
    the [B,C,D] blend; ``llm.generate`` then runs the decode loop (injection
    auto-skips cached steps)."""
    model = _llama_model(injection_mode="interpolate", gate_init=0.5).eval()
    ids, graphs, inj = _scene_and_inputs()
    emb = model.prepare_generation(ids, graphs, inj)
    assert model._pe_signal is not None and tuple(emb.shape) == (1, _C, _D)
    with torch.no_grad():
        out = model.llm.generate(
            inputs_embeds=emb, max_new_tokens=3, min_new_tokens=3,
            do_sample=False, use_cache=True,
            pad_token_id=model.config.eos_token_id or 0,
        )
    model._pe_signal = None
    assert out.shape[1] == 3


def test_llama_signal_disarmed_after_forward():
    """No stale signal leakage: a non-checkpointed forward clears all _pe_* signals."""
    model = _llama_model(injection_mode="interpolate", gate_init=0.5).eval()
    ids, graphs, inj = _scene_and_inputs()
    with torch.no_grad():
        model(input_ids=ids, graphs=graphs, injection_maps=inj)
    assert model._pe_signal is None and model._pe_C is None and model._pe_Psi is None


def test_llama_fixed_seed_determinism():
    """Determinism: fixed-seed probes ⇒ identical logits across two forwards."""
    model = _llama_model(injection_mode="interpolate", gate_init=0.5).eval()
    model.gt_model.pe_model.fixed_seed_mode = True
    ids, graphs, inj = _scene_and_inputs()
    with torch.no_grad():
        a = model(input_ids=ids, graphs=graphs, injection_maps=inj).logits
        b = model(input_ids=ids, graphs=graphs, injection_maps=inj).logits
    assert torch.allclose(a, b, atol=1e-5)


def test_llama_batch_forward_finite():
    """Batch handling: B>1 forward is finite and correctly shaped."""
    model = _llama_model(injection_mode="interpolate", gate_init=0.5).eval()
    ids, graphs, inj = _scene_and_inputs(batch=2)
    with torch.no_grad():
        out = model(input_ids=ids, graphs=graphs, injection_maps=inj)
    assert out.logits.shape == (2, _C, _VOCAB) and torch.isfinite(out.logits).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"{name}: PASS")
            except Exception as e:  # noqa: BLE001 — script runner reports, doesn't halt
                print(f"{name}: FAIL — {type(e).__name__}: {str(e)[:200]}")
    print("done")
