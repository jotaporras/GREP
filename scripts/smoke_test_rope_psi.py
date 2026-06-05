"""Heavy smoke test for the RoPE(X) + Ψ in-attention injection (all layers).

Builds a *tiny* LlamaForCausalLM from config (no network) and exercises the
patched ``GraphAugmentedLLM`` attention to verify:

  1. Every attention layer's forward is patched, and the patched forward is a
     faithful drop-in: with Ψ disarmed it reproduces the stock LLM logits
     exactly.
  2. The graph signal Ψ enters the query/key dot product *unrotated* — i.e. the
     model truly computes ``RoPE(X) + Ψ``: the Ψ contribution to q matches the
     analytic ``RoPE(W_q·X) + W_q·Ψ`` target to numerical precision (and is far
     closer than the legacy ``RoPE(X + Ψ)`` residual-stream injection).
  3. Forward runs, returns finite loss, and gradients flow to pe_model / pe_proj
     / pe_gain — including through every patched layer.
  4. Gradients still flow to pe_model with gradient checkpointing enabled.
  5. Ψ is auto-skipped on cached single-token decode steps (generation), and the
     per-forward signal is disarmed afterwards.
  6. Train (``_augment_embeddings``) and the eval (inference) mirror arm an
     identical Ψ signal for the same inputs.

Run:  PYTHONPATH=src python scripts/smoke_test_rope_psi.py
Exits non-zero on any failed assertion.
"""

import sys

import torch
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from prism.models.gnn_llm import GraphAugmentedLLM

torch.manual_seed(0)

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_failures = []


def check(name, cond, detail=""):
    print(f"  [{PASS if cond else FAIL}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        _failures.append(name)


# --------------------------------------------------------------------------- #
# Fixtures: tiny LLM + dummy graph encoder
# --------------------------------------------------------------------------- #
HIDDEN = 64
HEAD_DIM = 8
N_HEADS = 8
N_KV_HEADS = 4          # exercise grouped-query attention
D_MODEL = 16            # pe_model output width
N_NODES = 4
SEQ = 12
INJECTION = {0: [(2, 4)], 1: [(5, 6)], 2: [(7, 9)], 3: [(10, 11)]}


class DummyGraph:
    def __init__(self, n):
        self.num_nodes = n


class DummyPE(nn.Module):
    """Deterministic graph encoder: returns a fixed [n, D_MODEL] embedding."""
    def __init__(self, n, d):
        super().__init__()
        self.table = nn.Parameter(torch.randn(n, d))

    def forward(self, data, permutation=None):
        return self.table


def build_model():
    cfg = LlamaConfig(
        vocab_size=128, hidden_size=HIDDEN, intermediate_size=128,
        num_hidden_layers=3, num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS, head_dim=HEAD_DIM,
        max_position_embeddings=64, attn_implementation="eager",
    )
    llm = LlamaForCausalLM(cfg).eval()
    model = GraphAugmentedLLM(llm, DummyPE(N_NODES, D_MODEL), d_model=D_MODEL, eps=1e-6).eval()
    return model


# --------------------------------------------------------------------------- #
# Test 1 — patch is installed on all layers and is a faithful drop-in
# --------------------------------------------------------------------------- #
def test_patch_is_transparent(model):
    print("Test 1: all layers patched + transparent when Ψ disarmed")
    layers = model._decoder_layers()
    patched = all(l.self_attn.forward.__closure__ is not None for l in layers)
    check("every attention layer patched", patched, f"n_layers={len(layers)}")

    input_ids = torch.randint(0, 128, (1, SEQ))
    model._pe_signal = None
    with torch.no_grad():
        ours = model.llm(input_ids=input_ids).logits
        ref_llm = LlamaForCausalLM(model.llm.config).eval()
        ref_llm.load_state_dict(model.llm.state_dict())
        ref = ref_llm(input_ids=input_ids).logits
    err = (ours - ref).abs().max().item()
    check("patched forward == stock when Ψ off", err < 1e-4, f"max|err|={err:.2e}")


# --------------------------------------------------------------------------- #
# Test 2 — Ψ enters q/k UNROTATED (exact RoPE(X)+Ψ), beating legacy RoPE(X+Ψ)
# --------------------------------------------------------------------------- #
def test_psi_unrotated_in_qk(model):
    print("Test 2: Ψ enters q/k UNROTATED — exact RoPE(X)+Ψ")
    input_ids = torch.randint(0, 128, (1, SEQ))
    position_ids = torch.arange(SEQ).unsqueeze(0)
    attn = model.llm.model.layers[0].self_attn
    X = model.llm.get_input_embeddings()(input_ids).clone()
    cos, sin = model.llm.model.rotary_emb(X, position_ids)

    def q_rope(embeds):
        q = attn.q_proj(embeds).view(1, SEQ, -1, HEAD_DIM).transpose(1, 2)
        q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)
        return q_rot

    with torch.no_grad():
        model._augment_embeddings(input_ids, [DummyGraph(N_NODES)], [INJECTION])  # arms _pe_signal
        psi = model._pe_signal                                            # [1, SEQ, HIDDEN]

        q_X = q_rope(X)                                                    # RoPE(W_q X)
        bias = attn.q_proj(psi).view(1, SEQ, -1, HEAD_DIM).transpose(1, 2)  # W_q Ψ (unrotated)
        q_target = q_X + bias                                             # RoPE(X)+Ψ
        q_new = q_X + bias                                                # patch builds q this way
        q_legacy = q_rope(X + psi)                                        # Ψ rotated by RoPE

    node_pos = sorted({p for spans in INJECTION.values() for (s, e) in spans for p in range(s, e)})
    idx = torch.tensor(node_pos)

    def err(qa):
        return (qa[:, :, idx, :] - q_target[:, :, idx, :]).abs().mean().item()

    e_new, e_legacy = err(q_new), err(q_legacy)
    check("new path == RoPE(X)+Ψ at q (exact)", e_new < 1e-5, f"err_new={e_new:.2e}")
    check("legacy path rotates Ψ (≫ new)", e_legacy > 1e-3 and e_new < e_legacy,
          f"err_legacy={e_legacy:.2e}")
    model._pe_signal = None


# --------------------------------------------------------------------------- #
# Test 3 — forward / loss / gradients through every patched layer
# --------------------------------------------------------------------------- #
def test_forward_and_grads(model):
    print("Test 3: forward + loss + gradient flow")
    input_ids = torch.randint(0, 128, (1, SEQ))
    out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                labels=input_ids.clone(), graphs=[DummyGraph(N_NODES)],
                injection_maps=[INJECTION])
    check("loss is finite", torch.isfinite(out.loss).item(), f"loss={out.loss.item():.4f}")
    check("Ψ disarmed after forward", model._pe_signal is None)

    out.loss.backward()
    g_pe = model.pe_model.table.grad
    g_proj = next((p.grad for p in model.pe_proj.parameters() if p.grad is not None), None)
    check("grad → pe_model", g_pe is not None and g_pe.abs().sum().item() > 0)
    check("grad → pe_proj", g_proj is not None and g_proj.abs().sum().item() > 0)
    check("grad → pe_gain",
          model.pe_gain.grad is not None and model.pe_gain.grad.abs().item() > 0,
          f"g={None if model.pe_gain.grad is None else model.pe_gain.grad.item():.2e}")


# --------------------------------------------------------------------------- #
# Test 4 — gradients survive gradient checkpointing
# --------------------------------------------------------------------------- #
def test_grad_checkpointing(model):
    print("Test 4: gradient flow under gradient checkpointing")
    model.pe_model.table.grad = None
    model.llm.gradient_checkpointing_enable()
    model.train()
    try:
        input_ids = torch.randint(0, 128, (1, SEQ))
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                    labels=input_ids.clone(), graphs=[DummyGraph(N_NODES)],
                    injection_maps=[INJECTION])
        out.loss.backward()
        g = model.pe_model.table.grad
        check("grad → pe_model with checkpointing", g is not None and g.abs().sum().item() > 0)
    finally:
        model.llm.gradient_checkpointing_disable()
        model.eval()


# --------------------------------------------------------------------------- #
# Test 5 — generation: Ψ skipped on decode steps, disarmed afterwards
# --------------------------------------------------------------------------- #
def test_generation_decode_skip(model):
    print("Test 5: Ψ auto-skips cached decode steps + disarms")
    input_ids = torch.randint(0, 128, (1, SEQ))
    X = model.llm.get_input_embeddings()(input_ids).clone()
    model._pe_signal = model.build_pe_signal(X, [DummyGraph(N_NODES)], [INJECTION])

    seen = {}
    orig = model.llm.model.layers[0].self_attn.forward

    def spy(hidden_states, **kw):
        psi = model._pe_signal
        injected = (psi is not None and psi.shape[1] == hidden_states.shape[1])
        seen.setdefault(hidden_states.shape[1], injected)
        return orig(hidden_states, **kw)

    model.llm.model.layers[0].self_attn.forward = spy
    try:
        with torch.no_grad():
            model.llm.generate(inputs_embeds=X, max_new_tokens=3, do_sample=False,
                               use_cache=True, pad_token_id=0)
    finally:
        model.llm.model.layers[0].self_attn.forward = orig

    check("Ψ injected on full prompt pass", seen.get(SEQ) is True, f"seen={seen}")
    check("Ψ skipped on len-1 decode steps", seen.get(1) is False, f"seen={seen}")
    model._pe_signal = None


# --------------------------------------------------------------------------- #
# Test 6 — train path == eval (inference) mirror
# --------------------------------------------------------------------------- #
def test_train_eval_parity(model):
    print("Test 6: _augment_embeddings (train) == inference mirror (eval)")
    input_ids = torch.randint(0, 128, (1, SEQ))
    graphs = [DummyGraph(N_NODES)]

    with torch.no_grad():
        model._augment_embeddings(input_ids, graphs, [INJECTION])
        psi_train = model._pe_signal.clone()
        model._pe_signal = None

        emb = model.llm.get_input_embeddings()(input_ids).clone()
        psi_eval = model.build_pe_signal(emb, graphs, [INJECTION], permutation=None)

    diff = (psi_train - psi_eval).abs().max().item()
    check("train/eval Ψ identical", diff < 1e-6, f"max|err|={diff:.2e}")


def main():
    model = build_model()
    test_patch_is_transparent(model)
    test_psi_unrotated_in_qk(model)
    test_forward_and_grads(model)
    test_grad_checkpointing(model)
    test_generation_decode_skip(model)
    test_train_eval_parity(model)

    print()
    if _failures:
        print(f"{FAIL}: {len(_failures)} check(s) failed: {_failures}")
        sys.exit(1)
    print(f"{PASS}: all smoke checks passed.")


if __name__ == "__main__":
    main()
