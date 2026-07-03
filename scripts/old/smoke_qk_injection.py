"""Smoke test for InjectedCompositeGraphLLM — GT code injected into q/k/v in place of
RoPE, written in eval_unification's patched-attention style.

Verifies, on a tiny random Llama + tiny composite graph (CPU-only, no real weights):
  1. the attention patch is faithful: with the signal off, the (RoPE-disabled) LLM is
     byte-identical to the un-patched forward,
  2. RoPE is OFF (identity rotary) — the injected code is the sole position signal,
  3. dedicated W_q/W_k/W_v carry the code (q→H·Dh, k/v→Hkv·Dh),
  4. inputs_embeds is the gated GT blend M7(X, Y_tok) (the Layer-0 injection),
  5. injecting Y_tok into q/k/v at every layer changes the logits; value toggles,
  6. gradients flow to W_q/W_k/W_v and the Graph Transformer,
  7. generation runs (injection skips cached decode steps),
  8. fixed-seed determinism.
"""
import torch
from torch_geometric.data import Data
from transformers import LlamaConfig, LlamaForCausalLM

from prism.models.gt import GraphTransformer
from prism.models.composite_graph_llm import InjectedCompositeGraphLLM
from prism.models.llama import disable_rope, _IdentityRotaryEmbedding

torch.manual_seed(0)
PASS, FAIL = [], []
def check(name, cond, info=""):
    (PASS if cond else FAIL).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {info}" if info else ""))

D = 32          # hidden / d_model
C = 8           # prompt length (token-cycle length)
N_LAYERS = 3
N_SCENE = 4

def make_llm():
    cfg = LlamaConfig(vocab_size=64, hidden_size=D, intermediate_size=64,
                      num_hidden_layers=N_LAYERS, num_attention_heads=2,
                      num_key_value_heads=1, max_position_embeddings=128)
    return LlamaForCausalLM(cfg)

def make_gt():
    return GraphTransformer(num_layers=2, pe_hidden_channels=16, pe_num_layers=2,
                            d_model=D, heads=2, num_samples=4, k_pe=2, k_gt=2,
                            eps=1e-6, spectral_norm_linears=False,
                            pe_readout="second_moment")

def make_model(inject_v=True, injection_mode="interpolate", gate_init=0.5):
    torch.manual_seed(0)
    return InjectedCompositeGraphLLM(
        make_llm(), make_gt(), d_model=D, inject_v=inject_v,
        injection_mode=injection_mode, gate_init=gate_init)

def make_inputs(B=1):
    scene = Data(x=torch.zeros(N_SCENE, 1),
                 edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]))
    scene.edge_weight = torch.ones(3)
    scene.num_nodes = N_SCENE
    graphs = [scene for _ in range(B)]
    inj = [{0: [(1, 3)], 2: [(4, 6)]} for _ in range(B)]
    input_ids = torch.randint(0, 64, (B, C))
    return input_ids, graphs, inj

# ---------- 1. faithful patch: signal off == un-patched (RoPE-off) forward ----------
torch.manual_seed(0)
llm_a = make_llm().eval()
disable_rope(llm_a)                       # match the model's RoPE-off content path
ids0 = torch.randint(0, 64, (1, C))
X0 = llm_a.get_input_embeddings()(ids0)
with torch.no_grad():
    logits_pre = llm_a(inputs_embeds=X0).logits
model = make_model().eval()               # seed-0 make_llm() → same weights, RoPE off
model._pe_signal = None
with torch.no_grad():
    logits_post = model.llm(inputs_embeds=X0).logits
check("attention patch faithful: signal-off == un-patched forward",
      torch.allclose(logits_pre, logits_post, atol=1e-5),
      f"max|Δ|={(logits_pre-logits_post).abs().max():.2e}")

# ---------- 2. RoPE OFF (identity rotary) ----------
check("RoPE is OFF by default (identity rotary)",
      isinstance(model.llm.model.rotary_emb, _IdentityRotaryEmbedding))

# ---------- 3. dedicated W_q/W_k/W_v ----------
check("dedicated W_q : d_model -> num_heads*head_dim",
      model.pe_q_proj.weight.shape == (2 * 16, D))
check("dedicated W_k : d_model -> num_kv_heads*head_dim",
      model.pe_k_proj.weight.shape == (1 * 16, D))
check("dedicated W_v : d_model -> num_kv_heads*head_dim",
      model.pe_v_proj.weight.shape == (1 * 16, D))

# ---------- 4. inputs_embeds is the gated GT blend M7(X, Y_tok) ----------
ids, graphs, inj = make_inputs(B=1)
X = model.llm.get_input_embeddings()(ids)
emb, sig = model._build_signal(ids, graphs, inj)
check("signal is Y_tok ([B,C,D]) and inputs_embeds is the blend ([B,C,D])",
      tuple(sig.shape) == (1, C, D) and tuple(emb.shape) == (1, C, D))
check("inputs_embeds == M7 interpolate blend 0.5*X + 0.5*Y_tok",
      torch.allclose(emb, 0.5 * X + 0.5 * sig, atol=1e-4))
check("blend differs from plain X and from raw Y_tok",
      not torch.allclose(emb, X) and not torch.allclose(emb, sig))

# ---------- 5. injection at every layer changes the logits; value toggles ----------
with torch.no_grad():
    out_on = model(input_ids=ids, graphs=graphs, injection_maps=inj).logits
    saved = {n: getattr(model, n).weight.clone() for n in ("pe_q_proj", "pe_k_proj", "pe_v_proj")}
    for n in saved:
        getattr(model, n).weight.zero_()
    out_off = model(input_ids=ids, graphs=graphs, injection_maps=inj).logits
    for n, w in saved.items():
        getattr(model, n).weight.copy_(w)
check("per-layer q/k/v injection changes the logits",
      not torch.allclose(out_on, out_off, atol=1e-5),
      f"max|Δ|={(out_on-out_off).abs().max():.3e}")
check("signal disarmed after forward", model._pe_signal is None)
with torch.no_grad():
    model._pe_inject_value = True
    out_v = model(input_ids=ids, graphs=graphs, injection_maps=inj).logits
    model._pe_inject_value = False
    out_nov = model(input_ids=ids, graphs=graphs, injection_maps=inj).logits
    model._pe_inject_value = True
check("value injection has an effect (toggles logits)",
      not torch.allclose(out_v, out_nov, atol=1e-5),
      f"max|Δ|={(out_v-out_nov).abs().max():.3e}")
check("inject_v=False has no pe_v_proj", getattr(make_model(inject_v=False), "pe_v_proj", None) is None)

# ---------- 6. gradient flow ----------
model.train()
ids2, graphs2, inj2 = make_inputs(B=2)
out = model(input_ids=ids2, graphs=graphs2, injection_maps=inj2, labels=ids2)
out.loss.backward()
def gnorm(p):
    return 0.0 if p.grad is None else float(p.grad.abs().sum())
for n in ("pe_q_proj", "pe_k_proj", "pe_v_proj"):
    check(f"{n} receives nonzero grad", gnorm(getattr(model, n).weight) > 0,
          f"{gnorm(getattr(model, n).weight):.3e}")
gt_grad = sum(1 for p in model.gt_model.parameters()
              if p.grad is not None and p.grad.abs().sum() > 0)
check("Graph Transformer receives grad through the injection", gt_grad > 0, f"{gt_grad} tensors")
check("no NaN gradients",
      not any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None))

# ---------- 7. generation ----------
gen = make_model().eval()
ids1, g1, j1 = make_inputs(B=1)
emb1 = gen.prepare_generation(ids1, g1, j1)
check("prepare_generation arms signal and returns the blend [1,C,D]",
      gen._pe_signal is not None and tuple(emb1.shape) == (1, C, D))
with torch.no_grad():
    # min_new_tokens=4 so the tiny random model can't stop early on EOS — we are
    # checking the decode loop runs (injection skips cached steps), not the content.
    o = gen.llm.generate(inputs_embeds=emb1, max_new_tokens=4, min_new_tokens=4,
                         do_sample=False, use_cache=True,
                         pad_token_id=gen.config.eos_token_id or 0)
gen._pe_signal = None
check("generation runs the decode loop (4 new tokens)", o.shape[1] == 4, f"shape={tuple(o.shape)}")

# ---------- 8. determinism ----------
det = make_model().eval()
det.gt_model.pe_model.fixed_seed_mode = True
with torch.no_grad():
    a = det(input_ids=ids1, graphs=g1, injection_maps=j1).logits
    b = det(input_ids=ids1, graphs=g1, injection_maps=j1).logits
check("fixed-seed forward is deterministic", torch.allclose(a, b, atol=1e-5))

print("\n================  SUMMARY  ================")
print(f"PASSED {len(PASS)} / {len(PASS) + len(FAIL)}")
if FAIL:
    print("FAILED:", FAIL)
    raise SystemExit(1)
