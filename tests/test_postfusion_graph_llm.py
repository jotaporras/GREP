"""Tests for the EXPERIMENTAL post-fusion (output) graph attention (``postfusion_graph_llm``).

``PostFusionGraphLLM`` runs the base LLM unchanged, then — via a forward pre-hook on the
lm_head — rewrites the head's input from ``Y`` to ``Y + tanh(gate)*CrossAttn(Q=Y, K,V=Psi)``
where ``Psi = pe_model(graph)``. These tests assert the properties that make the idea sound:

* cold start (gate=0) is EXACTLY the base LLM (no silent perturbation at init);
* opening the gate changes the logits, and the change is differentiable back to BOTH the
  cross-attention (``fusion``) and the graph encoder (``pe_model``);
* the fusion fires at query length 1 — the decode step — so it is not a train-only effect
  that vanishes at generation (the whole point of using cross-attention over a bare additive
  placement); a ``generate`` smoke confirms the persistent hook path;
* under the REAL sdpa path with a right-padded B=2 batch, per-example logits match the same
  example run UNBATCHED (padding-node masking in the cross-attention is correct);
* PEFT does NOT adapt the fusion projections (the ``to_*`` names dodge the LoRA target
  suffixes) — a silent mis-training trap this guards against;
* the GradientDebugCallback tolerates the pe_model-only (no pe_proj/pe_gain) shape.

Runs on GPU when available (the sdpa path is the realistic one); falls back to CPU.
"""
import sys
sys.path.insert(0, "src")

import torch
from torch import nn
from torch_geometric.data import Data

from prism.models.postfusion_graph_llm import PostFusionGraphLLM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tiny_llm(hidden=32, attn="eager"):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=hidden, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, attn_implementation=attn)
    return LlamaForCausalLM(cfg).to(DEVICE)


class _StubPE(nn.Module):
    """Differentiable, graph-size-independent per-node embedding Psi ∈ [N, P].

    Linear(feat_dim -> P) over per-node features ``g.x`` [N, feat_dim]. Isolates the fusion
    from the real GraphTransformer (whose Psi the fusion treats identically)."""

    def __init__(self, feat_dim=4, p=16):
        super().__init__()
        self.lin = nn.Linear(feat_dim, p)

    def forward(self, g, permutation=None):
        return self.lin(g.x.float())


def _graph(n, edges, feat_dim=4, seed=1):
    gen = torch.Generator().manual_seed(seed)
    ei = (torch.tensor(edges, dtype=torch.long).t().contiguous()
          if edges else torch.zeros(2, 0, dtype=torch.long))
    g = Data(x=torch.randn(n, feat_dim, generator=gen).to(DEVICE),
             edge_index=ei.to(DEVICE), num_nodes=n)
    g.node_names = [f"node{i}" for i in range(n)]
    return g


def _wrap(gate_init=0.0, num_heads=4, hidden=32, p=16, attn="eager"):
    torch.manual_seed(0)
    llm = _tiny_llm(hidden, attn=attn)
    pe = _StubPE(feat_dim=4, p=p)
    return PostFusionGraphLLM(llm, pe, pe_dim=p, num_heads=num_heads,
                              gate_init=gate_init).eval()


# ---------------------------------------------------------------------------
# Cold start + gate behavior
# ---------------------------------------------------------------------------

def test_coldstart_gate_zero_is_identity():
    """gate_init=0 ⇒ tanh(0)=0 ⇒ the fused hidden == Y exactly ⇒ logits identical to the
    base LLM (forward with graphs must equal forward without)."""
    wrap = _wrap(gate_init=0.0)
    ids = torch.randint(1, 64, (1, 8), device=DEVICE)
    g = _graph(3, edges=[(0, 1), (1, 2)])
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    with torch.no_grad():
        fused = wrap(input_ids=ids, graphs=[g], injection_maps=imap).logits
        plain = wrap(input_ids=ids, graphs=None, injection_maps=None).logits
    assert torch.isfinite(fused).all()
    assert torch.allclose(fused, plain, atol=1e-6)   # cold start == base LLM
    assert wrap._graph_ctx is None                    # disarmed after forward


def test_open_gate_changes_logits():
    """A nonzero gate makes the graph fusion perturb the logits (vs the base LLM)."""
    wrap = _wrap(gate_init=0.0)
    wrap.gate.data.fill_(1.0)                          # open the gate
    ids = torch.randint(1, 64, (1, 8), device=DEVICE)
    g = _graph(3, edges=[(0, 1), (1, 2)])
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    with torch.no_grad():
        fused = wrap(input_ids=ids, graphs=[g], injection_maps=imap).logits
        plain = wrap(input_ids=ids, graphs=None, injection_maps=None).logits
    assert torch.isfinite(fused).all()
    assert not torch.allclose(fused, plain, atol=1e-5)


def test_differentiable_to_fusion_and_pe():
    """Loss grad reaches BOTH the cross-attention projections and the graph encoder."""
    wrap = _wrap(gate_init=0.0)
    wrap.gate.data.fill_(0.5)
    wrap.train()
    ids = torch.randint(1, 64, (1, 8), device=DEVICE)
    g = _graph(3, edges=[(0, 1), (1, 2)])
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    out = wrap(input_ids=ids, graphs=[g], injection_maps=imap, labels=ids)
    assert torch.isfinite(out.logits).all()
    out.loss.backward()
    # cross-attention (queries come from Y, keys/values from Psi)
    for name in ("to_q", "to_k", "to_v", "proj_out"):
        w = getattr(wrap.fusion, name).weight
        assert w.grad is not None and w.grad.abs().sum() > 0, f"no grad on fusion.{name}"
    # graph encoder + gate
    assert wrap.pe_model.lin.weight.grad is not None and wrap.pe_model.lin.weight.grad.abs().sum() > 0
    assert wrap.gate.grad is not None and wrap.gate.grad.abs().item() > 0


# ---------------------------------------------------------------------------
# Generation: the fusion must fire at query length 1 (the decode step)
# ---------------------------------------------------------------------------

def test_fusion_fires_at_query_len_one():
    """The lm_head pre-hook applies the cross-attention even when the head sees a single
    token [B,1,H] (cached decode) — otherwise post-fusion would be a train-only no-op."""
    wrap = _wrap(gate_init=0.0)
    wrap.gate.data.fill_(1.0)
    g = _graph(3, edges=[(0, 1), (1, 2)])
    hidden = torch.randn(1, 1, 32, device=DEVICE)          # a single decode-step hidden state
    with torch.no_grad():
        wrap._graph_ctx = wrap.build_graph_context([g], DEVICE)
        try:
            armed = wrap.llm.lm_head(hidden)               # hook fires (query len 1)
        finally:
            wrap._graph_ctx = None
        bare = wrap.llm.lm_head(hidden)                    # hook no-op (disarmed)
    assert armed.shape == (1, 1, wrap.llm.config.vocab_size)
    assert torch.isfinite(armed).all()
    assert not torch.allclose(armed, bare, atol=1e-5)      # graph context changed the logits


def test_generate_smoke():
    """model.llm.generate runs with the graph context armed (persistent hook path)."""
    wrap = _wrap(gate_init=0.0)
    wrap.gate.data.fill_(0.3)
    g = _graph(3, edges=[(0, 1), (1, 2)])
    ids = torch.randint(1, 64, (1, 6), device=DEVICE)
    with torch.no_grad():
        wrap._graph_ctx = wrap.build_graph_context([g], DEVICE)
        try:
            out = wrap.llm.generate(input_ids=ids, max_new_tokens=4, do_sample=False,
                                    use_cache=True, pad_token_id=0)
        finally:
            wrap._graph_ctx = None
    assert out.shape[1] == 6 + 4


# ---------------------------------------------------------------------------
# REAL sdpa path with padding — per-example logits vs unbatched
# ---------------------------------------------------------------------------

def test_sdpa_padded_batch_matches_unbatched():
    """Under the real sdpa attention with a right-padded B=2 batch (different graph sizes),
    each example's real-token logits match the SAME example run unbatched. Confirms the
    cross-attention masks padding nodes and the LLM's own padding handling is intact."""
    wrap = _wrap(gate_init=0.0, attn="sdpa")
    wrap.gate.data.fill_(1.0)                              # open so the fusion is exercised
    full = torch.randint(1, 64, (2, 8), device=DEVICE)
    lens = [8, 6]
    attn = torch.zeros(2, 8, dtype=torch.long, device=DEVICE)
    for i, L in enumerate(lens):
        attn[i, :L] = 1
    gA = _graph(4, edges=[(0, 1), (1, 2), (2, 3)], seed=1)  # 4 nodes
    gB = _graph(2, edges=[(0, 1)], seed=2)                  # 2 nodes -> padded to 4 in the batch
    imA = [{0: [(1, 2)]}]
    with torch.no_grad():
        batched = wrap(input_ids=full, attention_mask=attn,
                       graphs=[gA, gB], injection_maps=[{}, {}]).logits
        refA = wrap(input_ids=full[0:1, :8], attention_mask=attn[0:1, :8],
                    graphs=[gA], injection_maps=[{}]).logits
        refB = wrap(input_ids=full[1:2, :6], attention_mask=attn[1:2, :6],
                    graphs=[gB], injection_maps=[{}]).logits
    assert torch.isfinite(batched).all()
    dA = (batched[0, :8] - refA[0]).abs().max().item()
    dB = (batched[1, :6] - refB[0]).abs().max().item()
    assert dA < 1e-3, f"example A diverged from unbatched ref: {dA}"
    assert dB < 1e-3, f"example B (padded nodes) diverged from unbatched ref: {dB}"


# ---------------------------------------------------------------------------
# PEFT name-collision guard — the fusion must NOT be LoRA-adapted
# ---------------------------------------------------------------------------

def test_peft_does_not_adapt_fusion_projections():
    """PEFT matches target modules by suffix (``key.endswith('.'+target)``). The fusion
    projections are named ``to_q``/``to_k``/``to_v``/``proj_out`` precisely so they dodge the
    ``q_proj``/``k_proj``/``v_proj``/``o_proj`` targets — otherwise they'd get a LoRA adapter
    AND be frozen (silent mis-training). Assert no LoRA under ``fusion``; LLM attn IS adapted."""
    from peft import LoraConfig, get_peft_model
    wrap = _wrap(gate_init=0.0)
    lora = LoraConfig(
        r=4, lora_alpha=4, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM")
    peft_model = get_peft_model(wrap, lora)
    names = [n for n, _ in peft_model.named_modules()]
    fusion_lora = [n for n in names if "fusion" in n and "lora_" in n]
    assert not fusion_lora, f"fusion projections were LoRA-adapted (name collision!): {fusion_lora}"
    # sanity: PEFT actually ran (the LLM's own attention projections got adapters)
    assert any("self_attn.q_proj.lora_A" in n for n in names)


# ---------------------------------------------------------------------------
# Diagnostics callback
# ---------------------------------------------------------------------------

def test_gradient_debug_callback_handles_postfusion():
    """GradientDebugCallback must handle the pe_model-only shape (no pe_proj/pe_gain) without
    crashing, and capture the GT grad."""
    from prism.eval.callbacks import GradientDebugCallback
    from prism.models import gt as gt_module
    torch.manual_seed(0)
    llm = _tiny_llm()
    gt = gt_module.GraphTransformer(num_layers=2, pe_hidden_channels=16, pe_num_layers=2,
                                    d_model=16, heads=4, num_samples=8, dropout=0.0,
                                    k_pe=2, k_gt=2, node_feature_dim=None)
    model = PostFusionGraphLLM(llm, gt, pe_dim=16, num_heads=4, gate_init=0.3)

    cb = GradientDebugCallback()
    assert cb._supported(cb._unwrap_peft(model))
    cb.on_train_begin(None, None, None, model=model)

    g = _graph(3, edges=[(0, 1), (1, 2)])
    ids = torch.randint(1, 64, (1, 8), device=DEVICE)
    out = model(input_ids=ids, graphs=[g], injection_maps=[{}], labels=ids)
    out.loss.backward()
    cb._capture_grad_norms(model)
    assert cb._captured_grad_norms.get("gnn", 0.0) > 0

    class _State:
        log_history = []
        global_step = 1
    cb.on_log(None, _State(), None, model=model)   # must not raise


# ---------------------------------------------------------------------------
# Construction guard
# ---------------------------------------------------------------------------

def test_num_heads_must_divide_hidden():
    torch.manual_seed(0)
    try:
        PostFusionGraphLLM(_tiny_llm(hidden=32), _StubPE(p=16), pe_dim=16, num_heads=5)
        assert False, "num_heads not dividing hidden_size should raise"
    except ValueError:
        pass


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
