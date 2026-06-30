"""Tests for the LEARNABLE relative-PE attention mask (``learnable_graph_mask``).

``LearnableGraphMaskLLM`` adds a *learned* additive bias on the attention logits at
node-token positions::

    M[i,j] = alpha + (1-alpha)*sim(Psi_i, Psi_j)   adjacent / self-loop
    M[i,j] = finfo.min                              non-adjacent (hard block)

where ``Psi = pe_model(graph)`` and the N×N form is the outer product ``Psi Psi^T``.
These tests assert the mask math directly (hard block, scaling bound, no fully-masked
row, differentiability w.r.t. the PE), the dense-only layer routing (flag AND behavior),
and — critically — the REAL SDPA path with a padded batch (the eager/CPU path cannot
surface the boolean-mask bug). A tiny **stub PE** (Linear over per-node features) isolates
the mask math from the real GraphTransformer; it is graph-size independent and differentiable.

Runs on GPU when available (the SDPA path is the realistic one); falls back to CPU.
"""
import sys
sys.path.insert(0, "src")

import torch
from torch import nn
from torch_geometric.data import Data

from prism.models.gnn_llm import LearnableGraphMaskLLM


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NEG = torch.finfo(torch.float32).min


def _tiny_llm(hidden=32, attn="eager"):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=hidden, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, attn_implementation=attn)
    return LlamaForCausalLM(cfg).to(DEVICE)


class _StubPE(nn.Module):
    """Differentiable, graph-size-independent per-node embedding Psi ∈ [N, d].

    Linear(feat_dim -> d) over per-node features ``g.x`` [N, feat_dim]. Isolates the
    mask math from the real GraphTransformer (whose Psi the mask treats identically).
    """

    def __init__(self, feat_dim=4, d=8):
        super().__init__()
        self.lin = nn.Linear(feat_dim, d)

    def forward(self, g, permutation=None):
        return self.lin(g.x.float())


def _graph(n, edges, feat_dim=4, seed=1):
    """Directed edge list -> PyG Data with deterministic per-node features (on DEVICE)."""
    gen = torch.Generator().manual_seed(seed)
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
    g = Data(x=torch.randn(n, feat_dim, generator=gen).to(DEVICE),
             edge_index=ei.to(DEVICE), num_nodes=n)
    g.node_names = [f"node{i}" for i in range(n)]
    return g


def _wrap(alpha=0.7, layer_scope="all", psi_scale="cosine", k_hops=1,
          symmetrize=True, use_edges=True, d=8, hidden=32):
    torch.manual_seed(0)
    llm = _tiny_llm(hidden)
    pe = _StubPE(feat_dim=4, d=d)  # moved to the LLM device inside the constructor
    return LearnableGraphMaskLLM(
        llm, pe, alpha=alpha, layer_scope=layer_scope, psi_scale=psi_scale,
        k_hops=k_hops, symmetrize=symmetrize, use_edges=use_edges).eval()


def _mask(wrap, seq, graphs, imap):
    """Return the [seq, seq] bias on CPU (grad still flows back through .cpu())."""
    return wrap.build_structural_mask(seq, graphs, imap, DEVICE, dtype=torch.float32)[0, 0].cpu()


# ---------------------------------------------------------------------------
# Mask math
# ---------------------------------------------------------------------------

def test_hard_block_nonadjacent_and_learned_allowed():
    """node2 isolated; node0–node1 share an edge. Non-adjacent node pairs are finfo.min;
    adjacent pairs carry the learned value α+(1-α)·sim (finite, nonzero); node↔non-node
    and non-node rows stay 0."""
    wrap = _wrap()
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1)])  # only 0<->1
    bias = _mask(wrap, seq, [g], imap)

    assert bias[5, 1] == NEG and bias[5, 3] == NEG       # node2 -> node0/node1: no edge
    assert torch.isfinite(bias[3, 1]) and bias[3, 1] != NEG and bias[3, 1] != 0  # edge: learned
    assert torch.isclose(bias[1, 1], torch.tensor(1.0))  # cosine self-sim = 1
    assert torch.isclose(bias[5, 5], torch.tensor(1.0))
    assert bias[5, 0] == 0                                # node -> BOS (non-node): untouched
    assert torch.all(bias[7] == 0)                       # non-node row


def test_cosine_bound_and_self_similarity():
    """psi_scale='cosine' -> every allowed entry in [2α-1, 1]; diagonal (self-loop) = 1."""
    alpha = 0.7
    wrap = _wrap(alpha=alpha, psi_scale="cosine")
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1), (1, 2)])
    bias = _mask(wrap, seq, [g], imap)
    allowed = bias[bias > NEG / 2]
    allowed = allowed[allowed != 0]
    assert allowed.min() >= (2 * alpha - 1) - 1e-4 and allowed.max() <= 1.0 + 1e-4
    for p in (1, 3, 5):
        assert torch.isclose(bias[p, p], torch.tensor(1.0))


def test_mask_is_differentiable_wrt_pe():
    """Gradient flows through the allowed (edge) bias entries back to the PE params."""
    wrap = _wrap(psi_scale="cosine")
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1), (1, 2)])
    bias = _mask(wrap, seq, [g], imap)
    bias[bias > 0.1].sum().backward()   # .cpu() in _mask is differentiable
    w = wrap.pe_model.lin.weight
    assert w.grad is not None and w.grad.abs().sum() > 0


def test_inv_sqrt_d_differentiable_and_finite():
    """The inv_sqrt_d scaling path produces a finite mask and is differentiable."""
    wrap = _wrap(psi_scale="inv_sqrt_d")
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1), (1, 2)])
    bias = _mask(wrap, seq, [g], imap)
    assert torch.isfinite(bias[bias > NEG / 2]).all()
    bias[(bias > NEG / 2) & (bias != 0)].sum().backward()
    assert wrap.pe_model.lin.weight.grad.abs().sum() > 0


def test_no_node_row_is_fully_masked():
    """Every node-token row keeps its diagonal (self-loop = 1.0) and BOS (= 0)."""
    wrap = _wrap()
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[])  # no edges
    bias = _mask(wrap, seq, [g], imap)
    for p in (1, 3, 5):
        assert torch.isclose(bias[p, p], torch.tensor(1.0))  # self-loop
        assert bias[p, 0] == 0                                # BOS
    assert bias[3, 1] == NEG and bias[5, 3] == NEG


def test_symmetrize_and_k_hops_match_adjacency():
    """Adjacency rules (symmetrize, k_hops) gate which pairs are blocked vs learned."""
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1), (1, 2)])

    asym = _mask(_wrap(symmetrize=False), seq, [g], imap)
    assert asym[3, 1] == NEG  # no 1->0 edge without symmetrization

    k1 = _mask(_wrap(k_hops=1), seq, [g], imap)
    assert k1[5, 1] == NEG    # node2 -> node0 is 2 hops
    k2 = _mask(_wrap(k_hops=2), seq, [g], imap)
    assert torch.isfinite(k2[5, 1]) and k2[5, 1] != NEG  # reachable within 2 hops


def test_use_edges_false_blocks_all_cross_node():
    """Edgeless ablation (with inv_sqrt_d, since cosine+edgeless is guarded): only
    self-loops remain, so every cross-node pair is blocked."""
    abl = _wrap(use_edges=False, psi_scale="inv_sqrt_d")
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1), (1, 2)])  # real edges, but ablated away
    bias = _mask(abl, seq, [g], imap)
    assert bias[3, 1] == NEG and bias[5, 3] == NEG       # adjacent pairs now blocked
    assert torch.isfinite(bias[3, 3]) and bias[3, 3] != NEG  # self still allowed
    assert bias[3, 0] == 0                                # BOS still allowed


# ---------------------------------------------------------------------------
# Layer routing — flag AND behavior
# ---------------------------------------------------------------------------

def test_dense_only_routing_deactivates_sliding_layers():
    """layer_scope='dense' flags sliding-window layers inactive, full-attention layers active."""
    torch.manual_seed(0)
    llm = _tiny_llm()
    llm.model.layers[0].self_attn.is_sliding = True
    llm.model.layers[1].self_attn.is_sliding = False
    wrap = LearnableGraphMaskLLM(llm, _StubPE(), layer_scope="dense").eval()
    layers = wrap._decoder_layers()
    assert layers[0].self_attn._graph_mask_active is False
    assert layers[1].self_attn._graph_mask_active is True


def test_layer_scope_all_routes_every_layer():
    """layer_scope='all' activates every layer regardless of is_sliding."""
    torch.manual_seed(0)
    llm = _tiny_llm()
    llm.model.layers[0].self_attn.is_sliding = True
    llm.model.layers[1].self_attn.is_sliding = False
    wrap = LearnableGraphMaskLLM(llm, _StubPE(), layer_scope="all").eval()
    assert all(l.self_attn._graph_mask_active for l in wrap._decoder_layers())


def test_inactive_layer_output_unchanged_active_changes():
    """BEHAVIORAL: an inactive (sliding) layer's attention output is identical with vs
    without the graph mask, while an active (full) layer's output changes. (Catches a
    regression that would leak the learned bias into local/sliding layers.)"""
    torch.manual_seed(0)
    llm = _tiny_llm()
    llm.model.layers[0].self_attn.is_sliding = True   # inactive under dense scope
    llm.model.layers[1].self_attn.is_sliding = False  # active
    wrap = LearnableGraphMaskLLM(llm, _StubPE(), layer_scope="dense").eval()
    seq = 8
    input_ids = torch.randint(1, 64, (1, seq), device=DEVICE)
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1), (1, 2)])

    cap = {}
    def hook(key):
        def fn(mod, inp, out):
            cap[key] = (out[0] if isinstance(out, tuple) else out).detach().clone()
        return fn
    layers = wrap._decoder_layers()
    h0 = layers[0].self_attn.register_forward_hook(hook("l0"))
    h1 = layers[1].self_attn.register_forward_hook(hook("l1"))
    try:
        with torch.no_grad():
            wrap(input_ids=input_ids, graphs=None, injection_maps=None)
            l0_plain, l1_plain = cap["l0"].clone(), cap["l1"].clone()
            wrap(input_ids=input_ids, graphs=[g], injection_maps=imap)
            l0_graph, l1_graph = cap["l0"].clone(), cap["l1"].clone()
    finally:
        h0.remove(); h1.remove()
    assert torch.allclose(l0_plain, l0_graph, atol=1e-6)        # inactive: unchanged
    assert not torch.allclose(l1_plain, l1_graph, atol=1e-5)    # active: changed


# ---------------------------------------------------------------------------
# REAL SDPA path with padding — the bug the eager/CPU tests structurally miss
# ---------------------------------------------------------------------------

def test_sdpa_padded_batch_no_future_or_padding_leak():
    """Critical regression: under the REAL sdpa attention with a right-padded B=2 batch,
    the learned mask must not leak future or padding tokens. Each example's masked logits
    at real positions must match the SAME example run UNBATCHED (no padding). The stub PE
    is deterministic, so the only difference is padding/causal handling."""
    torch.manual_seed(0)
    llm = _tiny_llm(attn="sdpa")
    wrap = LearnableGraphMaskLLM(llm, _StubPE(), layer_scope="all").eval()

    full = torch.randint(1, 64, (2, 8), device=DEVICE)
    lens = [8, 6]
    attn = torch.zeros(2, 8, dtype=torch.long, device=DEVICE)
    for i, L in enumerate(lens):
        attn[i, :L] = 1
    gA, gB = _graph(3, edges=[(0, 1), (1, 2)], seed=1), _graph(3, edges=[(0, 2)], seed=2)
    imA = {0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}
    imB = {0: [(1, 2)], 1: [(3, 4)], 2: [(4, 5)]}

    with torch.no_grad():
        batched = wrap(input_ids=full, attention_mask=attn,
                       graphs=[gA, gB], injection_maps=[imA, imB]).logits
        refA = wrap(input_ids=full[0:1, :8], attention_mask=attn[0:1, :8],
                    graphs=[gA], injection_maps=[imA]).logits
        refB = wrap(input_ids=full[1:2, :6], attention_mask=attn[1:2, :6],
                    graphs=[gB], injection_maps=[imB]).logits

    assert torch.isfinite(batched).all()
    dA = (batched[0, :8] - refA[0]).abs().max().item()
    dB = (batched[1, :6] - refB[0]).abs().max().item()
    assert dA < 1e-3, f"example A real-token logits diverged from unbatched ref: {dA}"
    assert dB < 1e-3, f"example B real-token logits diverged from unbatched ref: {dB}"


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------

def test_alpha_one_rejected():
    torch.manual_seed(0)
    try:
        LearnableGraphMaskLLM(_tiny_llm(), _StubPE(), alpha=1.0)
        assert False, "alpha=1.0 should raise"
    except ValueError:
        pass


def test_cosine_edgeless_rejected():
    """cosine + use_edges=False is a silent zero-gradient degeneracy -> must raise."""
    torch.manual_seed(0)
    try:
        LearnableGraphMaskLLM(_tiny_llm(), _StubPE(), psi_scale="cosine", use_edges=False)
        assert False, "cosine + use_edges=False should raise"
    except ValueError:
        pass


def test_invalid_scope_and_scale_rejected():
    torch.manual_seed(0)
    for bad in (dict(layer_scope="bogus"), dict(psi_scale="bogus")):
        try:
            LearnableGraphMaskLLM(_tiny_llm(), _StubPE(), **bad)
            assert False
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_forward_runs_differs_and_grad_flows():
    """Forward with graphs is finite, differs from plain causal, propagates grad to the
    PE, and disarms the bias afterward."""
    torch.manual_seed(0)
    wrap = _wrap(layer_scope="all")
    seq = 8
    input_ids = torch.randint(0, 64, (1, seq), device=DEVICE)
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1), (1, 2)])

    wrap.train()
    out = wrap(input_ids=input_ids, graphs=[g], injection_maps=imap, labels=input_ids)
    assert torch.isfinite(out.logits).all()
    out.loss.backward()
    assert wrap.pe_model.lin.weight.grad is not None
    assert wrap.pe_model.lin.weight.grad.abs().sum() > 0

    wrap.eval()
    with torch.no_grad():
        masked = wrap(input_ids=input_ids, graphs=[g], injection_maps=imap).logits
        plain = wrap(input_ids=input_ids, graphs=None, injection_maps=None).logits
    assert torch.isfinite(masked).all()
    assert not torch.allclose(masked, plain, atol=1e-5)
    assert wrap._struct_bias is None


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
