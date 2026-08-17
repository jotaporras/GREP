"""Tests for the LEARNABLE relative-PE attention mask (``learnable_graph_mask``).

``LearnableGraphMaskLLM`` adds a *learned* additive bias on the attention logits at
node-token positions::

    M[i,j] = log(alpha + (1-alpha)*sim(Psi_i, Psi_j))   adjacent / self-loop
    M[i,j] = finfo.min                                   non-adjacent (hard block)

where ``Psi = pe_model(graph)`` and the N×N form is the outer product ``Psi Psi^T``.

The ``log`` makes the bias a MULTIPLICATIVE (Hadamard) gate on the post-softmax attention
weights, which is the documented design (see ``LearnableGraphMaskLLM`` and
``MaskDecodeInjector``). Consequences the assertions below rely on: the allowed range is
``[log(2*alpha-1), 0]``, not ``[2*alpha-1, 1]``; and a cosine self-loop scores
``log(1) = 0``, i.e. numerically indistinguishable from an untouched non-node pair — so
"is this pair blocked?" must be tested as ``> NEG/2``, never as ``!= 0``.

These tests assert the mask math directly (hard block, scaling bound, no fully-masked
row, differentiability w.r.t. the PE), the dense-only layer routing (flag AND behavior),
the ``--permutation-seed`` acceptance check (the permutation must reach BOTH Psi and the
adjacency, and must not be inert), and — critically — the REAL SDPA path with a padded
batch (the eager/CPU path cannot surface the boolean-mask bug). A tiny **stub PE** (Linear
over per-node features) isolates the mask math from the real GraphTransformer; the
permutation tests instead use the real GT (probes pinned) because only a topology-dependent
Psi can observe a relabelling at all.

Runs on GPU when available (the SDPA path is the realistic one); falls back to CPU.
"""
import math
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
    adjacent pairs carry the learned value log(α+(1-α)·sim) (finite, nonzero); node↔non-node
    and non-node rows stay 0."""
    wrap = _wrap()
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1)])  # only 0<->1
    bias = _mask(wrap, seq, [g], imap)

    assert bias[5, 1] == NEG and bias[5, 3] == NEG       # node2 -> node0/node1: no edge
    assert torch.isfinite(bias[3, 1]) and bias[3, 1] != NEG and bias[3, 1] != 0  # edge: learned
    # cosine self-sim = 1 -> gate = 1 -> log-gate = 0 (an unbiased, fully-visible pair)
    assert torch.isclose(bias[1, 1], torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(bias[5, 5], torch.tensor(0.0), atol=1e-6)
    assert bias[5, 0] == 0                                # node -> BOS (non-node): untouched
    assert torch.all(bias[7] == 0)                       # non-node row


def test_cosine_bound_and_self_similarity():
    """psi_scale='cosine' -> every allowed entry in [log(2α-1), 0]; diagonal (self-loop) = 0.

    sim ∈ [-1, 1] ⇒ gate = α+(1-α)·sim ∈ [2α-1, 1] ⇒ log-gate ∈ [log(2α-1), 0].
    """
    alpha = 0.7
    wrap = _wrap(alpha=alpha, psi_scale="cosine")
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1), (1, 2)])
    bias = _mask(wrap, seq, [g], imap)
    # Node-token positions only: a non-node entry is 0, and so is a self-loop log-gate,
    # so the two cannot be separated by value — select by position instead.
    node_pos = torch.tensor([1, 3, 5])
    allowed = bias[node_pos][:, node_pos]
    allowed = allowed[allowed > NEG / 2]
    lo = math.log(2 * alpha - 1)
    # Path 0-1-2 at k_hops=1: 3 self-loops + 2 edges x2 directions allowed; (0,2)/(2,0) blocked.
    assert allowed.numel() == 7, allowed.numel()
    assert allowed.min() >= lo - 1e-4 and allowed.max() <= 0.0 + 1e-4
    for p in (1, 3, 5):
        assert torch.isclose(bias[p, p], torch.tensor(0.0), atol=1e-6)


def test_mask_is_differentiable_wrt_pe():
    """Gradient flows through the allowed (edge) bias entries back to the PE params."""
    wrap = _wrap(psi_scale="cosine")
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1), (1, 2)])
    bias = _mask(wrap, seq, [g], imap)
    # Allowed log-gate entries are <= 0, so a positive threshold would select the EMPTY
    # set and back-propagate an all-zero gradient without failing. Select the same way the
    # inv_sqrt_d sibling test does: finite (not hard-blocked) and not an untouched pair.
    bias[(bias > NEG / 2) & (bias != 0)].sum().backward()   # .cpu() in _mask is differentiable
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
    """Every node-token row keeps its diagonal (self-loop log-gate = 0.0) and BOS (= 0)."""
    wrap = _wrap()
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[])  # no edges
    bias = _mask(wrap, seq, [g], imap)
    for p in (1, 3, 5):
        assert bias[p, p] > NEG / 2                           # self-loop is never blocked
        assert torch.isclose(bias[p, p], torch.tensor(0.0), atol=1e-6)
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


# ---------------------------------------------------------------------------
# Identity-RoPE on graph tokens (model.disable_graph_token_rope)
# ---------------------------------------------------------------------------

def _capture_position_ids(wrap):
    """Record the position_ids the wrapper hands to the inner LLM (None if it passes none)."""
    seen = {}
    inner = wrap.llm.forward

    def spy(*args, **kwargs):
        seen["position_ids"] = kwargs.get("position_ids")
        return inner(*args, **kwargs)

    wrap.llm.forward = spy
    return seen


def test_identity_rope_zeroes_injected_spans_only():
    """disable_graph_token_rope=True ⇒ the injected spans get position 0, everything
    else keeps its arange index. Off (the default) ⇒ no position_ids at all, so the LLM
    numbers the sequence itself."""
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 5)]}]
    g = _graph(2, edges=[(0, 1)])
    input_ids = torch.randint(0, 64, (1, seq), device=DEVICE)

    on = _wrap()
    on._disable_graph_token_rope = True
    seen_on = _capture_position_ids(on)
    off = _wrap()
    seen_off = _capture_position_ids(off)
    with torch.no_grad():
        on(input_ids=input_ids, graphs=[g], injection_maps=imap)
        off(input_ids=input_ids, graphs=[g], injection_maps=imap)

    assert seen_off["position_ids"] is None
    pos = seen_on["position_ids"]
    assert pos.shape == (1, seq)
    assert pos[0].tolist() == [0, 0, 2, 0, 0, 5, 6, 7]


def test_identity_rope_changes_logits_and_respects_caller_position_ids():
    """The flag is a real architectural arm (logits move), and an explicit position_ids
    from the caller wins — the wrapper must not overwrite it."""
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 5)]}]
    g = _graph(2, edges=[(0, 1)])
    input_ids = torch.randint(0, 64, (1, seq), device=DEVICE)

    on, off = _wrap(), _wrap()
    on.load_state_dict(off.state_dict())      # same weights: only the flag differs
    on._disable_graph_token_rope = True
    with torch.no_grad():
        a = on(input_ids=input_ids, graphs=[g], injection_maps=imap).logits
        b = off(input_ids=input_ids, graphs=[g], injection_maps=imap).logits
    assert not torch.allclose(a, b, atol=1e-5)

    explicit = torch.arange(seq, device=DEVICE).unsqueeze(0)
    seen = _capture_position_ids(on)
    with torch.no_grad():
        on(input_ids=input_ids, graphs=[g], injection_maps=imap, position_ids=explicit)
    assert torch.equal(seen["position_ids"], explicit)


def test_identity_rope_constructor_flag_reaches_the_attribute():
    """The constructor arg is what architectures/loaders pass; it must set the attribute
    inference.py duck-types on."""
    llm = _tiny_llm()
    wrap = LearnableGraphMaskLLM(llm, _StubPE(), disable_graph_token_rope=True)
    assert wrap._disable_graph_token_rope is True
    assert LearnableGraphMaskLLM(_tiny_llm(), _StubPE())._disable_graph_token_rope is False


def test_gradient_debug_callback_handles_learnable_mask():
    """Regression for the on_train_begin AttributeError: GradientDebugCallback must not
    assume GraphAugmented's pe_proj/pe_gain. LearnableGraphMaskLLM has a GT `pe_model` but
    no pe_proj, so the callback must install hooks, capture grad norms, and log without crashing."""
    from prism.eval.callbacks import GradientDebugCallback
    from prism.models import gt as gt_module
    torch.manual_seed(0)
    llm = _tiny_llm()
    gt = gt_module.GraphTransformer(num_layers=2, pe_hidden_channels=16, pe_num_layers=2,
                                    d_model=24, heads=4, num_samples=8, dropout=0.0,
                                    k_pe=2, k_gt=2, node_feature_dim=None)
    model = LearnableGraphMaskLLM(llm, gt, layer_scope="all")

    cb = GradientDebugCallback()
    assert cb._supported(cb._unwrap_peft(model))
    cb.on_train_begin(None, None, None, model=model)   # was the crash site

    g = _graph(3, edges=[(0, 1), (1, 2)])
    ids = torch.randint(1, 64, (1, 8), device=DEVICE)
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    out = model(input_ids=ids, graphs=[g], injection_maps=imap, labels=ids)
    out.loss.backward()
    cb._capture_grad_norms(model)
    assert cb._captured_grad_norms.get("gnn", 0.0) > 0    # GT (pe_model) grad captured

    class _State:
        log_history = []
        global_step = 1
    cb.debug_metrics(model, _State())                     # must not raise (no pe_gain access)


# ---------------------------------------------------------------------------
# --permutation-seed (eval-time node relabelling) — ACCEPTANCE TEST
#
# This is the check that would have caught the silent no-op: before the fix
# `build_structural_mask` took no `permutation` argument at all, so the whole
# transferability sweep reported permuted numbers for an unpermuted graph.
# ---------------------------------------------------------------------------

def _real_gt_wrap(d=16):
    """A mask model whose Ψ is the REAL GraphTransformer, with the R-PEARL probes pinned
    (fixed_seed_mode) so Ψ is a deterministic function of the TOPOLOGY — the only way a
    permutation can be observed at all."""
    torch.manual_seed(0)
    from prism.models import gt as gt_module
    gt = gt_module.GraphTransformer(
        num_layers=2, pe_hidden_channels=8, pe_num_layers=2, d_model=d, heads=2,
        num_samples=8, dropout=0.0, k_pe=2, k_gt=1, node_feature_dim=None,
        fixed_seed_mode=True, fixed_seed_value=11)
    return LearnableGraphMaskLLM(_tiny_llm(), gt, layer_scope="all").eval()


def test_permutation_moves_both_mask_factors():
    """The mask is A ⊙ log-gate(ΨΨᵀ). BOTH factors must be computed on the relabelled
    graph: permuting one and not the other is not a permutation of anything."""
    from prism.models.utils import Permutation
    wrap = _real_gt_wrap()
    g = _graph(6, edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    perm = Permutation(seed=3)
    with torch.no_grad():
        psi_0, psi_p = wrap.pe_model(g), wrap.pe_model(g, permutation=perm)
    assert not torch.allclose(psi_0, psi_p), "permutation did not reach the Ψ producer"
    a_0 = wrap._node_adjacency(g, DEVICE)
    a_p = wrap._node_adjacency(g, DEVICE, permutation=perm)
    assert not torch.equal(a_0, a_p), "permutation did not reach the adjacency A"


def test_permutation_seed_is_not_inert_for_the_mask():
    """FAIL-LOUD guard: a permuted mask must differ from the unpermuted one.

    Passing this vacuously (identical masks) is exactly the pre-fix behaviour, in which
    `--permutation-seed` changed nothing while the sweep reported it as applied.
    """
    from prism.models.utils import Permutation
    wrap = _real_gt_wrap()
    seq = 14
    g = _graph(6, edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    imap = [{i: [(2 * i + 1, 2 * i + 2)] for i in range(6)}]
    with torch.no_grad():
        base = _mask(wrap, seq, [g], imap)
        permuted = wrap.build_structural_mask(
            seq, [g], imap, DEVICE, dtype=torch.float32,
            permutation=Permutation(seed=3))[0, 0].cpu()
    assert not torch.allclose(base, permuted), \
        "--permutation-seed is INERT for learnable_graph_mask (silent no-op)"


def test_permutation_equals_an_explicitly_relabelled_graph():
    """Where the theory says it must hold: permuting INSIDE build_structural_mask is
    identical to relabelling edge_index up front and permuting nothing. That pins the
    convention (Ψ and A relabelled together, node→token wiring untouched) and rules out a
    fix that merely perturbs the mask."""
    from prism.models.utils import Permutation
    wrap = _real_gt_wrap()
    seq = 14
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    g = _graph(6, edges=edges)
    imap = [{i: [(2 * i + 1, 2 * i + 2)] for i in range(6)}]
    perm = Permutation(seed=3)
    with torch.no_grad():
        permuted = wrap.build_structural_mask(
            seq, [g], imap, DEVICE, dtype=torch.float32, permutation=perm)[0, 0].cpu()
        g2 = Data(x=g.x, num_nodes=6,
                  edge_index=perm.apply(g.edge_index, 6, device=DEVICE))
        g2.node_names = g.node_names
        manual = _mask(wrap, seq, [g2], imap)
    assert torch.allclose(permuted, manual, atol=1e-5), \
        (permuted - manual).abs().max().item()


def test_no_permutation_is_byte_identical_to_before():
    """permutation=None must leave the mask exactly as the default call produces it —
    the training path must not move."""
    wrap = _real_gt_wrap()
    seq = 14
    g = _graph(6, edges=[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    imap = [{i: [(2 * i + 1, 2 * i + 2)] for i in range(6)}]
    with torch.no_grad():
        a = _mask(wrap, seq, [g], imap)
        b = wrap.build_structural_mask(seq, [g], imap, DEVICE, dtype=torch.float32,
                                       permutation=None)[0, 0].cpu()
    assert torch.equal(a, b)


if __name__ == "__main__":
    print(f"device: {DEVICE}")
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
