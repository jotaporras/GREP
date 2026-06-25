"""Tests for the structural-attention-mask architecture (``graph_mask_llm``).

``GraphMaskLLM`` adds NO positional encoding and NO graph parameters. Its only effect
is the attention MASK: two token positions that both belong to graph nodes may attend
to each other only if those nodes share an edge (within ``k_hops``) — otherwise the
additive bias is ``finfo.min`` (a_ij = 0). These tests assert that invariant on the
mask builder directly, plus the adjacency rules and an end-to-end forward/generate.

All tiny, random-init, CPU — no GPU required.
"""
import sys
sys.path.insert(0, "src")

import torch
from torch_geometric.data import Data

from prism.models.gnn_llm import GraphMaskLLM


def _tiny_llm(hidden=32):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=hidden, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, attn_implementation="eager")
    return LlamaForCausalLM(cfg)


def _wrap(k_hops=1, symmetrize=True, hidden=32):
    return GraphMaskLLM(_tiny_llm(hidden), k_hops=k_hops, symmetrize=symmetrize).eval()


def _graph(n, edges):
    """Directed edge list -> PyG Data (edges as given; symmetrization is the wrap's job)."""
    if edges:
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        ei = torch.zeros(2, 0, dtype=torch.long)
    g = Data(x=torch.zeros(n, 1), edge_index=ei, num_nodes=n)
    g.node_names = [f"node{i}" for i in range(n)]
    return g


NEG = torch.finfo(torch.float32).min


def test_mask_blocks_nonadjacent_node_pairs():
    """node2 is isolated; node0–node1 share an edge. Causally-later non-adjacent
    node pairs are −inf; adjacent / same-node / non-node pairs are 0."""
    wrap = _wrap()
    seq = 8
    # node0@1, node1@3, node2@5 (BOS=0 and gaps are non-node tokens)
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1)])  # only 0<=>1
    bias = wrap.build_structural_mask(seq, [g], imap, torch.device("cpu"), dtype=torch.float32)[0, 0]

    # node2 (pos5) attending node0 (pos1) and node1 (pos3): no edge -> blocked.
    assert bias[5, 1] == NEG
    assert bias[5, 3] == NEG
    # node1 (pos3) attending node0 (pos1): edge exists -> allowed.
    assert bias[3, 1] == 0
    # self / same position -> allowed.
    assert bias[1, 1] == 0 and bias[5, 5] == 0
    # node token attending a NON-node token (BOS) -> never blocked.
    assert bias[5, 0] == 0
    # non-node row (pos 7) -> entirely 0.
    assert torch.all(bias[7] == 0)


def test_symmetrize_makes_directed_edge_bidirectional():
    """Edge 0->1 only. With symmetrize, both (n1,n0) and (n0,n1) are allowed."""
    seq = 6
    imap = [{0: [(1, 2)], 1: [(3, 4)]}]
    g = _graph(2, edges=[(0, 1)])  # directed 0->1

    sym = _wrap(symmetrize=True).build_structural_mask(seq, [g], imap, torch.device("cpu"), dtype=torch.float32)[0, 0]
    # pos3 (node1) attends pos1 (node0): causal & symmetric edge -> allowed.
    assert sym[3, 1] == 0

    asym = _wrap(symmetrize=False).build_structural_mask(seq, [g], imap, torch.device("cpu"), dtype=torch.float32)[0, 0]
    # Without symmetrization there is no 1->0 edge, so node1 cannot attend node0.
    assert asym[3, 1] == NEG


def test_k_hops_widens_neighbourhood():
    """Chain 0-1-2. k=1 blocks 0<=>2; k=2 allows it (2 hops via node1)."""
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1), (1, 2)])

    k1 = _wrap(k_hops=1).build_structural_mask(seq, [g], imap, torch.device("cpu"), dtype=torch.float32)[0, 0]
    assert k1[5, 1] == NEG  # node2 -> node0: 2 hops, blocked at k=1

    k2 = _wrap(k_hops=2).build_structural_mask(seq, [g], imap, torch.device("cpu"), dtype=torch.float32)[0, 0]
    assert k2[5, 1] == 0    # node2 -> node0: reachable within 2 hops


def test_node_adjacency_has_self_loops_and_symmetry():
    wrap = _wrap(symmetrize=True)
    g = _graph(3, edges=[(0, 1)])
    adj = wrap._node_adjacency(g, torch.device("cpu"))
    assert adj.dtype == torch.bool and adj.shape == (3, 3)
    assert bool(adj[0, 0]) and bool(adj[1, 1]) and bool(adj[2, 2])  # self-loops
    assert bool(adj[0, 1]) and bool(adj[1, 0])                       # symmetric
    assert not bool(adj[0, 2]) and not bool(adj[2, 0])               # no edge


def test_no_node_row_is_fully_masked():
    """Every node-token row keeps at least its diagonal + BOS, so softmax is safe."""
    wrap = _wrap()
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[])  # no edges at all (worst case)
    bias = wrap.build_structural_mask(seq, [g], imap, torch.device("cpu"), dtype=torch.float32)[0, 0]
    # Each node row has a 0 at the diagonal and at BOS (pos 0).
    for p in (1, 3, 5):
        assert (bias[p] == 0).any()
        assert bias[p, p] == 0 and bias[p, 0] == 0


def test_repeated_node_mentions_attend_each_other():
    """Two mentions of the SAME node (no edges) still attend (self-loop)."""
    wrap = _wrap()
    seq = 8
    imap = [{0: [(1, 2), (5, 6)]}]  # node0 mentioned twice
    g = _graph(1, edges=[])
    bias = wrap.build_structural_mask(seq, [g], imap, torch.device("cpu"), dtype=torch.float32)[0, 0]
    assert bias[5, 1] == 0  # second mention attends first (same node)


def test_use_edges_false_blocks_all_cross_node():
    """Edgeless ablation: with use_edges=False the adjacency is self-loops only, so
    EVERY node token is blocked from attending to any OTHER node token (even adjacent)."""
    wrap = _wrap()  # default use_edges=True
    abl = GraphMaskLLM(_tiny_llm(), k_hops=1, symmetrize=True, use_edges=False).eval()
    seq = 8
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1), (1, 2)])  # has real edges

    full = wrap.build_structural_mask(seq, [g], imap, torch.device("cpu"), dtype=torch.float32)[0, 0]
    none = abl.build_structural_mask(seq, [g], imap, torch.device("cpu"), dtype=torch.float32)[0, 0]
    # With edges: node1@3 attends node0@1 (adjacent) -> allowed.
    assert full[3, 1] == 0
    # Edgeless: that same adjacent pair is now blocked.
    assert none[3, 1] == NEG
    # Edgeless adjacency is pure identity: self-loop allowed, all other node pairs blocked.
    adj = abl._node_adjacency(g, torch.device("cpu"))
    assert bool(adj.diag().all()) and adj.sum() == 3  # only the 3 diagonal entries
    # Self and non-node still reachable (no fully-masked row).
    assert none[3, 3] == 0 and none[3, 0] == 0


def test_forward_runs_and_mask_changes_logits():
    """End-to-end: forward with graphs differs from plain causal, and disarms after."""
    torch.manual_seed(0)
    wrap = _wrap()
    seq = 8
    input_ids = torch.randint(0, 64, (1, seq))
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1)])  # node2 isolated -> mask blocks real, causal pairs

    with torch.no_grad():
        masked = wrap(input_ids=input_ids, graphs=[g], injection_maps=imap).logits
        plain = wrap(input_ids=input_ids, graphs=None, injection_maps=None).logits

    assert torch.isfinite(masked).all()
    # The structural mask removes causally-allowed node->node attention, so it MUST
    # change the output relative to the unmasked causal forward.
    assert not torch.allclose(masked, plain, atol=1e-5)
    # Bias is disarmed after a (non-checkpointed) forward.
    assert wrap._struct_bias is None


def test_generate_decode_skips_mask_without_error():
    """Arming the prompt-length bias and generating must not crash: decode steps
    (query len 1) fall through the shape guard."""
    torch.manual_seed(0)
    wrap = _wrap()
    seq = 8
    input_ids = torch.randint(0, 64, (1, seq))
    imap = [{0: [(1, 2)], 1: [(3, 4)], 2: [(5, 6)]}]
    g = _graph(3, edges=[(0, 1)])
    wrap._struct_bias = wrap.build_structural_mask(seq, [g], imap, torch.device("cpu"), dtype=torch.float32)
    try:
        with torch.no_grad():
            out = wrap.llm.generate(input_ids=input_ids, max_new_tokens=4, do_sample=False,
                                    use_cache=True, pad_token_id=0)
        assert out.shape[1] == seq + 4
    finally:
        wrap._struct_bias = None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
