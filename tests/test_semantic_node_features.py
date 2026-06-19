"""Tests for semantic R-PEARL node features (``pe_node_features="word_embeddings"``).

Instead of the PEARL random probes, R-PEARL can take a deterministic per-node feature
(the mean LLM word-embedding of the node's name tokens) and run ONE GCN pass. These
tests assert the three invariants that make that safe and correct:

  1. The GCN's input width is the embedding dim (``node_feature_dim``), and a forward runs.
  2. The signal is DETERMINISTIC — two forwards give identical Ψ (no random probes).
  3. Coverage is FAIL-LOUD — a graph node with no prompt mention raises (no silent zero).

All tiny, random-init, CPU — no GPU required.
"""
import sys
sys.path.insert(0, "src")

import pytest
import torch
from torch_geometric.data import Data

from prism.models.gnn_llm import GraphAugmentedLLM
from prism.models.gt import SemanticGraphTransformer
from prism.models.r_pearl import RandomGNNPositionalEncodings


def _tiny_llm(hidden=32):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=hidden, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, attn_implementation="eager")
    return LlamaForCausalLM(cfg)


def _semantic_wrap(hidden=32, d_model=8):
    """GraphAugmentedLLM in word-embedding mode (gate open so Ψ is non-trivial)."""
    llm = _tiny_llm(hidden)
    pe_model = RandomGNNPositionalEncodings(
        pe_hidden_channels=16, pe_num_layers=2, d_model=d_model,
        num_samples=4, dropout=0.0, k=1, node_feature_dim=hidden,
    )
    return GraphAugmentedLLM(llm, pe_model, d_model=d_model, pe_gain_init=1.0,
                             use_pe_norm=True, pe_node_features="word_embeddings").eval()


def _graph(n=3):
    ei = torch.tensor([[0, 1, 2], [1, 2, 0]])  # directed triangle
    g = Data(x=torch.zeros(n, 1), edge_index=ei, num_nodes=n)
    g.node_names = [f"node{i}" for i in range(n)]
    return g


def test_gcn_input_width_is_embedding_dim():
    hidden = 32
    wrap = _semantic_wrap(hidden=hidden)
    # The first TAGConv consumes node_feature_dim (= embedding dim), not 1.
    assert wrap.pe_model.node_feature_dim == hidden
    assert wrap.pe_model.pe_gcn.convs[0].in_channels == hidden


def test_semantic_signal_is_deterministic():
    """No random probes ⇒ two forwards give byte-identical Ψ."""
    torch.manual_seed(0)
    wrap = _semantic_wrap()
    n, seq, hidden = 3, 12, 32
    emb = torch.randn(1, seq, hidden)
    graphs = [_graph(n)]
    imap = [{0: [(1, 3)], 1: [(4, 6)], 2: [(7, 8)]}]  # every node mentioned
    with torch.no_grad():
        psi_a = wrap.build_pe_signal(emb, graphs, imap)
        psi_b = wrap.build_pe_signal(emb, [_graph(n)], imap)
    assert torch.equal(psi_a, psi_b), "semantic Ψ should be deterministic (no probes)"
    # And it is actually non-zero at the mention spans (gate open, features non-trivial).
    assert psi_a.abs().max().item() > 0, "Ψ should be non-zero where nodes are mentioned"


def test_random_mode_is_nondeterministic_contrast():
    """Sanity contrast: random-probe mode DOES vary forward-to-forward (probes resampled)."""
    torch.manual_seed(0)
    llm = _tiny_llm()
    pe_model = RandomGNNPositionalEncodings(pe_hidden_channels=16, pe_num_layers=2,
                                            d_model=8, num_samples=8, dropout=0.0, k=2)
    wrap = GraphAugmentedLLM(llm, pe_model, d_model=8, pe_gain_init=1.0,
                             use_pe_norm=True, pe_node_features="random").eval()
    emb = torch.randn(1, 12, 32)
    imap = [{0: [(1, 3)], 1: [(4, 6)], 2: [(7, 8)]}]
    with torch.no_grad():
        a = wrap.build_pe_signal(emb, [_graph(3)], imap)
        b = wrap.build_pe_signal(emb, [_graph(3)], imap)
    assert not torch.equal(a, b), "random-probe Ψ should differ between forwards"


def test_uncovered_node_fails_loud():
    """A graph node with no mention span must raise (no silent zero feature)."""
    wrap = _semantic_wrap()
    emb = torch.randn(1, 12, 32)
    imap = [{0: [(1, 3)], 1: [(4, 6)]}]  # node 2 missing → must raise
    with pytest.raises(ValueError, match="word_embeddings"):
        wrap.build_pe_signal(emb, [_graph(3)], imap)


def test_rpearl_deterministic_forward_directly():
    """RandomGNNPositionalEncodings.forward is deterministic in semantic mode."""
    torch.manual_seed(0)
    hidden = 32
    pe = RandomGNNPositionalEncodings(pe_hidden_channels=16, pe_num_layers=2, d_model=8,
                                      num_samples=4, dropout=0.0, k=1,
                                      node_feature_dim=hidden).eval()
    g = _graph(3); g.x = torch.randn(3, hidden)
    with torch.no_grad():
        o1 = pe(Data(x=g.x.clone(), edge_index=g.edge_index, num_nodes=3))
        o2 = pe(Data(x=g.x.clone(), edge_index=g.edge_index, num_nodes=3))
    assert o1.shape == (3, 8)
    assert torch.equal(o1, o2)


def _semantic_gt_wrap(hidden=32, d_model=8):
    """GraphAugmentedLLM whose pe_model is a pure GT over word-embeddings (gt_llm arch)."""
    llm = _tiny_llm(hidden)
    pe_model = SemanticGraphTransformer(
        node_feature_dim=hidden, d_model=d_model, num_layers=2, heads=4, dropout=0.0, k_gt=1,
    )
    return GraphAugmentedLLM(llm, pe_model, d_model=d_model, pe_gain_init=1.0,
                             use_pe_norm=True, pe_node_features="word_embeddings").eval()


def test_semantic_gt_forward_direct():
    """SemanticGraphTransformer: projects [N, d_emb] -> blocks -> [N, d_model], deterministic."""
    torch.manual_seed(0)
    hidden, d_model = 32, 8
    gt = SemanticGraphTransformer(node_feature_dim=hidden, d_model=d_model,
                                  num_layers=2, heads=4, dropout=0.0, k_gt=1).eval()
    g = _graph(4); g.x = torch.randn(4, hidden)
    with torch.no_grad():
        o1 = gt(Data(x=g.x.clone(), edge_index=g.edge_index, num_nodes=4))
        o2 = gt(Data(x=g.x.clone(), edge_index=g.edge_index, num_nodes=4))
    assert o1.shape == (4, d_model)
    assert torch.isfinite(o1).all()
    assert torch.equal(o1, o2), "gt_llm PE should be deterministic"


def test_gt_llm_end_to_end_signal():
    """GraphAugmentedLLM(pe_model=SemanticGraphTransformer) builds a finite, deterministic Ψ."""
    torch.manual_seed(0)
    wrap = _semantic_gt_wrap()
    emb = torch.randn(1, 12, 32)
    imap = [{0: [(1, 3)], 1: [(4, 6)], 2: [(7, 8)]}]
    with torch.no_grad():
        a = wrap.build_pe_signal(emb, [_graph(3)], imap)
        b = wrap.build_pe_signal(emb, [_graph(3)], imap)
    assert torch.isfinite(a).all() and a.abs().max().item() > 0
    assert torch.equal(a, b)


def test_gt_llm_uncovered_node_fails_loud():
    wrap = _semantic_gt_wrap()
    emb = torch.randn(1, 12, 32)
    imap = [{0: [(1, 3)], 1: [(4, 6)]}]  # node 2 missing
    with pytest.raises(ValueError, match="word_embeddings"):
        wrap.build_pe_signal(emb, [_graph(3)], imap)


if __name__ == "__main__":
    test_gcn_input_width_is_embedding_dim(); print("input-width: PASS")
    test_semantic_signal_is_deterministic(); print("determinism: PASS")
    test_random_mode_is_nondeterministic_contrast(); print("random-contrast: PASS")
    test_uncovered_node_fails_loud(); print("fail-loud: PASS")
    test_rpearl_deterministic_forward_directly(); print("rpearl-forward: PASS")
    print("done")
