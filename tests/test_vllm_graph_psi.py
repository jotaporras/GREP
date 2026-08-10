"""Unit tests for driver-side Ψ transport construction (vllm_graph.psi).

No vllm import — this module must stay runnable in the training env
(GREP-PRISM-v3). The Ψ producer is a real GraphAugmentedLLM over the tiny
Gemma-4 fixture, so spans, scaling, and the identity-RoPE fail-loud check are
exercised against the actual HF chain, not a reimplementation.
"""
import pytest
import torch
from torch_geometric.data import Data

from gemma4_fixture import BASE_MODEL, gemma4_31b_shaped

from prism.models.gnn_llm import GraphAugmentedLLM, build_injection_map, node_token_variants
from prism.models.r_pearl import RandomGNNPositionalEncodings
from prism.models.vllm_graph.psi import build_psi_transport, transport_dim

NODE_NAMES = ["kitchen", "hallway", "garage", "bedroom", "office"]
EDGES = [(0, 1), (1, 2), (1, 3), (3, 4)]


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)


@pytest.fixture(scope="module")
def graph():
    edge_index = torch.tensor(EDGES + [(b, a) for a, b in EDGES]).T
    g = Data(x=torch.zeros(len(NODE_NAMES), 1), edge_index=edge_index,
             num_nodes=len(NODE_NAMES))
    g.node_names = list(NODE_NAMES)
    return g


def _graph_model(disable_graph_token_rope=False, seed=0):
    llm = gemma4_31b_shaped(seed=seed)
    torch.manual_seed(seed)
    # fixed_seed_mode: R-PEARL resamples probes per forward otherwise, and these
    # tests compare Ψ across calls.
    pe_model = RandomGNNPositionalEncodings(
        pe_hidden_channels=32, pe_num_layers=2, d_model=16, num_samples=8,
        fixed_seed_mode=True,
    )
    return GraphAugmentedLLM(
        llm, pe_model, d_model=16, pe_gain_init=1.0,
        disable_graph_token_rope=disable_graph_token_rope,
    ).eval()  # eval semantics: dropout off, as in the eval/rollout path


def _prompt_ids(tokenizer):
    text = ("Scene graph nodes: kitchen, hallway, garage, bedroom, office. "
            "You are in the kitchen. Give the shortest route to the office.")
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def test_transport_shape_and_span_column(tokenizer, graph):
    model = _graph_model()
    prompt_ids = _prompt_ids(tokenizer)
    transport, imap = build_psi_transport(model, tokenizer, prompt_ids, graph)

    hidden = model.llm.config.hidden_size
    assert transport.shape == (len(prompt_ids), transport_dim(hidden))
    assert set(imap) == set(range(len(NODE_NAMES))), "every node must be mentioned"

    span_col = transport[:, hidden]
    injected = set()
    for spans in imap.values():
        for s, e in spans:
            injected.update(range(s, e))
    for t in range(len(prompt_ids)):
        assert (span_col[t] > 0.5) == (t in injected)

    # Ψ zero exactly off the injected spans, nonzero on them.
    psi = transport[:, :hidden]
    off = [t for t in range(len(prompt_ids)) if t not in injected]
    assert torch.all(psi[off] == 0)
    assert torch.all(psi[sorted(injected)].abs().sum(-1) > 0)


def test_psi_matches_build_pe_signal(tokenizer, graph):
    """Transport Ψ must be bit-identical to the HF chain's build_pe_signal."""
    model = _graph_model()
    prompt_ids = _prompt_ids(tokenizer)
    transport, imap = build_psi_transport(model, tokenizer, prompt_ids, graph)

    ids = torch.tensor([prompt_ids])
    with torch.no_grad():
        emb = model.llm.get_input_embeddings()(ids)
        ref = model.build_pe_signal(emb, [graph], [imap])[0]
    hidden = model.llm.config.hidden_size
    assert torch.equal(transport[:, :hidden], ref.float())


def test_injection_map_matches_module_fn(tokenizer, graph):
    model = _graph_model()
    prompt_ids = _prompt_ids(tokenizer)
    _, imap = build_psi_transport(model, tokenizer, prompt_ids, graph)
    ref = build_injection_map(
        prompt_ids, node_token_variants(NODE_NAMES, tokenizer), scope_start=0)
    assert imap == ref


def test_identity_rope_final_position_fails_loud(tokenizer, graph):
    """Prompt ending inside a node mention + identity-RoPE must raise (parity
    with inference._identity_rope_kwargs's check)."""
    model = _graph_model(disable_graph_token_rope=True)
    text = "Scene graph nodes: kitchen, hallway, garage, bedroom, office"
    prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    with pytest.raises(RuntimeError, match="FINAL prompt position"):
        build_psi_transport(model, tokenizer, prompt_ids, graph)
