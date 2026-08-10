"""Locally-runnable M2 tests: engine policy gate, Ψ-producer equivalence,
serving-dir resolution.

No vllm and no spine imports — this file must run in both the training env and
the vllm venv. The SPINE-client wiring (vllm_graph.spine_client + the evaluate
``client=`` seam) needs the cluster's spine package and a GPU engine; its
end-to-end check is the M2 parity run on a real e14 checkpoint.
"""
import json
import os

import pytest
import torch

from gemma4_fixture import gemma4_31b_shaped
from vllm_graph_helpers import PROMPT, load_tokenizer, make_graph

from prism.models import loaders
from prism.models.vllm_graph.engine import checkpoint_engine_policy, materialize_serving_dir
from prism.models.vllm_graph.psi import build_psi_transport
from prism.models.vllm_graph.psi_producer import load_psi_producer

# The gnn hyperparameters recorded by a fake rpearl_llm run — values chosen to
# match RandomGNNPositionalEncodings defaults where the test's reference model
# uses defaults, so reference and producer rebuild identical towers.
GNN_CFG = {
    "arch": "rpearl_llm",
    "pe_hidden_channels": 32,
    "pe_num_layers": 2,
    "d_model": 16,
    "num_samples": 8,
    "dropout": 0.0,
    "k_pe": 3,
    "eps": 1e-8,
    "use_layer_norm": True,
    "pe_gain_init": 1.0,
    "use_pe_norm": True,
    "disable_graph_token_rope": False,
    "pe_node_features": "random",
}


def _write_ckpt(tmp_path, *, architecture, base_model, gnn=None, gnn_weights=None):
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    with open(ckpt / "train_config.json", "w") as f:
        json.dump({"architecture": architecture, "base_model": str(base_model),
                   "gnn": gnn or {}}, f)
    if gnn_weights is not None:
        torch.save(gnn_weights, ckpt / "gnn_weights.pt")
    return str(ckpt)


def test_policy_additive_ok(tmp_path):
    ckpt = _write_ckpt(tmp_path, architecture="gt_llm", base_model="some/base",
                       gnn={"disable_graph_token_rope": True})
    policy = checkpoint_engine_policy(ckpt)
    assert policy["architecture"] == "gt_llm"
    assert policy["identity_rope"] is True
    assert policy["pe_inject_value"] is True
    assert policy["base_model"] == "some/base"


def test_policy_mask_arch_raises(tmp_path):
    ckpt = _write_ckpt(tmp_path, architecture="learnable_graph_mask",
                       base_model="some/base")
    with pytest.raises(ValueError, match="not vLLM-servable"):
        checkpoint_engine_policy(ckpt)


def test_serving_dir_without_adapter_is_base_model(tmp_path):
    ckpt = _write_ckpt(tmp_path, architecture="rpearl_llm", base_model="some/base")
    assert materialize_serving_dir(ckpt, is_gnn=True) == "some/base"


@pytest.fixture(scope="module")
def fixture_base_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("gemma4_base")
    llm = gemma4_31b_shaped()
    llm.save_pretrained(d, safe_serialization=True)
    return str(d), llm


def test_psi_producer_matches_full_model(tmp_path, fixture_base_dir):
    """The embeddings-only shim must reproduce the full-model Ψ transport
    bit-for-bit — same recorded config, same gnn_weights.pt, same RNG."""
    base_dir, llm = fixture_base_dir
    tokenizer = load_tokenizer()

    reference = loaders.additive_model_from_config(
        llm, {**GNN_CFG, "base_model": base_dir}, _save_weights(tmp_path, llm))
    reference.eval()

    ckpt = _write_ckpt(
        tmp_path, architecture="rpearl_llm", base_model=base_dir, gnn=GNN_CFG,
        gnn_weights=_weights_from(reference))
    producer = load_psi_producer(ckpt)

    prompt_ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"]
    graph = make_graph()
    torch.manual_seed(7)
    ref_transport, ref_map = build_psi_transport(reference, tokenizer, prompt_ids, graph)
    torch.manual_seed(7)
    got_transport, got_map = build_psi_transport(producer, tokenizer, prompt_ids, graph)

    assert got_map == ref_map
    assert torch.equal(got_transport, ref_transport)


def _save_weights(tmp_path, llm):
    """A weights dir for the REFERENCE build: fresh random tower saved out."""
    from prism.models.r_pearl import RandomGNNPositionalEncodings
    from prism.models.gnn_llm import GraphAugmentedLLM
    torch.manual_seed(0)
    pe_model = RandomGNNPositionalEncodings(
        pe_hidden_channels=GNN_CFG["pe_hidden_channels"],
        pe_num_layers=GNN_CFG["pe_num_layers"],
        d_model=GNN_CFG["d_model"],
        num_samples=GNN_CFG["num_samples"],
        dropout=GNN_CFG["dropout"],
        k=GNN_CFG["k_pe"],
        eps=GNN_CFG["eps"],
        use_layer_norm=GNN_CFG["use_layer_norm"],
    )
    seed_model = GraphAugmentedLLM(
        llm, pe_model, d_model=GNN_CFG["d_model"], pe_gain_init=1.0)
    d = tmp_path / "seed_weights"
    d.mkdir()
    torch.save(_weights_from(seed_model), d / "gnn_weights.pt")
    return str(d)


def _weights_from(model):
    w = {
        "pe_model": model.pe_model.state_dict(),
        "pe_proj": model.pe_proj.state_dict(),
        "pe_gain": model.pe_gain.data,
    }
    if model.pe_norm is not None:
        w["pe_norm"] = model.pe_norm.state_dict()
    return w
