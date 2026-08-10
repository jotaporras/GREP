"""save_run_dir round-trip: an RL-saved run dir must reload through the SAME
checkpoint machinery eval uses (loaders/is_gnn/resolve_*), byte-identical Ψ
tower. This is the contract that makes vLLM eval work on RL outputs for free.
No spine/vllm/trl needed."""
import torch

from gemma4_fixture import gemma4_31b_shaped
from vllm_graph_helpers import build_hf_graph_model, load_tokenizer

from prism.eval import checkpoint as ckpt_mod
from prism.models import loaders
from prism.training.run_dir import save_run_dir

GNN_CFG = {
    "architecture": "rpearl_llm",
    "text_edge_list": "none",
    "injection_scope": "prompt_only",
    "edge_weights": "binary",
    "spine_tools": "none",
    "icl_examples": 0,
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


def test_save_run_dir_reloads_through_eval_stack(tmp_path):
    base_dir = tmp_path / "base"
    llm = gemma4_31b_shaped()
    llm.save_pretrained(base_dir, safe_serialization=True)
    tokenizer = load_tokenizer()
    tokenizer.save_pretrained(base_dir)

    model = build_hf_graph_model(llm)
    out = tmp_path / "run_dir"
    save_run_dir(model, {**GNN_CFG, "base_model": str(base_dir)}, str(out))
    tokenizer.save_pretrained(out)

    assert ckpt_mod.is_gnn_checkpoint(str(out))
    assert ckpt_mod.resolve_text_edge_list(str(out), True, None) == "none"
    assert ckpt_mod.resolve_edge_weights(str(out)) == "binary"
    assert ckpt_mod.resolve_injection_scope(str(out)) == "prompt_only"

    reloaded, _tok = loaders.graph_augmented_llm_from_pretrained(str(out))
    # device_map="auto" may land the reload on MPS/GPU — compare on CPU.
    for (n1, p1), (n2, p2) in zip(
            model.pe_proj.state_dict().items(),
            reloaded.pe_proj.state_dict().items()):
        assert n1 == n2 and torch.equal(p1.cpu(), p2.cpu())
    assert torch.equal(model.pe_gain.data.cpu(), reloaded.pe_gain.data.cpu())
    if model.pe_norm is not None:
        assert torch.equal(model.pe_norm.weight.cpu(), reloaded.pe_norm.weight.cpu())
