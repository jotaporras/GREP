"""2-step GRPO smoke for the MASK arch (MaskGRPOTrainer) on the tiny Gemma-4
fixture: batched HF rollouts with the decode-consistent injector, mask-armed
loss forward, reward wiring, run-dir save — end to end, no vLLM.

Rewards from a random-init policy are ~constant, so the smoke asserts
MECHANICS (finite loss, prompt cache live, gradient path into the GT stub,
batched injector armed), not learning.
"""
import os

os.environ["WANDB_DISABLED"] = "true"

import pytest

pytest.importorskip("trl")

import torch
from datasets import Dataset
from torch import nn

from vllm_graph_helpers import load_tokenizer, save_fixture_dir

from prism.models import loaders
from prism.models.gnn_llm import LearnableGraphMaskLLM
from prism.training.rewards import make_reward_funcs
from prism.training.trainers_rl import MaskGRPOTrainer

GRAPH = {
    "regions": [
        {"name": "kitchen", "coords": [0.0, 0.0]},
        {"name": "hallway", "coords": [1.0, 0.0]},
        {"name": "garage", "coords": [2.0, 0.0]},
        {"name": "office", "coords": [3.0, 0.0]},
    ],
    "objects": [],
    "region_connections": [["kitchen", "hallway"], ["hallway", "garage"],
                           ["garage", "office"]],
    "object_connections": [],
    "robot_location": "kitchen",
}

GNN_CFG = {
    "architecture": "learnable_graph_mask",
    "text_edge_list": "none",
    "injection_scope": "decode_consistent",
    "edge_weights": "binary",
    "spine_tools": "none",
    "icl_examples": 0,
    "structural_lr_mult": 2.0,
    "mask_alpha": 0.0,
    "mask_psi_scale": "cosine",
    "mask_layer_scope": "dense",
    "mask_k_hops": 1,
}


class _StubPE(nn.Module):
    """Trainable per-node Ψ, graph-size sliced — isolates trainer mechanics
    from the real GraphTransformer (whose Ψ the mask treats identically)."""

    def __init__(self, max_nodes=16, d=8):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(max_nodes, d))

    def forward(self, g, permutation=None):
        return self.emb[:g.num_nodes]


def _dataset():
    prompt = (f"You are a robot planner. Scene graph:{GRAPH}\n"
              "Task: give the shortest route from the kitchen to the office.")
    rows = [{
        "prompt": prompt,
        "scene_graph_dict": GRAPH,
        "answer_regex": r"(?i)\boffice\b",
        "init_node": "kitchen",
    } for _ in range(2)]
    return Dataset.from_list(rows)


def test_mask_grpo_two_steps(tmp_path):
    from peft import LoraConfig
    from trl import GRPOConfig

    model_dir = tmp_path / "base"
    hf_llm = save_fixture_dir(model_dir)
    tokenizer = load_tokenizer()
    graph_model = LearnableGraphMaskLLM(
        hf_llm, _StubPE(), alpha=GNN_CFG["mask_alpha"],
        layer_scope=GNN_CFG["mask_layer_scope"],
        k_hops=GNN_CFG["mask_k_hops"],
        psi_scale=GNN_CFG["mask_psi_scale"]).train()

    args = GRPOConfig(
        output_dir=str(tmp_path / "out"),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_generations=2,
        max_completion_length=8,
        max_prompt_length=None,
        temperature=1.0,
        beta=0.0,
        use_vllm=False,
        remove_unused_columns=False,
        learning_rate=1e-4,
        logging_steps=1,
        max_steps=2,
        report_to=[],
    )
    trainer = MaskGRPOTrainer(
        graph_model,
        gnn_config={**GNN_CFG, "base_model": str(model_dir)},
        args=args,
        rollout_batch_size=4,
        train_dataset=_dataset(),
        reward_funcs=make_reward_funcs(),
        processing_class=tokenizer,
        peft_config=LoraConfig(r=4, lora_alpha=8,
                               target_modules=["q_proj", "v_proj"],
                               task_type="CAUSAL_LM"),
    )
    result = trainer.train()

    assert result.training_loss == result.training_loss  # not NaN
    assert trainer._prompt_cache, "prompt cache never populated — rollouts bypassed the mask"

    # The GT must TRAIN: requires_grad on, in the optimizer, and the loss-side
    # structural bias backpropagates into it.
    structural = trainer._core.structural_parameters()
    assert structural and all(p.requires_grad for p in structural), \
        "GT frozen — mask RL requires unfrozen PE weights"
    opt_params = {id(p) for g in trainer.optimizer.param_groups
                  for p in g["params"]}
    assert all(id(p) in opt_params for p in structural), \
        "GT params missing from the optimizer"
    row = _dataset()[0]
    entry = trainer._entry_for_prompt(row["prompt"])
    pids, pyg_graph, imap, _seqs, _nv, _psi = entry
    device = next(trainer._core.parameters()).device
    bias = trainer._core.build_structural_mask(
        len(pids), [pyg_graph], [imap], device)
    trainer.optimizer.zero_grad(set_to_none=True)
    # Sum only the finite (allowed-pair) entries; the blocked finfo.min
    # constants would swamp the float sum.
    bias[bias > torch.finfo(bias.dtype).min / 2].sum().backward()
    grads = [p.grad for p in structural if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads), \
        "structural bias does not backpropagate into the GT"

    out = tmp_path / "run_dir"
    trainer.save_model(str(out))
    cfg = loaders.load_gnn_config(str(out))
    assert cfg["architecture"] == "learnable_graph_mask"
    assert (out / "gnn_weights.pt").exists()
    assert (out / "adapter_config.json").exists(), "LoRA adapter not saved"
