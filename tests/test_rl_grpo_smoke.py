"""2-step GRPO smoke on the tiny Gemma-4 fixture: the full RL loop —
graph-conditioned vLLM rollouts, Ψ-armed loss forward, LoRA→engine weight
sync, reward wiring, run-dir save — end to end on CPU.

Rewards from a random-init policy are ~constant (≈0 on path components), so the
loss is near zero — the smoke asserts MECHANICS (finite loss, Ψ cache and dbg
counters live, checkpoint reloads), not learning. Reward ascent is the plaza
PoC's job (M4).
"""
import os

# Before the FIRST vllm import anywhere in the process.
os.environ["VLLM_PLUGINS"] = ""
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["WANDB_DISABLED"] = "true"

import pytest

pytest.importorskip("vllm")
pytest.importorskip("trl")

import torch
from datasets import Dataset

from vllm_graph_helpers import build_hf_graph_model, load_tokenizer, save_fixture_dir, spin_engine

from prism.models import loaders
from prism.training.rewards import make_reward_funcs
from prism.training.trainers_rl import GraphGRPOTrainer

GRAPH = {
    "regions": [
        # 2-D coords, matching the data_gen graph format.
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


def test_grpo_two_steps(tmp_path):
    from peft import LoraConfig
    from trl import GRPOConfig

    model_dir = tmp_path / "base"
    hf_llm = save_fixture_dir(model_dir)
    tokenizer = load_tokenizer()
    graph_model = build_hf_graph_model(hf_llm).train()
    # The trainer syncs the policy as a LoRARequest — engine must accept it.
    # max_lora_rank must be one of vLLM's allowed sizes (8 is the floor that
    # covers the r=4 adapter); LoRA Triton kernels are fp16/bf16-only, so this
    # smoke runs the engine at bf16 (parity tests keep float32, no LoRA).
    rollout_llm, wrapper = spin_engine(model_dir, enable_lora=True,
                                       max_lora_rank=8, dtype="bfloat16")

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
    trainer = GraphGRPOTrainer(
        graph_model,
        gnn_config={**GNN_CFG, "base_model": str(model_dir)},
        rollout_llm=rollout_llm,
        rollout_wrapper=wrapper,
        args=args,
        sync_every=1,
        train_dataset=_dataset(),
        reward_funcs=make_reward_funcs(),
        processing_class=tokenizer,
        peft_config=LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"],
                               task_type="CAUSAL_LM"),
    )
    result = trainer.train()

    assert result.training_loss == result.training_loss  # not NaN
    assert trainer._transport_cache, "Ψ cache never populated — rollouts bypassed Ψ"
    assert wrapper.dbg["attn_hit"] > 0, "engine attention never consumed Ψ"
    assert trainer._lora_request is not None, "policy LoRA never synced to engine"
    assert trainer._lora_version >= 1

    out = tmp_path / "run_dir"
    trainer.save_model(str(out))
    cfg = loaders.load_gnn_config(str(out))
    assert cfg["architecture"] == "rpearl_llm"
    assert (out / "gnn_weights.pt").exists()
    assert (out / "adapter_config.json").exists(), "LoRA adapter not saved"
