"""Smoke test: train Qwen2.5-0.5B for 5 steps on both architectures, save, reload, generate.
This script was made to make sure checkpoints are saved and loaded correctly."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # single GPU: avoids DataParallel wrapping GraphAugmentedLLM
import shutil
import json

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from prism.training.train_v2 import (
    GraphSFTTrainer,
    _ensure_pad_tokens,
    _model_short_name,
    _standardize_conversations,
)
from prism.data import data_col
from prism.models import gnn_llm
from prism.models import loaders
from prism.models import r_pearl

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SMOKE_DIR = "outputs/smoke_lora"
MAX_STEPS = 5

# A minimal scene graph that safe_parse_graph can handle
SCENE_GRAPH = (
    "{'objects': [{'name': 'shed_1', 'coords': [1, 3]}, "
    "{'name': 'house_1', 'coords': [-1, -1]}], "
    "'regions': [{'name': 'field_1', 'coords': [0, 1]}], "
    "'object_connections': [['shed_1', 'field_1'], ['house_1', 'field_1']], "
    "'region_connections': [], "
    "'robot_location': 'field_1'}"
)

FAKE_CONVERSATIONS = [
    [
        {"role": "user", "content": f"task: Go to the shed.Scene graph:{SCENE_GRAPH}"},
        {"role": "assistant", "content": '{"plan": "[goto(field_1)]"}'},
    ],
    [
        {"role": "user", "content": f"task: Find a tool.Scene graph:{SCENE_GRAPH}"},
        {"role": "assistant", "content": '{"plan": "[goto(field_1), map_region(field_1)]"}'},
    ],
    [
        {"role": "user", "content": f"task: Explore the area.Scene graph:{SCENE_GRAPH}"},
        {"role": "assistant", "content": '{"plan": "[map_region(field_1)]"}'},
    ],
    [
        {"role": "user", "content": f"task: Check the house.Scene graph:{SCENE_GRAPH}"},
        {"role": "assistant", "content": '{"plan": "[goto(field_1)]"}'},
    ],
    [
        {"role": "user", "content": f"task: Navigate home.Scene graph:{SCENE_GRAPH}"},
        {"role": "assistant", "content": '{"plan": "[goto(field_1)]"}'},
    ],
]


def _make_dataset(tokenizer):
    """Build a tiny Dataset with messages + tokenized fields for both architectures."""
    records = []
    for conv in FAKE_CONVERSATIONS:
        tokenized = tokenizer.apply_chat_template(conv, tokenize=True, return_dict=True)
        records.append({
            "conversations": conv,
            "messages": conv,
            **tokenized,
        })
    ds = Dataset.from_list(records)
    ds = _standardize_conversations(ds, tokenizer)
    return ds


def _snap_params(model, key_filter, max_params=5):
    """Return up to max_params named parameters whose key contains key_filter."""
    snap = {}
    for k, v in model.named_parameters():
        if key_filter in k:
            snap[k] = v.detach().cpu().clone()
        if len(snap) >= max_params:
            break
    return snap


def _assert_weights_match(before, after, label):
    assert set(before.keys()) == set(after.keys()), (
        f"{label}: key mismatch\n  before: {sorted(before.keys())}\n  after:  {sorted(after.keys())}"
    )
    for k in before:
        assert torch.allclose(before[k].float(), after[k].float()), f"{label}: mismatch at {k}"
    print(f"  Weights match ({len(before)} tensors checked): {label}")


def _lora_config():
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )


def _sft_config(output_dir):
    return SFTConfig(
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        fp16=False,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        dataset_num_proc=1,
        max_length=512,
    )


# ── Test 1: Plain LLM architecture ──────────────────────────────────────────

def test_llm():
    print(f"\n{'='*60}")
    print("TEST: llm architecture (plain LoRA)")
    print(f"{'='*60}")

    model_slug = _model_short_name(BASE_MODEL)
    ckpt_dir = os.path.join(SMOKE_DIR, f"smoke_llm_{model_slug}_r8")

    llm = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    _ensure_pad_tokens(tokenizer, llm)

    ds = _make_dataset(tokenizer)

    trainer = SFTTrainer(
        model=llm,
        processing_class=tokenizer,
        peft_config=_lora_config(),
        train_dataset=ds,
        args=_sft_config(ckpt_dir),
    )
    trainer.train()
    lora_before = _snap_params(trainer.model, "lora_A")
    trainer.save_model()
    tokenizer.save_pretrained(ckpt_dir)
    print(f"  Saved to {ckpt_dir}")

    # Reload via loaders.from_pretrained
    loaded_model, loaded_tok = loaders.from_pretrained(ckpt_dir)
    lora_after = _snap_params(loaded_model, "lora_A")
    _assert_weights_match(lora_before, lora_after, "llm LoRA-A weights")
    prompt = loaded_tok.apply_chat_template(
        [{"role": "user", "content": "Say hello."}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = loaded_tok(prompt, return_tensors="pt").to(loaded_model.device)
    out = loaded_model.generate(**inputs, max_new_tokens=20)
    decoded = loaded_tok.decode(out[0], skip_special_tokens=True)
    print(f"  Generate output: {decoded[:120]}")
    print("  PASS: llm")
    return True


# ── Test 2: rpearl_llm architecture ─────────────────────────────────────────

def test_rpearl_llm():
    print(f"\n{'='*60}")
    print("TEST: rpearl_llm architecture (GraphAugmentedLLM + LoRA)")
    print(f"{'='*60}")

    model_slug = _model_short_name(BASE_MODEL)
    ckpt_dir = os.path.join(SMOKE_DIR, f"smoke_rpearl_llm_{model_slug}_r8")

    llm = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    _ensure_pad_tokens(tokenizer, llm)

    d_model = 896  # Qwen2.5-0.5B hidden_size is 896
    pe_model = r_pearl.RandomGNNPositionalEncodings(
        pe_hidden_channels=64,
        pe_num_layers=2,
        d_model=d_model,
        num_samples=5,
        dropout=0.1,
        k=2,
        use_layer_norm=True,
    )
    model = gnn_llm.GraphAugmentedLLM(llm, pe_model, tokenizer, pe_dim=d_model)

    collator = data_col.DataCollatorForGraphAugmentedLLM(tokenizer, mlm=False, text_edge_list="present")
    ds = _make_dataset(tokenizer)

    gnn_config = {
        "base_model": BASE_MODEL,
        "pe_hidden_channels": 64,
        "pe_num_layers": 2,
        "d_model": d_model,
        "num_samples": 5,
        "dropout": 0.1,
        "k": 2,
        "use_layer_norm": True,
    }

    trainer = GraphSFTTrainer(
        model=model,
        data_collator=collator,
        processing_class=tokenizer,
        peft_config=_lora_config(),
        train_dataset=ds,
        args=_sft_config(ckpt_dir),
        gnn_config=gnn_config,
    )
    trainer.train()
    gnn_before = {
        **{f"pe_model.{k}": v.detach().cpu().clone() for k, v in trainer.model.pe_model.named_parameters()},
        **{f"pe_proj.{k}": v.detach().cpu().clone() for k, v in trainer.model.pe_proj.named_parameters()},
    }
    trainer.save_model()
    tokenizer.save_pretrained(ckpt_dir)
    print(f"  Saved to {ckpt_dir}")

    # Reload via loaders.graph_augmented_llm_from_pretrained
    loaded_model, loaded_tok = loaders.graph_augmented_llm_from_pretrained(ckpt_dir)
    gnn_after = {
        **{f"pe_model.{k}": v.detach().cpu().clone() for k, v in loaded_model.pe_model.named_parameters()},
        **{f"pe_proj.{k}": v.detach().cpu().clone() for k, v in loaded_model.pe_proj.named_parameters()},
    }
    _assert_weights_match(gnn_before, gnn_after, "rpearl_llm GNN weights (pe_model + pe_proj)")
    print(f"  Loaded GraphAugmentedLLM successfully")
    print("  PASS: rpearl_llm")
    return True


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Clean up previous smoke output
    if os.path.isdir(SMOKE_DIR):
        shutil.rmtree(SMOKE_DIR)
    os.makedirs(SMOKE_DIR, exist_ok=True)

    results = {}
    results["llm"] = test_llm()
    results["rpearl_llm"] = test_rpearl_llm()

    print(f"\n{'='*60}")
    print("SMOKE TEST RESULTS")
    print(f"{'='*60}")
    for arch, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {arch}: {status}")

    # Also verify _model_short_name
    assert _model_short_name("meta-llama/Llama-3.1-8B-Instruct") == "llama-3.1-8b"
    assert _model_short_name("Qwen/Qwen2.5-0.5B-Instruct") == "qwen2.5-0.5b"
    assert _model_short_name("meta-llama/Llama-3.2-3B-Instruct") == "llama-3.2-3b"
    print("  _model_short_name: PASS")
