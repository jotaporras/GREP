# hf_sft_lora.py
import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Env first (so W&B picks these up as soon as possible)
os.environ.setdefault("WANDB_PROJECT", "SLM-distill")
# Match your original env var usage
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")  # harmless here

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer, apply_chat_template


# ----------------------------
# Utilities
# ----------------------------
def _bf16_supported() -> bool:
    """Conservative check for bfloat16 support."""
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except Exception:
        return False


def _ensure_pad_tokens(tokenizer, model):
    # Many chat models use EOS as PAD during training
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id


def _normalize_role(role: str) -> str:
    role = (role or "").lower()
    if role in ("user", "human", "prompt", "customer", "asker"):
        return "user"
    if role in ("assistant", "gpt", "bot", "model"):
        return "assistant"
    if role in ("system", "developer"):
        return "system"
    # fallback—treat unknown as user
    return "user"


def _sharegpt_to_messages(example: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    """
    TO DO: oct 29 not sure if necessary.
    Convert common JSON patterns into a transformers-friendly `messages` list:
      - { "conversations": [{"from": "...", "value": "..."} ...] }
      - { "messages": [{"role": "...", "content": "..."} ...] }
      - Alpaca-style: {"instruction": "...", "input": "...", "output": "..."}
    Returns None if we can't detect any supported schema.
    """
    # Already in messages format
    if "messages" in example and isinstance(example["messages"], list):
        msgs = []
        for m in example["messages"]:
            if not isinstance(m, dict):
                continue
            role = _normalize_role(m.get("role", "user"))
            content = m.get("content", "")
            if content is None:
                content = ""
            msgs.append({"role": role, "content": str(content)})
        return msgs if msgs else None

    # ShareGPT style
    if "conversations" in example and isinstance(example["conversations"], list):
        msgs = []
        for c in example["conversations"]:
            if not isinstance(c, dict):
                continue
            role = _normalize_role(c.get("from", "user"))
            content = c.get("value", "")
            if content is None:
                content = ""
            msgs.append({"role": role, "content": str(content)})
        return msgs if msgs else None

    # Alpaca-style single-turn
    instr = example.get("instruction")
    output = example.get("output")
    if instr is not None and output is not None:
        user_text = instr
        if example.get("input"):
            # typical Alpaca concatenation
            user_text = f"{instr}\n\n{example['input']}"
        return [
            {"role": "user", "content": str(user_text)},
            {"role": "assistant", "content": str(output)},
        ]

    return None


def _standardize_conversations(ds: Dataset,tokenizer) -> Dataset:
    """
    Maps the 'conversations' column from the ShareGPT format to the Hugging Face 'messages' format,
    then apply the chat template to train with trl.
    """
    if not tokenizer.apply_chat_template:
        raise ValueError("Tokenizer does not support apply_chat_template. Make sure you're loading an Instruct version of the model.")
    # for text-based trl.
    #return ds.map(lambda e: apply_chat_template({"messages": e['conversations']},tokenizer=tokenizer,num_proc=1),num_proc=1,remove_columns=['conversations'])
    # for SFT on conversations, use the following:
    ds = ds.map(lambda e: {"messages": e['conversations']},num_proc=1,remove_columns=['conversations'])
    ds = ds.filter(lambda e: any(m.get("role") == "assistant" for m in e["messages"]))
    return ds


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    name: str
    checkpoint_dir: str
    data: str
    bit4: bool = False
    eval_data: str = "../data/eval/eval_1_multi_step.json"
    r: int = 16
    base_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    wandb_project: str = "SLM-distill"
    wandb_run_name: str = "spine_lora"
    wandb_tag: str = "spine"
    epochs: int = 2
    val_frac: float = 0.1
    # LoRA
    lora_alpha: int = 16
    lora_dropout: float = 0.2
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    # Trainer args
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    report_to: str = "wandb"
    learning_rate: float = 2e-4
    warmup_steps: int = 5
    weight_decay: float = 0.05
    debug: bool = False
    max_seq_length: int = 2048


# ----------------------------
# Training
# ----------------------------
def train_model(config: TrainConfig):
    # mirror original SAVE_NAME logic
    save_name = f"{config.name}_r{config.r}" + ("_4bit" if config.bit4 else "")

    # Quantization / dtype
    bnb_config = None
    if config.bit4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if _bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Model & tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype="auto",
        quantization_config=bnb_config,  # None if not 4-bit
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    _ensure_pad_tokens(tokenizer, model)
    tokenizer.padding_side = "right"

    # Load & optionally downsample data
    full_dataset = load_dataset("json", data_files=[config.data], split="train")
    if config.debug:
        full_dataset = full_dataset.select(range(min(100, len(full_dataset))))

    # Convert to `messages` so TRL can auto-apply chat template
    full_dataset = _standardize_conversations(full_dataset,tokenizer)

    # Train/val split
    if config.val_frac and config.val_frac > 0.0:
        dataset_size = len(full_dataset)
        val_size = int(dataset_size * config.val_frac)
        train_size = dataset_size - val_size
        split = full_dataset.train_test_split(
            test_size=val_size,
            train_size=train_size,
            seed=3407,
            shuffle=True,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Dataset split: {len(train_dataset)} train / {len(eval_dataset)} eval")
    else:
        train_dataset = full_dataset
        eval_dataset = None
        print(f"Using all {len(full_dataset)} samples for training (no validation).")

    # LoRA config (PEFT)
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=config.target_modules,
        task_type="CAUSAL_LM",
    )

    # Optimizer choice: use 8-bit AdamW when bitsandbytes is active; else fused AdamW
    optim = "adamw_bnb_8bit" if config.bit4 else "adamw_torch_fused"

    # probe = [
    #     {"role":"system","content":"You are helpful."},
    #     {"role":"user","content":"Say hi."},
    #     {"role":"assistant","content":"Hello!"},
    # ]   
    # enc = tokenizer.apply_chat_template(
    #     probe, tokenize=True, return_assistant_tokens_mask=True,return_dict=True
    # )
    #assert any(enc["assistant_masks"]), "Template still not producing assistant masks."

    # SFT trainer configuration
    sft_args = SFTConfig(
        dataset_num_proc=1,
        packing=False,
        max_length=config.max_seq_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        lr_scheduler_type="linear",
        logging_steps=1,
        # precision
        fp16=not _bf16_supported(),
        bf16=_bf16_supported(),
        # misc
        seed=3407,
        output_dir=str(os.path.join("outputs", save_name)),
        report_to=config.report_to,
        run_name=config.wandb_run_name,
        optim=optim,
        # key behavior parity with unsloth.train_on_responses_only
        # Temporarily disabled because qwen doesn't support it.
        #assistant_only_loss=True,  # train only on assistant tokens
        remove_unused_columns=False,  
        save_safetensors=True,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        peft_config=lora_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_args,
    )

    

    # Start training
    trainer.train()

    # Optionally: save adapter & tokenizer (adapters are what you’ll push/share)
    trainer.save_model()  # saves PEFT adapter if peft_config was used
    tokenizer.save_pretrained(sft_args.output_dir)

    return trainer


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = HfArgumentParser(TrainConfig)
    (cfg,) = parser.parse_args_into_dataclasses()

    # Keep your W&B naming convention
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    os.environ["WANDB_RUN_GROUP"] = cfg.wandb_tag

    print(cfg)
    train_model(cfg)
