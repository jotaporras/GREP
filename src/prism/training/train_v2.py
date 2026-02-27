# hf_sft_lora.py
import os
import sys
import json
from dataclasses import asdict, dataclass, field
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

from prism.data import data_col
from prism.eval import callbacks
from prism.eval import run_eval
from prism.models import gnn_llm
from prism.models import r_pearl

# Env first (so W&B picks these up as soon as possible)
os.environ.setdefault("WANDB_PROJECT", "SLM-distill")
# Match your original env var usage
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")  # harmless here

import wandb
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


def _fp16_supported() -> bool:
    """Conservative check for fp16 support."""
    if _bf16_supported():
        return False
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _model_short_name(base_model: str) -> str:
    """Extract a short filesystem-safe slug from a HuggingFace model ID.

    Examples:
        meta-llama/Llama-3.1-8B-Instruct → llama-3.1-8b
        Qwen/Qwen2.5-0.5B-Instruct       → qwen2.5-0.5b
    """
    import re
    name = base_model.split("/")[-1]          # drop org prefix
    name = re.sub(r"-[Ii]nstruct$", "", name) # drop -Instruct suffix
    name = name.lower()
    name = re.sub(r"-+", "-", name)           # collapse consecutive hyphens
    return name


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


class GraphSFTTrainer(SFTTrainer):
    def __init__(self, *args, gnn_config: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnn_config = gnn_config

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "gnn_config.json"), "w") as f:
            json.dump(self.gnn_config, f, indent=2)
        torch.save({
            'pe_model': self.model.pe_model.state_dict(),
            'pe_proj': self.model.pe_proj.state_dict(),
        }, os.path.join(output_dir, "gnn_weights.pt"))
        if any(p.requires_grad for p in self.model.llm.parameters()):
            super().save_model(output_dir, _internal_call)


# ----------------------------
# Config
# ----------------------------
@dataclass
class TrainConfig:
    name: str
    checkpoint_dir: str
    data: str
    bit4: bool = False
    eval_data: str = "data/eval/eval_1_multi_step.json"
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
    dataset_proportion: float = 0.1
    # Model args.
    pe_hidden_channels: int = 256
    pe_num_layers: int = 3
    d_model: int = 3072
    num_samples: int = 40
    dropout: float = 0.1
    k: int = 3
    use_layer_norm: bool = True
    freeze_llm: bool = False
    architecture: str = "rpearl_llm"  # "rpearl_llm" or "llm"
    text_edge_list: str = "present"   # "present" or "none"
    overwrite_ok: bool = False
    # Optional override for the checkpoint subdirectory name.
    # Default (None): auto-generated as "{name}_{architecture}_{model_slug}_r{r}[_4bit]_{wandb_run_id}"
    # Override: "{save_name}_{wandb_run_id}" — the run ID is always appended.
    save_name: str = None


# ----------------------------
# Training
# ----------------------------
def train_model(config: TrainConfig, config_file: str = None):
    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_RUN_GROUP"] = config.wandb_tag
    os.environ["WANDB_TAGS"] = config.wandb_tag

    wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        tags=[config.wandb_tag],
        group=config.wandb_tag,
    )
    wandb_run_id = wandb.run.id

    # Checkpoint subdirectory name.
    # Format: "{name}_{architecture}_{model_slug}_r{r}[_4bit]_{wandb_run_id}"
    # Override with --save_name to use "{save_name}_{wandb_run_id}" instead.
    model_slug = _model_short_name(config.base_model)
    if config.save_name is not None:
        save_name = f"{config.save_name}_{wandb_run_id}"
    else:
        save_name = f"{config.name}_{config.architecture}_{model_slug}_r{config.r}" + ("_4bit" if config.bit4 else "") + f"_{wandb_run_id}"

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
    llm = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype="auto",
        quantization_config=bnb_config,  # None if not 4-bit
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    _ensure_pad_tokens(tokenizer, llm)
    tokenizer.padding_side = "right"

    if config.architecture == "rpearl_llm":
        # R-PEARL model, graph-augmented model & data collator.
        r_pearl_model = r_pearl.RandomGNNPositionalEncodings(
            pe_hidden_channels=config.pe_hidden_channels,
            pe_num_layers=config.pe_num_layers,
            d_model=config.d_model,
            num_samples=config.num_samples,
            dropout=config.dropout,
            k=config.k,
            use_layer_norm=config.use_layer_norm
        )
        model = gnn_llm.GraphAugmentedLLM(llm, r_pearl_model, tokenizer, pe_dim=config.d_model)
        collator = data_col.DataCollatorForGraphAugmentedLLM(
            tokenizer, mlm=False, text_edge_list=config.text_edge_list
        )

        # Freeze the whole llm.
        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "llm":
        # Pure LLM baseline — scene graph text stays in the prompt as-is.
        # No custom collator: SFTTrainer's built-in collator handles
        # tokenization from the `messages` column and padding.
        model = llm
        collator = None
    else:
        raise ValueError(f"Unknown architecture: {config.architecture!r}. Choose 'rpearl_llm' or 'llm'.")

    # Load & optionally downsample data
    full_dataset = load_dataset("json", data_files=[config.data], split="train")
    if config.debug:
        full_dataset = full_dataset.select(range(round(len(full_dataset) * config.dataset_proportion)))

    # Define data maps.
    def _add_messages(example):
        example["messages"] = example["conversations"]
        return example

    def _tokenize_with_conversations(example):
        tokenized = tokenizer.apply_chat_template(
            example["messages"], tokenize=True, return_dict=True
        )
        tokenized["conversations"] = example["conversations"]
        tokenized["messages"] = example["messages"]
        return tokenized

    # Configure data.
    full_dataset = full_dataset.map(_add_messages)

    # Strip edge lists from the user message text when requested.
    # For rpearl_llm this is handled later inside the collator; for the llm
    # baseline the collator is absent so we must do it here on the raw text.
    if config.text_edge_list == "none" and config.architecture == "llm":
        def _strip_edges(example):
            example["messages"] = [
                {**m, "content": data_col.remove_edge_list(m["content"])} if m["role"] == "user" else m
                for m in example["messages"]
            ]
            return example
        full_dataset = full_dataset.map(_strip_edges)

    full_dataset = full_dataset.map(_tokenize_with_conversations)

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

    output_dir = str(os.path.join(config.checkpoint_dir, save_name))

    if os.path.isdir(output_dir) and os.listdir(output_dir) and not config.overwrite_ok:
        raise RuntimeError(
            f"Checkpoint directory already exists and is non-empty: {output_dir}\n"
            f"Set overwrite_ok: true in your config to allow overwriting, "
            f"or delete the directory manually."
        )

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
        logging_steps=15,
        # precision
        fp16=_fp16_supported(),
        bf16=_bf16_supported(),
        # misc
        seed=3407,
        output_dir=output_dir,
        report_to=config.report_to,
        run_name=config.wandb_run_name,
        optim=optim,
        # key behavior parity with unsloth.train_on_responses_only
        # Temporarily disabled because qwen doesn't support it.
        #assistant_only_loss=True,  # train only on assistant tokens
        remove_unused_columns=False,
        # Checkpointing
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        # Validation
        eval_strategy="steps",
        eval_steps=15,
        do_eval=True,
    )

    if config.architecture == "rpearl_llm":
        gnn_config = {
            "base_model": config.base_model,
            "pe_hidden_channels": config.pe_hidden_channels,
            "pe_num_layers": config.pe_num_layers,
            "d_model": config.d_model,
            "num_samples": config.num_samples,
            "dropout": config.dropout,
            "k": config.k,
            "use_layer_norm": config.use_layer_norm,
        }
        trainer = GraphSFTTrainer(
            model=model,
            data_collator=collator,
            processing_class=tokenizer,
            peft_config=lora_config if not config.freeze_llm else None,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_args,
            gnn_config=gnn_config,
        )
    else:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            peft_config=lora_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_args,
        )

    # Log all training config parameters to wandb
    if wandb.run is not None:
        wandb.config.update(asdict(config), allow_val_change=True)
        if config_file is not None:
            with open(config_file) as f:
                wandb.config.update({"_config_yaml": f.read()}, allow_val_change=True)

    # Load eval samples before training
    with open(config.eval_data) as f:
        data = json.load(f)
        tasks = data["tasks"]
        graph_data = data["graph"]

    eval_samples = []
    for entry in tasks:
        eval_samples.append(
            run_eval.EvalSample(
                task=entry["task"],
                answer=entry["answer"],
                graph=graph_data,
                init_node=entry["init_node"],
            )
        )

    trainer.add_callback(callbacks.EvalCallback(eval_samples, tokenizer=tokenizer, eval_epoch_interval=1.0))

    if config.architecture == "rpearl_llm":
        trainer.add_callback(callbacks.GradientDebugCallback())

    # Start training
    trainer.train()

    # Save model artifacts
    if config.architecture == "rpearl_llm":
        if config.freeze_llm:
            torch.save({
                'pe_model': model.pe_model.state_dict(),
                'pe_proj': model.pe_proj.state_dict(),
            }, os.path.join(sft_args.output_dir, "gnn_weights.pt"))
        else:
            trainer.save_model()  # saves PEFT adapter if peft_config was used
            tokenizer.save_pretrained(sft_args.output_dir)
    else:
        trainer.save_model()
        tokenizer.save_pretrained(sft_args.output_dir)

    return trainer


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = HfArgumentParser(TrainConfig)
    if len(sys.argv) == 2 and sys.argv[1].endswith((".yaml", ".yml")):
        (cfg,) = parser.parse_yaml_file(sys.argv[1])
        config_file = sys.argv[1]
    else:
        (cfg,) = parser.parse_args_into_dataclasses()
        config_file = None

    print(cfg)
    train_model(cfg, config_file=config_file)
