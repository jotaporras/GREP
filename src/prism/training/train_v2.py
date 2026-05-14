import os
import re
import sys

from prism.data import data
from prism.eval import callbacks
from prism.eval import run_eval
from prism.models import gnn_llm
from prism.models import r_pearl as r_pearl_module
from prism.models import gt as gt_module

import json
from dataclasses import asdict, dataclass, field
from typing import List, Dict, Any, Optional, no_type_check

from dotenv import load_dotenv
load_dotenv()

import wandb
import torch
import datasets
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



class GraphSFTTrainer(SFTTrainer):
    def __init__(self, *args, gnn_config: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self.gnn_config = gnn_config
        # PEFT freezes all non-LoRA parameters. Re-enable gradients for the
        # graph encoder and projection head so they actually train.
        for p in self.model.pe_model.parameters():
            p.requires_grad = True
        for p in self.model.pe_proj.parameters():
            p.requires_grad = True

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch, **kwargs)
        # Gradients exist now (backward already ran inside super().training_step).
        # Capture norms before the training loop calls zero_grad().
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, callbacks.GradientDebugCallback):
                cb._capture_grad_norms(model)
                break
        return loss

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "gnn_config.json"), "w") as f:
            json.dump(self.gnn_config, f, indent=2)
        if self.gnn_config.get("architecture") == "rpearl_gt_llm":
            # Full GT: save the whole GraphTransformer (includes R-PEARL inside) + projection head.
            torch.save({
                'gt_model': self.model.pe_model.state_dict(),
                'pe_proj': self.model.pe_proj.state_dict(),
            }, os.path.join(output_dir, "gnn_weights.pt"))
            # Also save the inner R-PEARL separately for analysis / reuse.
            torch.save({
                'rpearl': self.model.pe_model.pe_model.state_dict(),
            }, os.path.join(output_dir, "rpearl_weights.pt"))
        else:
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
    max_steps: int = -1  # If > 0, overrides epochs and switches eval/save to step-based (dev use)
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
    dataloader_num_workers: int = 4
    report_to: str = "wandb"
    learning_rate: float = 2e-4
    warmup_steps: int = 5 # TODO consider warmup_steps: float= 0.03
    weight_decay: float = 0.05
    debug: bool = False
    max_seq_length: int = 2048
    dataset_num_proc: int = 8
    dataset_proportion: float = 0.1
    # Model args.
    pe_hidden_channels: int = 256
    pe_num_layers: int = 3
    d_model: int = 3072
    num_samples: int = 40
    dropout: float = 0.1
    k_pe: int = 3
    use_layer_norm: bool = True
    freeze_llm: bool = False
    architecture: str = "rpearl_llm"  # "rpearl_llm", "rpearl_gt_llm", or "llm"
    # GT-specific params (used when architecture == "rpearl_gt_llm")
    gt_num_layers: int = 3
    gt_heads: int = 8
    eps: float = 1e-8

    def __post_init__(self):
        self.eps = float(self.eps)
    k_gt: int = 3
    text_edge_list: str = "present"   # "present" or "none"
    device: int = 0                   # GPU index to pin the model to; -1 = let device_map="auto" decide
    overwrite_ok: bool = False
    # Optional override for the checkpoint subdirectory name.
    # Default (None): auto-generated as "{name}_{architecture}_{model_slug}_r{r}[_4bit]_{wandb_run_id}"
    # Override: "{save_name}_{wandb_run_id}" — the run ID is always appended.
    save_name: str = None


def _load_eval_samples(eval_data_path: str) -> list:
    with open(eval_data_path) as f:
        data = json.load(f)
    tasks = data["tasks"]
    graph_data = data["graph"]
    return [
        run_eval.EvalSample(
            task=entry["task"],
            answer=entry["answer"],
            graph=graph_data,
            init_node=entry["init_node"],
        )
        for entry in tasks
    ]


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
    device_map = {"": 0} if config.device >= 0 else "auto"
    llm = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype="auto",
        quantization_config=bnb_config,  # None if not 4-bit
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    _ensure_pad_tokens(tokenizer, llm)
    tokenizer.padding_side = "right" # ty: ignore[invalid-assignment]

    if config.architecture == "rpearl_llm":
        # R-PEARL only: GCN positional encodings, no GT attention blocks.
        pe_model = r_pearl_module.RandomGNNPositionalEncodings(
            pe_hidden_channels=config.pe_hidden_channels,
            pe_num_layers=config.pe_num_layers,
            d_model=config.d_model,
            num_samples=config.num_samples,
            dropout=config.dropout,
            k=config.k_pe,
            eps=config.eps,
            use_layer_norm=config.use_layer_norm,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=config.d_model, eps=config.eps)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "rpearl_gt_llm":
        # Full Graph Transformer: R-PEARL inside Sparse Attention blocks.
        pe_model = gt_module.GraphTransformer(
            num_layers=config.gt_num_layers,
            pe_hidden_channels=config.pe_hidden_channels,
            pe_num_layers=config.pe_num_layers,
            d_model=config.d_model,
            heads=config.gt_heads,
            num_samples=config.num_samples,
            dropout=config.dropout,
            k_pe=config.k_pe,
            k_gt=config.k_gt,
            eps=config.eps,
            use_layer_norm=config.use_layer_norm,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=config.d_model, eps=config.eps)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "llm":
        # Pure LLM baseline — scene graph text stays in the prompt as-is.
        # No custom collator: SFTTrainer's built-in collator handles
        # tokenization from the `messages` column and padding.
        model = llm
        collator = None
    else:
        raise ValueError(f"Unknown architecture: {config.architecture!r}. Choose 'rpearl_llm', 'rpearl_gt_llm', or 'llm'.")

    # Load & optionally downsample data
    full_dataset = datasets.load_dataset("json", data_files=[config.data], split="train")
    if config.debug:
        full_dataset = full_dataset.select(range(round(len(full_dataset) * config.dataset_proportion)))

    full_dataset = data.preprocess_dataset(
        full_dataset, tokenizer,
        architecture=config.architecture,
        text_edge_list=config.text_edge_list,
    )

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

    output_dir = str(os.path.join(config.checkpoint_dir, save_name))

    if os.path.isdir(output_dir) and os.listdir(output_dir) and not config.overwrite_ok:
        raise RuntimeError(
            f"Checkpoint directory already exists and is non-empty: {output_dir}\n"
            f"Set overwrite_ok: true in your config to allow overwriting, "
            f"or delete the directory manually."
        )

    # SFT trainer configuration
    sft_args = SFTConfig(
        dataset_num_proc=config.dataset_num_proc,
        dataloader_num_workers=config.dataloader_num_workers,
        packing=False, # packing combines multiple examples into a single input_id. Keep disabled to avoid graph contamination.
        max_length=config.max_seq_length,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.epochs,
        max_steps=config.max_steps,
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
        # Checkpointing / Validation: step-based when max_steps is set (dev), else epoch-based.
        save_strategy="steps" if config.max_steps > 0 else "epoch",
        save_steps=max(1, config.max_steps // 2) if config.max_steps > 0 else 500,
        save_total_limit=3,
        eval_strategy="steps" if config.max_steps > 0 else "epoch",
        eval_steps=max(1, config.max_steps // 2) if config.max_steps > 0 else 0.5,
        do_eval=True,
    )

    if config.architecture in ("rpearl_llm", "rpearl_gt_llm"):
        gnn_config = {
            "architecture": config.architecture,
            "base_model": config.base_model,
            "pe_hidden_channels": config.pe_hidden_channels,
            "pe_num_layers": config.pe_num_layers,
            "d_model": config.d_model,
            "num_samples": config.num_samples,
            "dropout": config.dropout,
            "k_pe": config.k_pe,
            "use_layer_norm": config.use_layer_norm,
            "text_edge_list": config.text_edge_list,
            "eps": config.eps,
            **({"k_gt": config.k_gt, "gt_num_layers": config.gt_num_layers,
                "gt_heads": config.gt_heads}
               if config.architecture == "rpearl_gt_llm" else {}),
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

    eval_samples = _load_eval_samples(config.eval_data)
    trainer.add_callback(callbacks.EvalCallback(eval_samples, tokenizer=tokenizer, eval_epoch_interval=1.0, text_edge_list=config.text_edge_list))

    if config.architecture in ("rpearl_llm", "rpearl_gt_llm"):
        trainer.add_callback(callbacks.GradientDebugCallback())

    # Start training
    trainer.train()

    # Save model artifacts
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
