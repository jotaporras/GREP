import os
import re
import sys

from prism.data import data
from prism.eval import callbacks
from prism.eval import evaluate
from prism.eval import loading
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
        # graph encoder and gate/projection so they actually train.
        if gnn_config.get("architecture") == "augmented_graph_gt":
            for p in self.model.gt_model.parameters():
                p.requires_grad = True
            for p in self.model.injection.parameters():
                p.requires_grad = True
        else:
            for p in self.model.pe_model.parameters():
                p.requires_grad = True
            for p in self.model.pe_proj.parameters():
                p.requires_grad = True
            self.model.pe_gain.requires_grad = True

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
        if self.gnn_config.get("architecture") == "augmented_graph_gt":
            # M9: save the Graph Transformer (R-PEARL inside) and the M7 gate.
            torch.save({
                'gt_model': self.model.gt_model.state_dict(),
                'injection': self.model.injection.state_dict(),
            }, os.path.join(output_dir, "gnn_weights.pt"))
            torch.save({
                'rpearl': self.model.gt_model.pe_model.state_dict(),
            }, os.path.join(output_dir, "rpearl_weights.pt"))
        elif self.gnn_config.get("architecture") == "rpearl_gt_llm":
            # Full GT: save the whole GraphTransformer (includes R-PEARL inside) + projection head.
            torch.save({
                'gt_model': self.model.pe_model.state_dict(),
                'pe_proj': self.model.pe_proj.state_dict(),
                'pe_gain': self.model.pe_gain.data,
            }, os.path.join(output_dir, "gnn_weights.pt"))
            # Also save the inner R-PEARL separately for analysis / reuse.
            torch.save({
                'rpearl': self.model.pe_model.pe_model.state_dict(),
            }, os.path.join(output_dir, "rpearl_weights.pt"))
        else:
            torch.save({
                'pe_model': self.model.pe_model.state_dict(),
                'pe_proj': self.model.pe_proj.state_dict(),
                'pe_gain': self.model.pe_gain.data,
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
    # Optional pre-split validation file (same schema as `data`). When set,
    # `val_frac` is ignored and this file is loaded as the eval dataset.
    val_data: Optional[str] = None
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
    architecture: str = "rpearl_llm"  # "rpearl_llm", "rpearl_gt_llm", "augmented_graph_gt", or "llm"
    # GT-specific params (used when architecture == "rpearl_gt_llm" / "augmented_graph_gt")
    gt_num_layers: int = 3
    gt_heads: int = 8
    eps: float = 1e-8

    def __post_init__(self):
        self.eps = float(self.eps)
    k_gt: int = 3
    text_edge_list: str = "present"   # "present" or "none"
    # Augmented-graph params (used when architecture == "augmented_graph_gt")
    cycle_directed: bool = True
    cycle_weight: float = 1.0
    affinity_kernel: str = "gaussian"
    sigma_mode: str = "median"
    keep_raw_distance_feature: bool = True
    crosslink_weight: float = 1.0
    crosslink_mention_to_node: bool = True
    crosslink_mention_clique: bool = True
    # R-PEARL sampling
    probe_distribution: str = "gaussian"
    m_test: int = 128
    fixed_seed_mode: bool = False
    fixed_seed_value: int = 0
    # Gated injection
    injection_mode: str = "interpolate"
    gate_init: float = 0.0
    gate_per_dim: bool = False
    disable_rope: bool = True
    # Prompt / debug-viz switches (carried for config fidelity)
    n_icl_examples: int = 2
    log_fiedler: bool = True
    log_scene_mass: bool = True
    enable_visualizer: bool = False
    device: int = 0                   # GPU index to pin the model to; -1 = let device_map="auto" decide
    overwrite_ok: bool = False
    # Optional override for the checkpoint subdirectory name.
    # Default (None): auto-generated as "{name}_{architecture}_{model_slug}_r{r}[_4bit]_{wandb_run_id}"
    # Override: "{save_name}_{wandb_run_id}" — the run ID is always appended.
    save_name: str = None
    # Optional path (file or directory of {graph, tasks} JSONs) to run a
    # post-training cross-eval on. When set, after training finishes the
    # in-memory model is evaluated over the resolved files and per-graph
    # results are written to <output_dir>/eval_logs/cross_eval/<graph>.json.
    # Replaces the previous Stage 3 sbatch step that invoked
    # scripts/eval_checkpoint_on_graphs.py against the same checkpoint.
    post_train_eval_graphs: Optional[str] = None
    # Whether to enable SPINE in-context-learning examples during both the
    # train-time eval callback and the optional post-train cross-eval.
    # Argparse-layer default; library functions never default this on the
    # caller's behalf. Historical behavior: True (None used to cascade to
    # SPINE's default of True).
    eval_use_icl: bool = True


def _load_eval_samples(eval_data_path: str) -> list:
    """Load a single-graph eval file as a list of `EvalSample`s.

    Used by the train-time periodic `EvalCallback`. For multi-graph
    post-train eval see `_run_post_train_cross_eval`.
    """
    with open(eval_data_path) as f:
        payload = json.load(f)
    graph_name = os.path.splitext(os.path.basename(eval_data_path))[0]
    return evaluate.construct_eval_samples_from_dict(
        payload["graph"], payload["tasks"], graph_name=graph_name,
    )


def _run_post_train_cross_eval(model, tokenizer, config: "TrainConfig", output_dir: str) -> None:
    """Run cross-eval on the in-memory model after training and write per-graph JSONs.

    Disk I/O happens in `loading.load_samples_by_graph`; this function is
    pure orchestration: load → eval → write. Output shape matches the old
    `eval_checkpoint_on_graphs.py` (consumed by eval_viewer.html and the
    judge-eval skill).
    """
    target = config.post_train_eval_graphs
    if target is None:
        return

    samples_by_graph, graph_file_by_name = loading.load_samples_by_graph(target)

    is_gnn = config.architecture in ("rpearl_llm", "rpearl_gt_llm", "augmented_graph_gt")
    architecture = "graph-augmented" if is_gnn else "llm"
    out_dir = os.path.join(output_dir, "eval_logs", "cross_eval")
    os.makedirs(out_dir, exist_ok=True)

    model.eval()
    print(f"\n[post-train eval] {len(samples_by_graph)} graph file(s) -> {out_dir}")
    results = evaluate.eval_model_multiple_graphs(
        model,
        tokenizer,
        samples_by_graph,
        include_edge_list=(config.text_edge_list == "present"),
        use_icl=config.eval_use_icl,
        permutation=None,
        on_graph_done=None,
    )

    for name, result in results.items():
        _write_cross_eval_json(
            out_dir, name, result,
            checkpoint=output_dir,
            graph_file=graph_file_by_name[name],
            architecture=architecture,
            text_edge_list=config.text_edge_list,
        )

    evaluate.print_summary_table(list(results.values()))


def _write_cross_eval_json(
    out_dir: str,
    name: str,
    result: evaluate.GraphEvalResultSummary,
    *,
    checkpoint: str,
    graph_file: str,
    architecture: str,
    text_edge_list: str,
) -> None:
    log_data = {
        "checkpoint": checkpoint,
        "graph_file": graph_file,
        "architecture": architecture,
        "text_edge_list": text_edge_list,
        "accuracy": result.accuracy,
        "num_samples": result.num_total,
        "num_correct": result.num_correct,
        "path_metrics": result.path_metrics,
        "samples": result.samples,
    }
    out_file = os.path.join(out_dir, f"{name}.json")
    with open(out_file, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"  {name}: {result.num_correct}/{result.num_total} ({result.accuracy:.1%}) -> {out_file}")


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
    elif config.architecture == "augmented_graph_gt":
        # Augmented-graph pipeline (M4-M8): one graph (cycle + scene + cross-links)
        # per sequence; R-PEARL + GT refine it; the gate injects Y[V_Tx] into the
        # RoPE-disabled LLM. Reuses SpineDataCollator (scene graphs + injection maps);
        # the augmented graph is assembled inside the model forward.
        gt_model = gt_module.GraphTransformer(
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
            probe_distribution=config.probe_distribution,
            m_test=config.m_test,
            fixed_seed_mode=config.fixed_seed_mode,
            fixed_seed_value=config.fixed_seed_value,
            # Token embeddings are fused (M6) -> the X-carrying GT path is not
            # spectrally normalized (only the PE-side R-PEARL projection is).
            spectral_norm_linears=False,
        )
        model = gnn_llm.AugmentedGraphLLM(
            llm, gt_model, d_model=config.d_model,
            gate_init=config.gate_init,
            gate_per_dim=config.gate_per_dim,
            injection_mode=config.injection_mode,
            disable_llm_rope=config.disable_rope,
            cycle_weight=config.cycle_weight,
            cycle_directed=config.cycle_directed,
            crosslink_weight=config.crosslink_weight,
            crosslink_mention_to_node=config.crosslink_mention_to_node,
            crosslink_mention_clique=config.crosslink_mention_clique,
        )
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
        raise ValueError(f"Unknown architecture: {config.architecture!r}. Choose 'rpearl_llm', 'rpearl_gt_llm', 'augmented_graph_gt', or 'llm'.")

    # Load & optionally downsample data
    full_dataset = datasets.load_dataset("json", data_files=[config.data], split="train")
    if config.debug:
        full_dataset = full_dataset.select(range(round(len(full_dataset) * config.dataset_proportion)))

    full_dataset = data.preprocess_dataset(
        full_dataset, tokenizer,
        architecture=config.architecture,
        text_edge_list=config.text_edge_list,
    )

    # Train/val split: prefer an explicit pre-split val file when provided.
    if config.val_data:
        train_dataset = full_dataset
        eval_dataset = datasets.load_dataset("json", data_files=[config.val_data], split="train")
        if config.debug:
            eval_dataset = eval_dataset.select(range(round(len(eval_dataset) * config.dataset_proportion)))
        eval_dataset = data.preprocess_dataset(
            eval_dataset, tokenizer,
            architecture=config.architecture,
            text_edge_list=config.text_edge_list,
        )
        print(f"Using pre-split val file: {len(train_dataset)} train / {len(eval_dataset)} eval")
    elif config.val_frac and config.val_frac > 0.0:
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

    if config.architecture in ("rpearl_llm", "rpearl_gt_llm", "augmented_graph_gt"):
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
               if config.architecture in ("rpearl_gt_llm", "augmented_graph_gt") else {}),
            # Augmented-graph rebuild params (read back by loaders for eval).
            **({"k_gt": config.k_gt, "gt_num_layers": config.gt_num_layers,
                "gt_heads": config.gt_heads,
                "probe_distribution": config.probe_distribution, "m_test": config.m_test,
                "fixed_seed_mode": config.fixed_seed_mode, "fixed_seed_value": config.fixed_seed_value,
                "injection_mode": config.injection_mode, "gate_init": config.gate_init,
                "gate_per_dim": config.gate_per_dim, "disable_rope": config.disable_rope,
                "cycle_weight": config.cycle_weight, "cycle_directed": config.cycle_directed,
                "crosslink_weight": config.crosslink_weight,
                "crosslink_mention_to_node": config.crosslink_mention_to_node,
                "crosslink_mention_clique": config.crosslink_mention_clique}
               if config.architecture == "augmented_graph_gt" else {}),
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
    trainer.add_callback(callbacks.EvalCallback(
        eval_samples,
        tokenizer=tokenizer,
        use_icl=config.eval_use_icl,
        include_edge_list=(config.text_edge_list == "present"),
        eval_epoch_interval=1.0,
    ))

    if config.architecture in ("rpearl_llm", "rpearl_gt_llm"):
        trainer.add_callback(callbacks.GradientDebugCallback())
    elif config.architecture == "augmented_graph_gt":
        # M11: augmented-graph diagnostics (Fiedler, scene-mass, gate, contrib-ratio).
        # M12: when enable_visualizer is set, this callback also renders the
        # augmented-graph + spectral-clustering artifacts once (first eval-time log).
        trainer.add_callback(callbacks.AugGraphDebugCallback(
            enable_visualizer=config.enable_visualizer,
            visualizer_dir=os.path.join(output_dir, "visuals"),
        ))

    # Start training
    trainer.train()

    # Save model artifacts
    trainer.save_model()
    tokenizer.save_pretrained(sft_args.output_dir)

    # Optional inline post-training cross-eval (replaces the old Stage 3
    # sbatch invocation of scripts/eval_checkpoint_on_graphs.py).
    if config.post_train_eval_graphs:
        _run_post_train_cross_eval(
            trainer.model, tokenizer, config, sft_args.output_dir,
        )

    return trainer


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = HfArgumentParser(TrainConfig)
    if len(sys.argv) >= 2 and sys.argv[1].endswith((".yaml", ".yml")):
        import yaml as _yaml
        with open(sys.argv[1]) as f:
            cfg_dict = _yaml.safe_load(f) or {}
        # Overlay --key value pairs from sys.argv[2:] onto the yaml dict so
        # callers can override individual fields without writing a new yaml.
        i = 2
        while i < len(sys.argv):
            arg = sys.argv[i]
            if not arg.startswith("--"):
                raise SystemExit(f"Expected --key value after yaml path, got: {arg!r}")
            key = arg[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                cfg_dict[k] = _yaml.safe_load(v)
                i += 1
            else:
                if i + 1 >= len(sys.argv):
                    raise SystemExit(f"Missing value for override --{key}")
                cfg_dict[key] = _yaml.safe_load(sys.argv[i + 1])
                i += 2
        (cfg,) = parser.parse_dict(cfg_dict)
        config_file = sys.argv[1]
    else:
        (cfg,) = parser.parse_args_into_dataclasses()
        config_file = None

    print(cfg)
    train_model(cfg, config_file=config_file)
