import os
import re
import sys
import warnings

from prism.data import data
from prism.eval import callbacks
from prism.eval import evaluate
from prism.eval import loading
from prism.models import gnn_llm
from prism.models import r_pearl as r_pearl_module
from prism.models import gt as gt_module

import json
from dataclasses import asdict, dataclass, field
from typing import List, Dict, Any, Optional

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
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer


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




# Maps loss_target values to their precomputed per-example index column. "all" is
# absent (full-sequence). Add a new target by precomputing a column and registering it.
_LOSS_TARGET_COLUMN = {"responses": "assistant_idx", "edge_list": "edge_list_idx"}


class LossTargetMixin:
    """Restricts the next-token loss to a configurable token span (``loss_target``).

    ``all`` (default) = full sequence; ``responses`` = assistant turns;
    ``edge_list`` = edge bullets. Supervised positions come from per-example index
    columns precomputed in ``preprocess_dataset``, mapped via ``_LOSS_TARGET_COLUMN``.

    Masks ``inputs['labels']`` then defers to ``super().compute_loss`` (cooperative,
    MRO-safe). Mix in BEFORE :class:`GraphTokenAccuracyMixin` so masking happens first.
    """

    def _set_loss_target(self, loss_target: str):
        """Configure the supervised-token span. Call after ``super().__init__``.

        For masked targets (responses / edge_list), disables the loss-kwargs path so
        the LM loss uses reduction="mean" over only the supervised tokens, not HF's
        full-batch normalization (which would under-normalize the masked span).
        """
        self.loss_target = loss_target
        if loss_target != "all":
            self.model_accepts_loss_kwargs = False

    def _mask_labels_to_positions(self, inputs, idx_lists, target):
        """Mask ``inputs['labels']`` to -100 outside the per-example supervised positions.

        ``idx_lists`` is a per-example list of token positions (e.g. assistant_idx /
        edge_list_idx). If a batch ends up with NO supervised tokens, fall back to its
        full-sequence labels with a warning (better than a NaN all-masked loss).
        """
        labels = inputs["labels"]
        keep = torch.zeros_like(labels, dtype=torch.bool)
        for b, idx in enumerate(idx_lists):
            if not idx or b >= labels.shape[0]:
                continue
            pos = torch.as_tensor([p for p in idx if 0 <= p < labels.shape[1]],
                                  device=labels.device, dtype=torch.long)
            if pos.numel():
                keep[b, pos] = True
        if keep.any():
            inputs["labels"] = labels.masked_fill(~keep, -100)
        else:
            warnings.warn(
                f"loss_target={target!r} but no supervised tokens located in this "
                "batch; falling back to full-sequence loss for it.")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Pop the mask columns so they never reach the model forward, then restrict
        # the loss to the configured span ('all' / absent column => no masking).
        mask_columns = {name: inputs.pop(name, None)
                        for name in set(_LOSS_TARGET_COLUMN.values())}
        target = getattr(self, "loss_target", "all")
        col = _LOSS_TARGET_COLUMN.get(target)
        if col is not None and mask_columns.get(col) is not None \
                and inputs.get("labels") is not None:
            self._mask_labels_to_positions(inputs, mask_columns[col], target)
        return super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)


class GraphTokenAccuracyMixin:
    """This mixin computes and logs ``graph_acc/scene_block`` and ``graph_acc/answer_nodes`` metrics. teacher-forced
    next-token accuracy restricted to graph-related tokens.

    * ``scene_block`` — tokens in the scene-graph block that name a node (the
      positions the graph PE injects into).
    * ``answer_nodes`` — node-name tokens the model emits in its final answer.

    The two masks come from per-example index lists precomputed in
    ``preprocess_dataset`` and carried verbatim by the collator
    (:class:`prism.data.data.TokenIndexCollator`); right-padding keeps each index
    valid in the padded batch. Counts accumulate across the logging window and are
    flushed (DDP-reduced) in :meth:`log`, so the metric rides the trainer's existing
    wandb logging cadence with no extra callback. Mixed into both the graph trainer
    and the plain-``llm`` baseline trainer so the two are comparable.
    """

    def _reset_token_acc(self):
        self._gta = {"scene_c": 0, "scene_n": 0, "ans_c": 0, "ans_n": 0}

    def _accumulate_token_acc(self, outputs, input_ids, scene_idx, answer_idx):
        logits = getattr(outputs, "logits", None)
        if logits is None or (scene_idx is None and answer_idx is None):
            return
        if not hasattr(self, "_gta"):
            self._reset_token_acc()
        with torch.no_grad():
            # preds[t] predicts token t+1; a node token at position p is graded by
            # preds[p-1] == input_ids[p]. So compare in the shifted (target) frame
            # and map each index p -> p-1.
            preds = logits[:, :-1, :].argmax(dim=-1)      # [B, S-1]
            correct = preds == input_ids[:, 1:]            # [B, S-1]
            width = correct.shape[1]
            for b in range(correct.shape[0]):
                for key, idx in (("scene", scene_idx), ("ans", answer_idx)):
                    if not idx or b >= len(idx):
                        continue
                    pos = [p - 1 for p in idx[b] if 1 <= p <= width]
                    if not pos:
                        continue
                    sel = correct[b, torch.as_tensor(pos, device=correct.device)]
                    self._gta[f"{key}_n"] += sel.numel()
                    self._gta[f"{key}_c"] += int(sel.sum().item())

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        scene_idx = inputs.pop("scene_node_idx", None)
        answer_idx = inputs.pop("answer_node_idx", None)
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        try:
            self._accumulate_token_acc(outputs, inputs["input_ids"], scene_idx, answer_idx)
        except Exception as e:  # a diagnostic must never break training
            warnings.warn(f"graph-token-accuracy metric skipped: {type(e).__name__}: {e}")
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, *args, **kwargs):
        gta = getattr(self, "_gta", None)
        if gta is not None:
            counts = torch.tensor(
                [gta["scene_c"], gta["scene_n"], gta["ans_c"], gta["ans_n"]],
                dtype=torch.float64, device=self.args.device)
            if self.args.world_size > 1:
                counts = self.accelerator.reduce(counts, reduction="sum")
            scene_c, scene_n, ans_c, ans_n = counts.tolist()
            if scene_n > 0:
                logs["graph_acc/scene_block"] = scene_c / scene_n
            if ans_n > 0:
                logs["graph_acc/answer_nodes"] = ans_c / ans_n
            self._reset_token_acc()
        return super().log(logs, *args, **kwargs)


class GraphSFTTrainer(LossTargetMixin, GraphTokenAccuracyMixin, SFTTrainer):
    def __init__(self, *args, gnn_config: dict, freeze_pe: bool = False,
                 loss_target: str = "all", **kwargs):
        super().__init__(*args, **kwargs)
        self.gnn_config = gnn_config
        self.freeze_pe = freeze_pe
        self._set_loss_target(loss_target)
        if freeze_pe:
            # Stage-1 SFT: PE stays at init (gate closed); only LoRA trains.
            return
        # Re-enable gradients for graph encoder and gate/projection (PEFT froze them).
        if gnn_config.get("architecture") == "composite_graph_gt":
            for p in self.model.gt_model.parameters():
                p.requires_grad = True
            for p in self.model.injection.parameters():
                p.requires_grad = True
            # pe_qk_injection: re-enable q/k(/v) code projections PEFT froze.
            if hasattr(self.model, "pe_q_proj"):
                for name in ("pe_q_proj", "pe_k_proj", "pe_v_proj"):
                    mod = getattr(self.model, name, None)
                    if mod is not None:
                        for p in mod.parameters():
                            p.requires_grad = True
            # c_bias scalar gains λ_C/λ_S/λ_V are non-LoRA — re-enable them.
            for name in ("lam_c", "lam_psi", "lam_v"):
                p = getattr(self.model, name, None)
                if p is not None:
                    p.requires_grad = True
        elif gnn_config.get("architecture") == "graph_mask_llm":
            # graph_mask_llm: parameter-free mask — only LoRA adapters train.
            pass
        else:
            for p in self.model.pe_model.parameters():
                p.requires_grad = True
            for p in self.model.pe_proj.parameters():
                p.requires_grad = True
            self.model.pe_gain.requires_grad = True
            # pe_norm is a non-LoRA module; re-enable so magnitude calibration adapts.
            if getattr(self.model, "pe_norm", None) is not None:
                for p in self.model.pe_norm.parameters():
                    p.requires_grad = True

    def create_optimizer(self):
        """Two learning-rate groups: structural path (GT + R-PEARL + gate) at
        ``structural_lr_mult`` × base LR; LLM/LoRA at base LR. Falls back to the
        stock optimizer when the multiplier is 1.0.
        """
        mult = float(self.gnn_config.get("structural_lr_mult", 1.0))
        # graph_mask_llm: parameter-free, no structural params — use stock optimizer.
        if (self.optimizer is not None or mult == 1.0
                or self.gnn_config.get("architecture") == "graph_mask_llm"):
            return super().create_optimizer()

        opt_model = self.model
        if self.gnn_config.get("architecture") == "composite_graph_gt":
            structural = list(self.model.gt_model.parameters()) + list(self.model.injection.parameters())
            # pe_qk_injection: q/k(/v) projections are structural (boosted LR).
            if hasattr(self.model, "pe_q_proj"):
                for name in ("pe_q_proj", "pe_k_proj", "pe_v_proj"):
                    mod = getattr(self.model, name, None)
                    if mod is not None:
                        structural += list(mod.parameters())
            # c_bias scalar gains are structural too.
            for name in ("lam_c", "lam_psi", "lam_v"):
                p = getattr(self.model, name, None)
                if p is not None:
                    structural.append(p)
        else:
            structural = (list(self.model.pe_model.parameters())
                          + list(self.model.pe_proj.parameters()) + [self.model.pe_gain])
        structural_ids = {id(p) for p in structural}

        decay_names = set(self.get_decay_parameter_names(opt_model))
        base_lr = self.args.learning_rate
        groups = []
        # Structural at boosted LR, LLM/LoRA at base LR; each split into decay / no-decay.
        for is_struct, lr in ((True, base_lr * mult), (False, base_lr)):
            named = [(n, p) for n, p in opt_model.named_parameters()
                     if p.requires_grad and (id(p) in structural_ids) == is_struct]
            decay = [p for n, p in named if n in decay_names]
            no_decay = [p for n, p in named if n not in decay_names]
            if decay:
                groups.append({"params": decay, "lr": lr, "weight_decay": self.args.weight_decay})
            if no_decay:
                groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})

        try:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
        except TypeError:
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)
        optimizer_kwargs.pop("params", None)
        optimizer_kwargs.pop("lr", None)
        self.optimizer = optimizer_cls(groups, lr=base_lr, **optimizer_kwargs)
        n_struct = sum(p.numel() for p in structural if p.requires_grad)
        print(f"[train] structural LR group: {mult}x base = {base_lr * mult:.2e} "
              f"({n_struct / 1e6:.2f}M params); LLM/LoRA at base LR {base_lr:.2e}")
        return self.optimizer

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch, **kwargs)
        # Capture grad norms before zero_grad() (backward already ran in super).
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
        if self.gnn_config.get("architecture") == "composite_graph_gt":
            # Save the Graph Transformer (R-PEARL inside) and the cold-start gate.
            gnn_state = {
                'gt_model': self.model.gt_model.state_dict(),
                'injection': self.model.injection.state_dict(),
            }
            # In-attention injection variant: persist the dedicated q/k(/v) projections.
            if hasattr(self.model, "pe_q_proj"):
                gnn_state['pe_q_proj'] = self.model.pe_q_proj.state_dict()
                gnn_state['pe_k_proj'] = self.model.pe_k_proj.state_dict()
                if getattr(self.model, "pe_v_proj", None) is not None:
                    gnn_state['pe_v_proj'] = self.model.pe_v_proj.state_dict()
            # c_bias: persist scalar gains λ_C/λ_S/λ_V.
            if getattr(self.model, "c_bias", False):
                gnn_state['c_bias_gains'] = {
                    name: getattr(self.model, name).detach().cpu()
                    for name in ("lam_c", "lam_psi", "lam_v")
                    if getattr(self.model, name, None) is not None
                }
            torch.save(gnn_state, os.path.join(output_dir, "gnn_weights.pt"))
            torch.save({
                'rpearl': self.model.gt_model.pe_model.state_dict(),
            }, os.path.join(output_dir, "rpearl_weights.pt"))
        elif self.gnn_config.get("architecture") == "graph_mask_llm":
            # Parameter-free: mask rebuilt from config; gnn_config.json + LoRA adapter suffice.
            pass
        elif self.gnn_config.get("architecture") == "rpearl_gt_llm":
            # Full GT: save the whole GraphTransformer (includes R-PEARL inside) + projection head.
            torch.save({
                'gt_model': self.model.pe_model.state_dict(),
                'pe_proj': self.model.pe_proj.state_dict(),
                'pe_gain': self.model.pe_gain.data,
                **({'pe_norm': self.model.pe_norm.state_dict()}
                   if self.model.pe_norm is not None else {}),
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
                **({'pe_norm': self.model.pe_norm.state_dict()}
                   if self.model.pe_norm is not None else {}),
            }, os.path.join(output_dir, "gnn_weights.pt"))
        # Save the LoRA adapter whenever one is attached (including frozen carried-forward
        # adapters). freeze_llm=True attaches no adapter, so this is skipped for that path.
        if getattr(self.model, "peft_config", None) is not None:
            super().save_model(output_dir, _internal_call)


class BaselineSFTTrainer(LossTargetMixin, GraphTokenAccuracyMixin, SFTTrainer):
    """Plain-``llm`` baseline trainer. Identical to ``SFTTrainer`` except it logs the
    ``graph_acc/*`` metric (via :class:`GraphTokenAccuracyMixin`) so the baseline is
    comparable to the graph architectures. Paired with
    :class:`prism.data.data.TokenIndexCollator`, which carries the precomputed
    graph-token index columns the metric needs.

    Persists a ``train_config.json`` (the plain-LLM analogue of the graph trainer's
    ``gnn_config.json``) so the standalone eval boundary can recover the train-time
    ``text_edge_list`` policy. Without it an LLM trained with ``text_edge_list=none``
    would be silently evaluated with edge bullets re-added (train/eval mismatch).
    """

    def __init__(self, *args, train_config: dict, loss_target: str = "all", **kwargs):
        super().__init__(*args, **kwargs)
        self.train_config = train_config
        self._set_loss_target(loss_target)

    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "train_config.json"), "w") as f:
            json.dump(self.train_config, f, indent=2)
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
    # Zero-shot baseline: skip optimization and evaluate the untrained base model.
    no_train: bool = False
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
    # Activation recompute (saves memory). use_reentrant=False required: GT passes
    # grad-carrying inputs_embeds that the reentrant path mishandles.
    gradient_checkpointing: bool = True
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
    architecture: str = "rpearl_llm"  # "rpearl_llm", "rpearl_gt_llm", "gt_llm", "graph_mask_llm", "composite_graph_gt", or "llm"
    # GT-specific params (used when architecture == "rpearl_gt_llm" / "composite_graph_gt")
    gt_num_layers: int = 3
    gt_heads: int = 8
    eps: float = 1e-8

    def __post_init__(self):
        self.eps = float(self.eps)
        if self.loss_target not in ("all", "responses", "edge_list"):
            raise ValueError(
                f"loss_target must be 'all', 'responses', or 'edge_list', "
                f"got {self.loss_target!r}")
        if self.loss_target == "edge_list" and self.text_edge_list != "present":
            raise ValueError(
                "loss_target='edge_list' requires text_edge_list='present' "
                "(the edge bullets must be in the text to supervise them).")
    k_gt: int = 3
    text_edge_list: str = "present"   # "present" or "none"
    # Composite-graph params (used when architecture == "composite_graph_gt")
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
    max_gather_rows: int = 2_000_000
    fixed_seed_mode: bool = False
    fixed_seed_value: int = 0
    # R-PEARL readout fed to the GT (composite_graph_gt only): "mean" = E_q[Φ(q)];
    # "second_moment" = C @ X for C = E_q[Φ(q)Φ(q)ᵀ] (carries relative position).
    pe_readout: str = "mean"
    # Center the second moment into a covariance (C·s = E[Φ(Φᵀs)] − Ψ(Ψᵀs)); needed
    # so the mean doesn't dominate via rank-1 ΨΨᵀ and collapse C to averaging.
    pe_center_moment: bool = True
    # Initial value of the injection gate (g = tanh(pe_gain)). 0.0 = cold-start
    # (Ψ off at init, gate ramps it in); 1.0 = active from step 0.
    pe_gain_init: float = 0.0
    # RMS-normalize and rescale the projected Ψ to the base model's embedding scale
    # before injection (magnitude calibration).
    use_pe_norm: bool = True
    # R-PEARL input features: "random" = PEARL random probes; "word_embeddings" =
    # mean word-embedding of the node's name tokens (deterministic, no probes).
    pe_node_features: str = "random"
    # Set position_id 0 on graph (node-name) tokens so RoPE is the identity there;
    # their position comes from Ψ instead. Causality is unaffected.
    disable_graph_token_rope: bool = False
    # graph_mask_llm: node tokens attend only within mask_k_hops graph hops.
    mask_k_hops: int = 1
    # Symmetrize the adjacency (OR with its transpose) before building the mask.
    mask_symmetrize: bool = True
    # When False, mask is self-loops only (no cross-node attention) — floor ablation
    # for whether edge connectivity in the mask matters.
    mask_use_edges: bool = True
    # Gated injection (composite_graph_gt only)
    injection_mode: str = "interpolate"
    gate_init: float = 0.0
    gate_per_dim: bool = False
    disable_rope: bool = True
    # Inject the GT-refined token code into q/k(/v) inside every attention layer
    # (dedicated W_q/W_k/W_v), in place of RoPE. composite_graph_gt only.
    pe_qk_injection: bool = False
    # Also inject into the attention *value* (v += W_v·Y_tok), not just q/k. False =
    # q/k only (no value/readout path).
    pe_inject_v: bool = True
    # Replace post-RoPE q/k at every layer with the composite token covariance C_tok
    # (q ← C_tok·q, k ← C_tok·k) instead of additive projections. Deterministic, scaled
    # to ‖X‖. Selects InjectedCompositeGraphLLM; pairs with disable_rope=True.
    c_per_layer: bool = False
    # C_tok as additive logit bias (λ_C) + residual value mix (v ← v + λ_V·C·v);
    # no q/k transform. Optional scene bias λ_S via use_scene_bias. Gains: λ_C,λ_S,λ_V.
    c_bias: bool = False
    use_scene_bias: bool = True
    # Live c_bias covariance kernel: "sampled" (E_q[ΦΦᵀ]−ΨΨᵀ probe estimate) or
    # "analytic" (all-layer H(S)H(S)* matrix powers). See InjectedCompositeGraphLLM.
    c_kernel: str = "sampled"
    # GT + R-PEARL + gate train at structural_lr_mult × base LR; lora_warmup_steps
    # freezes LLM/LoRA for the first N steps so the structural path learns first.
    structural_lr_mult: float = 1.0
    lora_warmup_steps: int = 0
    # c_bias (Design D): ramp the additive covariance gain λ_C 0→1 over the first N steps.
    lam_c_warmup_steps: int = 0
    # Prompt / debug-viz switches
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
    # Path (file or directory of {graph, tasks} JSONs) for post-training cross-eval;
    # results written to <output_dir>/eval_logs/cross_eval/<graph>.json.
    post_train_eval_graphs: Optional[str] = None
    # Enable SPINE ICL examples during train-time eval and post-train cross-eval.
    eval_use_icl: bool = True
    # --- Multistage training ------------------------------------------------
    # Weight-only init from a prior stage (NOT HF resume; fresh optimizer each stage).
    # init_lora_from: attach a saved LoRA adapter instead of a fresh one.
    # init_pe_from: load gnn_weights.pt into the fresh PE module.
    init_lora_from: Optional[str] = None
    init_pe_from: Optional[str] = None
    # Freeze the carried-forward LoRA adapter (PE-only stage). Distinct from
    # freeze_llm, which drops the adapter and trains PE on the raw base.
    freeze_lora: bool = False
    # Freeze the PE module — Stage 1 SFT trains ONLY the LoRA while the PE stays
    # inert at init (paired with pe_gain_init=0). See GraphSFTTrainer.__init__.
    freeze_pe: bool = False
    # Token span for the next-token loss: "all" = full sequence; "responses" =
    # assistant turns only (via precomputed assistant_idx); "edge_list" = edge
    # bullets only (requires text_edge_list="present").
    loss_target: str = "all"
    # Number of graphs to evaluate each interval (first N from eval_data).
    eval_num_graphs: int = 5
    # EvalCallback cadence in epochs; raise for long PE-only stages to bound eval cost.
    eval_epoch_interval: float = 1.0


def _load_eval_samples_by_graph(eval_data: str, num_graphs: int) -> dict:
    """Resolve `eval_data` (file, directory, or glob) into `{graph_name: [EvalSample]}`,
    truncated to the first `num_graphs` graphs (sorted by file stem).

    Used by the train-time periodic `EvalCallback` and the `no_train` zero-shot
    baseline so both score the same multi-graph held-out set. A single-file
    `eval_data` resolves to one graph (back-compatible). `num_graphs <= 0` keeps
    all resolved graphs.
    """
    samples_by_graph, _ = loading.load_samples_by_graph(eval_data)
    if num_graphs and num_graphs > 0 and len(samples_by_graph) > num_graphs:
        kept = list(samples_by_graph.items())[:num_graphs]
        dropped = len(samples_by_graph) - num_graphs
        print(f"[eval] capping train-time eval to {num_graphs} of "
              f"{len(samples_by_graph)} graphs ({dropped} dropped)")
        samples_by_graph = dict(kept)
    return samples_by_graph


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

    is_gnn = config.architecture in ("rpearl_llm", "rpearl_gt_llm", "gt_llm", "graph_mask_llm", "composite_graph_gt")
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


def _run_zero_shot_eval(
    model, tokenizer, config: "TrainConfig", output_dir: str, eval_samples_by_graph: dict
) -> None:
    """Evaluate the untrained base model over the multi-graph eval set for the
    ``no_train`` baseline.

    Mirrors the per-epoch :class:`prism.eval.callbacks.EvalCallback` artifact
    (``eval_logs/step_000000_epoch_0.000.json`` + the ``eval/accuracy`` wandb
    point) so the base model's out-of-the-box score is directly comparable to the
    trained runs and consumable by the judge-eval skill. ``include_edge_list`` is
    resolved here from the same ``text_edge_list == "present"`` policy used
    everywhere else, so the with-/without-edges baselines differ only in the
    LLM-facing edge text.
    """
    eval_log_dir = os.path.join(output_dir, "eval_logs")
    os.makedirs(eval_log_dir, exist_ok=True)
    model.eval()
    results = evaluate.eval_model_multiple_graphs(
        model,
        tokenizer,
        eval_samples_by_graph,
        include_edge_list=(config.text_edge_list == "present"),
        use_icl=config.eval_use_icl,
        permutation=None,
        on_graph_done=None,
    )
    log_data = _aggregate_multi_graph_eval(results, step=0, epoch=0.0)
    log_file = os.path.join(eval_log_dir, "step_000000_epoch_0.000.json")
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(
        f"[no_train] zero-shot eval/accuracy = {log_data['accuracy']:.4f} "
        f"({log_data['num_correct']}/{log_data['num_samples']}) over "
        f"{log_data['num_graphs']} graph(s) -> {log_file}"
    )
    if wandb.run is not None:
        wandb.log({"eval/accuracy": log_data["accuracy"], "epoch": 0.0})


def _aggregate_multi_graph_eval(results: dict, *, step: int, epoch: float) -> dict:
    """Fold per-graph `GraphEvalResultSummary`s into one eval-log payload.

    Overall accuracy is the micro-average of the per-graph keyword rate
    (`GraphEvalResultSummary.accuracy`) weighted by sample count — Σ(rate·n) / Σn.
    This is the SAME objective-keyword metric the old single-graph EvalCallback
    logged as `eval/accuracy` and that the per-graph `eval/acc/<name>` breakdown
    reports, so the headline reconciles with the breakdown and stays comparable to
    prior runs. Path metrics are aggregated over the concatenated samples, and a
    `per_graph` breakdown is kept. Shape matches the old single-graph log apart from
    the extra `num_graphs` / `per_graph` keys.
    """
    all_samples = [s for r in results.values() for s in r.samples]
    num_total = sum(r.num_total for r in results.values())
    # Per-graph keyword-correct count = accuracy·n (integer); sum for the headline.
    num_correct = sum(round(r.accuracy * r.num_total) for r in results.values())
    accuracy = (num_correct / num_total) if num_total else 0.0
    return {
        "step": step,
        "epoch": epoch,
        "accuracy": accuracy,
        "num_graphs": len(results),
        "num_samples": num_total,
        "num_correct": num_correct,
        "path_metrics": evaluate._aggregate_path_metrics(all_samples),
        "per_graph": {name: {"accuracy": r.accuracy, "num_correct": r.num_correct,
                             "num_total": r.num_total}
                      for name, r in results.items()},
        "samples": all_samples,
    }


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
def _load_pe_weights_into(model, init_pe_from: str, architecture: str) -> None:
    """Load a saved PE module (``gnn_weights.pt``) from a prior stage into ``model``.

    Operates on a training model (no merge, no eval). Only rpearl_llm /
    rpearl_gt_llm PE layouts are supported.
    """
    weights_path = os.path.join(init_pe_from, "gnn_weights.pt")
    gnn_weights = torch.load(weights_path, map_location="cpu")
    if architecture == "rpearl_gt_llm":
        model.pe_model.load_state_dict(gnn_weights["gt_model"], strict=False)
    elif architecture == "rpearl_llm":
        model.pe_model.load_state_dict(gnn_weights["pe_model"], strict=False)
    else:
        raise NotImplementedError(
            f"init_pe_from is only wired for rpearl_llm / rpearl_gt_llm, got {architecture!r}")
    model.pe_proj.load_state_dict(gnn_weights["pe_proj"])
    if "pe_gain" in gnn_weights:
        model.pe_gain.data.copy_(gnn_weights["pe_gain"])
    if getattr(model, "pe_norm", None) is not None and "pe_norm" in gnn_weights:
        model.pe_norm.load_state_dict(gnn_weights["pe_norm"])
    print(f"[multistage] loaded PE weights from {weights_path}")


# Multimodal Gemma-4 bases (e.g. gemma-4-31B) include vision/audio towers whose
# projections use Gemma4ClippableLinear (not nn.Linear) — PEFT cannot adapt them,
# and their leaf names collide with the text decoder. Exclude them from LoRA targets.
# Text-only / "unified" bases (Llama, gemma-4-12B) have no towers; regex hits nothing.
_MM_TOWER_KEYS = ("vision_tower", "audio_tower")


def _peft_tower_exclude(model) -> "str | None":
    """Return a PEFT `exclude_modules` regex for a multimodal base, else None.

    Detected from the actual module tree (not the model-id) so it is robust to
    the LoRA wrapper prefix and to future multimodal variants.
    """
    has_tower = any(
        any(k in name for k in _MM_TOWER_KEYS) for name, _ in model.named_modules()
    )
    return r".*(?:" + "|".join(_MM_TOWER_KEYS) + r").*" if has_tower else None


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

    # Semantic-feature mode: the GNN takes the LLM text hidden size as input width.
    _text_hidden = llm.config.get_text_config().hidden_size
    _node_feature_dim = _text_hidden if config.pe_node_features == "word_embeddings" else None

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
            node_feature_dim=_node_feature_dim,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=config.d_model,
                                          eps=config.eps, pe_gain_init=config.pe_gain_init,
                                          disable_graph_token_rope=config.disable_graph_token_rope,
                                          use_pe_norm=config.use_pe_norm,
                                          pe_node_features=config.pe_node_features)
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
            node_feature_dim=_node_feature_dim,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=config.d_model,
                                          eps=config.eps, pe_gain_init=config.pe_gain_init,
                                          disable_graph_token_rope=config.disable_graph_token_rope,
                                          use_pe_norm=config.use_pe_norm,
                                          pe_node_features=config.pe_node_features)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "gt_llm":
        # Pure Graph Transformer over semantic node features — no R-PEARL, no probes.
        if config.pe_node_features != "word_embeddings":
            raise ValueError(
                "architecture 'gt_llm' requires pe_node_features='word_embeddings' "
                f"(got {config.pe_node_features!r}); the GT has no random-probe input."
            )
        pe_model = gt_module.SemanticGraphTransformer(
            node_feature_dim=_text_hidden,
            d_model=config.d_model,
            num_layers=config.gt_num_layers,
            heads=config.gt_heads,
            dropout=config.dropout,
            k_gt=config.k_gt,
        )
        model = gnn_llm.GraphAugmentedLLM(llm, pe_model, d_model=config.d_model,
                                          eps=config.eps, pe_gain_init=config.pe_gain_init,
                                          disable_graph_token_rope=config.disable_graph_token_rope,
                                          use_pe_norm=config.use_pe_norm,
                                          pe_node_features=config.pe_node_features)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "graph_mask_llm":
        # Parameter-free structural attention mask: node tokens attend only along graph
        # edges. Reuses SpineDataCollator so graphs + injection_maps reach the model.
        model = gnn_llm.GraphMaskLLM(
            llm, k_hops=config.mask_k_hops, symmetrize=config.mask_symmetrize,
            use_edges=config.mask_use_edges)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "composite_graph_gt":
        # Composite-graph pipeline: token-cycle + scene graph + cross-links; R-PEARL +
        # GT refine it; cold-start gate injects into the RoPE-disabled LLM.
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
            max_gather_rows=config.max_gather_rows,
            fixed_seed_mode=config.fixed_seed_mode,
            fixed_seed_value=config.fixed_seed_value,
            pe_readout=config.pe_readout,
            center_second_moment=config.pe_center_moment,
        )
        composite_kwargs = dict(
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
        if config.pe_qk_injection or config.c_per_layer or config.c_bias:
            # Selects InjectedCompositeGraphLLM: pe_qk_injection adds GT code to q/k/v;
            # c_per_layer replaces q/k with C_tok; c_bias uses C_tok as an additive bias.
            composite_kwargs["inject_v"] = config.pe_inject_v
            composite_kwargs["c_per_layer"] = config.c_per_layer
            composite_kwargs["c_bias"] = config.c_bias
            composite_kwargs["use_scene_bias"] = config.use_scene_bias
            composite_kwargs["c_kernel"] = config.c_kernel
            model = gnn_llm.InjectedCompositeGraphLLM(
                llm, gt_model, d_model=config.d_model, **composite_kwargs)
        else:
            model = gnn_llm.CompositeGraphLLM(
                llm, gt_model, d_model=config.d_model, **composite_kwargs)
        collator = data.SpineDataCollator(tokenizer, mlm=False)

        if config.freeze_llm:
            model.llm.requires_grad_(False)
    elif config.architecture == "llm":
        # Pure LLM baseline; TokenIndexCollator carries graph-token index columns so
        # graph_acc/* metrics are comparable to graph architectures.
        model = llm
        collator = data.TokenIndexCollator(tokenizer, mlm=False)
    else:
        raise ValueError(f"Unknown architecture: {config.architecture!r}. Choose 'rpearl_llm', 'rpearl_gt_llm', 'gt_llm', 'graph_mask_llm', 'composite_graph_gt', or 'llm'.")

    # --- Multistage init: weight-only carry-over from a prior stage (NOT HF resume). --
    # Load PE first (lives outside the LoRA adapter), then attach the carried adapter.
    is_graph_arch = config.architecture in (
        "rpearl_llm", "rpearl_gt_llm", "gt_llm", "graph_mask_llm", "composite_graph_gt")
    if (config.init_pe_from or config.init_lora_from) and not is_graph_arch:
        raise ValueError(
            "init_pe_from / init_lora_from are only supported for graph architectures.")
    if config.init_pe_from:
        _load_pe_weights_into(model, config.init_pe_from, config.architecture)
    attach_existing_adapter = config.init_lora_from is not None
    if attach_existing_adapter:
        model = PeftModel.from_pretrained(
            model, config.init_lora_from, is_trainable=not config.freeze_lora)
        if config.gradient_checkpointing and not config.freeze_lora:
            # Replicate TRL's get_peft_model behavior so LoRA gradients flow back
            # through the frozen base under gradient checkpointing.
            model.enable_input_require_grads()
        print(f"[multistage] attached {'frozen' if config.freeze_lora else 'trainable'} "
              f"LoRA adapter from {config.init_lora_from}")

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

    # LoRA config; for multimodal bases, exclude vision/audio towers (see _peft_tower_exclude).
    tower_exclude = _peft_tower_exclude(model)
    lora_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=config.target_modules,
        exclude_modules=tower_exclude,
        task_type="CAUSAL_LM",
    )
    if tower_exclude:
        print(f"[peft] multimodal base detected — excluding LoRA targets matching "
              f"{tower_exclude!r} (vision/audio towers)")

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
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
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

    if config.architecture in ("rpearl_llm", "rpearl_gt_llm", "gt_llm", "graph_mask_llm", "composite_graph_gt"):
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
            "pe_gain_init": config.pe_gain_init,
            "use_pe_norm": config.use_pe_norm,
            "pe_node_features": config.pe_node_features,
            **({"k_gt": config.k_gt, "gt_num_layers": config.gt_num_layers,
                "gt_heads": config.gt_heads}
               if config.architecture in ("rpearl_gt_llm", "gt_llm", "composite_graph_gt") else {}),
            # graph_mask_llm rebuild params (read back by loaders for eval).
            **({"mask_k_hops": config.mask_k_hops, "mask_symmetrize": config.mask_symmetrize,
                "mask_use_edges": config.mask_use_edges}
               if config.architecture == "graph_mask_llm" else {}),
            # Composite-graph rebuild params (read back by loaders for eval).
            **({"k_gt": config.k_gt, "gt_num_layers": config.gt_num_layers,
                "gt_heads": config.gt_heads,
                "probe_distribution": config.probe_distribution, "m_test": config.m_test,
                "max_gather_rows": config.max_gather_rows,
                "fixed_seed_mode": config.fixed_seed_mode, "fixed_seed_value": config.fixed_seed_value,
                "injection_mode": config.injection_mode, "gate_init": config.gate_init,
                "gate_per_dim": config.gate_per_dim, "disable_rope": config.disable_rope,
                "structural_lr_mult": config.structural_lr_mult, "pe_readout": config.pe_readout,
                "pe_center_moment": config.pe_center_moment,
                "cycle_weight": config.cycle_weight, "cycle_directed": config.cycle_directed,
                "crosslink_weight": config.crosslink_weight,
                "crosslink_mention_to_node": config.crosslink_mention_to_node,
                "crosslink_mention_clique": config.crosslink_mention_clique,
                "pe_qk_injection": config.pe_qk_injection,
                "pe_inject_v": config.pe_inject_v,
                "c_per_layer": config.c_per_layer,
                "c_bias": config.c_bias,
                "use_scene_bias": config.use_scene_bias,
                "c_kernel": config.c_kernel}
               if config.architecture == "composite_graph_gt" else {}),
        }
        trainer = GraphSFTTrainer(
            model=model,
            data_collator=collator,
            processing_class=tokenizer,
            # No fresh adapter when one was carried forward (it's already attached)
            # or when freeze_llm requests a PE-only run on the raw base.
            peft_config=lora_config if (not config.freeze_llm and not attach_existing_adapter) else None,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_args,
            gnn_config=gnn_config,
            freeze_pe=config.freeze_pe,
            loss_target=config.loss_target,
        )
    else:
        trainer = BaselineSFTTrainer(
            model=model,
            data_collator=collator,
            processing_class=tokenizer,
            peft_config=lora_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=sft_args,
            train_config={
                "architecture": config.architecture,
                "base_model": config.base_model,
                "text_edge_list": config.text_edge_list,
            },
            loss_target=config.loss_target,
        )

    # Log all training config parameters to wandb
    if wandb.run is not None:
        wandb.config.update(asdict(config), allow_val_change=True)
        if config_file is not None:
            with open(config_file) as f:
                wandb.config.update({"_config_yaml": f.read()}, allow_val_change=True)

    eval_samples_by_graph = _load_eval_samples_by_graph(config.eval_data, config.eval_num_graphs)
    trainer.add_callback(callbacks.EvalCallback(
        eval_samples_by_graph,
        tokenizer=tokenizer,
        use_icl=config.eval_use_icl,
        include_edge_list=(config.text_edge_list == "present"),
        eval_epoch_interval=config.eval_epoch_interval,
    ))

    if config.architecture in ("rpearl_llm", "rpearl_gt_llm", "gt_llm"):
        trainer.add_callback(callbacks.GradientDebugCallback())
    elif config.architecture == "composite_graph_gt":
        # Per-component grad norms, GT output magnitude, gate value, injection count.
        trainer.add_callback(callbacks.GradientDebugCallback())
        # Fiedler, scene-mass, gate, contrib-ratio diagnostics (+ visualizer if enabled).
        trainer.add_callback(callbacks.AugGraphDebugCallback(
            enable_visualizer=config.enable_visualizer,
            visualizer_dir=os.path.join(output_dir, "visuals"),
        ))
        if config.lora_warmup_steps > 0:
            trainer.add_callback(callbacks.LoraWarmupCallback(config.lora_warmup_steps))
        if getattr(config, "lam_c_warmup_steps", 0) > 0 and getattr(config, "c_bias", False):
            trainer.add_callback(callbacks.LamCWarmupCallback(config.lam_c_warmup_steps))

    # no_train: evaluate the untrained base model zero-shot instead of training.
    if config.no_train:
        print("[no_train] Skipping optimization — evaluating the base model zero-shot.")
        _run_zero_shot_eval(trainer.model, tokenizer, config, sft_args.output_dir, eval_samples_by_graph)
    else:
        trainer.train()

    # Save model artifacts
    trainer.save_model()
    tokenizer.save_pretrained(sft_args.output_dir)

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
