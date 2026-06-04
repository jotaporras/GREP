import json
import torch
import wandb
from pathlib import Path
from transformers.trainer_callback import TrainerCallback

from prism.eval import evaluate


class EvalCallback(TrainerCallback):
    def __init__(
        self,
        eval_samples,
        *,
        tokenizer,
        use_icl: bool,
        include_edge_list: bool,
        eval_epoch_interval: float = 1.0,
    ):
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
        self.use_icl = use_icl
        self.include_edge_list = include_edge_list
        self.eval_epoch_interval = eval_epoch_interval
        self._steps_per_interval: int | None = None
        self._last_eval_step: int = -1
        self.metrics = {}

    def on_train_begin(self, args, state, control, **kwargs):
        steps_per_epoch = state.max_steps / args.num_train_epochs
        self._steps_per_interval = max(1, round(steps_per_epoch * self.eval_epoch_interval))
        self._eval_log_dir = Path(args.output_dir) / "eval_logs"
        self._eval_log_dir.mkdir(parents=True, exist_ok=True)

    def _run_eval(self, args, state, **kwargs):
        model = kwargs["model"]
        model.eval()
        accuracy, sample_results = evaluate.eval_model_single_graph(
            model,
            self.tokenizer,
            self.eval_samples,
            include_edge_list=self.include_edge_list,
            use_icl=self.use_icl,
            permutation=None,
        )
        model.train()

        log_data = {
            "step": state.global_step,
            "epoch": state.epoch,
            "accuracy": accuracy,
            "num_samples": len(sample_results),
            "num_correct": sum(r["correct"] for r in sample_results),
            "samples": sample_results,
        }
        log_file = self._eval_log_dir / f"step_{state.global_step:06d}_epoch_{state.epoch:.3f}.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        if wandb.run is not None:
            wandb.save(str(log_file), base_path=str(self._eval_log_dir))

        wandb.log({"eval/accuracy": accuracy, "epoch": state.epoch})
        self.metrics = {
            "eval/accuracy": accuracy,
        }

    def on_step_end(self, args, state, control, **kwargs):
        if self._steps_per_interval is None:
            return control
        if (
            state.global_step % self._steps_per_interval == 0
            and state.global_step != self._last_eval_step
            and state.global_step > 0
        ):
            self._last_eval_step = state.global_step
            self._run_eval(args, state, **kwargs)
        return control

    def get_latest_metrics(self):
        return self.metrics

class GradientDebugCallback(TrainerCallback):
    """Logs per-component gradient norms, PE magnitude, and injection counts to W&B.

    For ``GraphAugmentedLLM`` (``rpearl_llm`` / ``rpearl_gt_llm``): a ``pe_model``
    (R-PEARL, optionally inside a GraphTransformer), a ``pe_proj`` head, a scalar
    ``pe_gain``, and ``_augment_embeddings`` for injection. Logs grad norms for the
    GNN/PE, projection, gain, and LoRA — and, for the ``rpearl_gt_llm`` variant,
    the inner R-PEARL, GT attention blocks, and GT output norm — plus the
    ``pe_proj`` output magnitude and the injection count.

    Hooks observe without duplicating logic. Gradient norms are captured by
    ``_capture_grad_norms`` (called from ``GraphSFTTrainer.training_step`` after
    backward, before zero_grad) so they reflect real gradients — HF Trainer
    zeroes grads before ``on_log`` fires, so reading ``.grad`` there returns 0.
    """

    def __init__(self):
        self._pe_norm = float("nan")
        self._pe_has_nan = False
        self._emb_norm = float("nan")
        self._num_injections = 0
        self._hooked = False
        self._captured_grad_norms = {}

    @staticmethod
    def _unwrap_peft(model):
        """Navigate PeftModel → LoraModel → GraphAugmentedLLM."""
        inner = model
        if hasattr(inner, 'base_model'):
            inner = inner.base_model
        if hasattr(inner, 'model') and hasattr(inner.model, 'pe_proj'):
            inner = inner.model
        return inner

    @staticmethod
    def _supported(inner):
        """True when the model exposes a PE source we can introspect."""
        return hasattr(inner, "pe_model")

    def _install_hooks(self, model):
        """Install lightweight hooks that observe without duplicating logic.

        Unwraps PEFT so the injection wrapper lives on the actual GraphAugmentedLLM
        instance (whose ``forward`` calls it), not on the PeftModel wrapper.
        """
        if self._hooked:
            return
        inner = self._unwrap_peft(model)
        callback = self

        # Hook 1: capture PE norm from pe_proj output via a forward hook.
        def _pe_proj_hook(_module, _input, output):
            callback._pe_norm = output.detach().norm().item()
            callback._pe_has_nan = bool(output.detach().isnan().any())

        inner.pe_proj.register_forward_hook(_pe_proj_hook)

        # Hook 2: wrap _augment_embeddings on the actual GraphAugmentedLLM so
        # that self._augment_embeddings() inside forward() hits our wrapper.
        orig_augment = inner._augment_embeddings

        def _wrapped_augment(input_ids, graphs, injection_maps):
            with torch.no_grad():
                emb_table = inner.llm.get_input_embeddings()
                callback._emb_norm = emb_table(input_ids.to(emb_table.weight.device)).norm().item()
            callback._num_injections = sum(
                len(spans) for imap in injection_maps for spans in imap.values()
            )
            return orig_augment(input_ids, graphs, injection_maps)

        inner._augment_embeddings = _wrapped_augment
        self._hooked = True

    @staticmethod
    def _grad_norm(params):
        sq_sum = 0.0
        for p in params:
            if p.grad is not None:
                sq_sum += p.grad.detach().float().norm().item() ** 2
        return sq_sum ** 0.5

    def _capture_grad_norms(self, model):
        """Snapshot per-component gradient norms.

        Must be called after backward() but before zero_grad() — i.e. from
        GraphSFTTrainer.training_step.
        """
        inner = self._unwrap_peft(model)
        if not self._supported(inner):
            return
        self._captured_grad_norms["lora"] = self._grad_norm(
            p for _, p in inner.llm.named_parameters() if p.requires_grad
        )

        # GraphAugmentedLLM: pe_model (+ optional GT) + pe_proj + pe_gain.
        self._captured_grad_norms["gnn"] = self._grad_norm(inner.pe_model.parameters())
        self._captured_grad_norms["pe_proj"] = self._grad_norm(inner.pe_proj.parameters())
        self._captured_grad_norms["pe_gain"] = self._grad_norm([inner.pe_gain])
        if hasattr(inner.pe_model, "blocks"):
            self._captured_grad_norms["rpearl"] = self._grad_norm(inner.pe_model.pe_model.parameters())
            self._captured_grad_norms["gt_blocks"] = self._grad_norm(inner.pe_model.blocks.parameters())
            self._captured_grad_norms["gt_output_norm"] = self._grad_norm(inner.pe_model.output_norm.parameters())

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None and self._supported(self._unwrap_peft(model)):
            self._install_hooks(model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None:
            return
        inner = self._unwrap_peft(model)
        if not self._supported(inner):
            return

        lr = state.log_history[-1].get("learning_rate", float("nan")) if state.log_history else float("nan")
        g = self._captured_grad_norms

        # GraphAugmentedLLM (rpearl_llm / rpearl_gt_llm).
        metrics = {
            "debug/grad_norm_gnn": g.get("gnn", 0.0),
            "debug/grad_norm_pe_proj": g.get("pe_proj", 0.0),
            "debug/grad_norm_lora": g.get("lora", 0.0),
            "debug/pe_output_norm": self._pe_norm,
            "debug/pe_has_nan": int(self._pe_has_nan),
            "debug/embedding_norm": self._emb_norm,
            "debug/num_injections": self._num_injections,
            "debug/lr": lr,
            "debug/pe_gain": inner.pe_gain.item(),
            "debug/grad_norm_pe_gain": g.get("pe_gain", 0.0),
        }
        # rpearl_gt_llm: split gradient norms by GT sub-component.
        if hasattr(inner.pe_model, "blocks"):
            metrics["debug/grad_norm_rpearl"] = g.get("rpearl", 0.0)
            metrics["debug/grad_norm_gt_blocks"] = g.get("gt_blocks", 0.0)
            metrics["debug/grad_norm_gt_output_norm"] = g.get("gt_output_norm", 0.0)

        if wandb.run is not None:
            wandb.log(metrics, step=state.global_step)


class MetricsTrackerCallback(TrainerCallback):
    """Custom callback that captures metrics for retrieval"""

    def __init__(self, trainer):
        self.trainer = trainer
        # Flag this as our custom callback
        self._is_metrics_tracker = True

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Capture metrics during evaluation"""
        if metrics:
            self.trainer.latest_metrics.update(metrics)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Capture metrics during logging"""
        if logs:
            self.trainer.latest_metrics.update(logs)
