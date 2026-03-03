import json
import torch
import wandb
from pathlib import Path
from transformers.trainer_callback import TrainerCallback

from prism.eval import run_eval


class EvalCallback(TrainerCallback):
    def __init__(self, eval_samples, tokenizer=None, eval_epoch_interval: float = 1.0):
        self.eval_samples = eval_samples
        self.tokenizer = tokenizer
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
        accuracy, sample_results = run_eval.eval_model(
            model=model,
            tokenizer=self.tokenizer,
        eval_samples=self.eval_samples,
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
    """Logs per-component gradient norms, PE magnitudes, and injection counts to W&B.

    Attaches to the model's _augment_embeddings to capture PE norms and bucket
    stats at every forward pass, and logs gradient norms split by component
    aftfmaxer every backward pass.
    """

    def __init__(self):
        self._pe_norm = float("nan")
        self._pe_has_nan = False
        self._emb_norm = float("nan")
        self._num_injections = 0
        self._hooked = False

    def _install_hooks(self, model):
        """Install lightweight hooks that observe without duplicating logic."""
        if self._hooked:
            return
        callback = self

        # Hook 1: capture PE norm from pe_proj output via a forward hook on pe_proj.
        def _pe_proj_hook(_module, _input, output):
            callback._pe_norm = output.detach().norm().item()
            callback._pe_has_nan = bool(output.detach().isnan().any())

        model.pe_proj.register_forward_hook(_pe_proj_hook)

        # Hook 2: capture embedding norm + injection count by wrapping _augment_embeddings.
        orig_augment = model._augment_embeddings

        def _wrapped_augment(input_ids, graphs, injection_maps):
            # Capture base embedding norm before injection.
            with torch.no_grad():
                callback._emb_norm = model.llm.get_input_embeddings()(input_ids).norm().item()
            # Count injections directly from pre-computed injection maps.
            total = 0
            for imap in injection_maps:
                total += sum(len(spans) for spans in imap.values())
            callback._num_injections = total
            # Call the real method — no duplicated injection logic.
            return orig_augment(input_ids, graphs, injection_maps)

        model._augment_embeddings = _wrapped_augment
        self._hooked = True

    def _grad_norm(self, params):
        grads = [p.grad for p in params if p.grad is not None]
        if not grads:
            return 0.0
        return torch.cat([g.detach().flatten() for g in grads]).norm().item()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if hasattr(model, "pe_model"):
            self._install_hooks(model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not hasattr(model, "pe_model"):
            return

        gnn_norm = self._grad_norm(model.pe_model.parameters())
        proj_norm = self._grad_norm(model.pe_proj.parameters())
        lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]
        lora_norm = self._grad_norm(lora_params)

        lr = state.log_history[-1].get("learning_rate", float("nan")) if state.log_history else float("nan")
        metrics = {
            "debug/grad_norm_gnn": gnn_norm,
            "debug/grad_norm_pe_proj": proj_norm,
            "debug/grad_norm_lora": lora_norm,
            "debug/pe_output_norm": self._pe_norm,
            "debug/pe_has_nan": int(self._pe_has_nan),
            "debug/embedding_norm": self._emb_norm,
            "debug/num_injections": self._num_injections,
            "debug/lr": lr,
        }
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
