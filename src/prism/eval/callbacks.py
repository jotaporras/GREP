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

        # M10 (R4) path-validity metrics over the generate-then-validate samples,
        # logged alongside the existing keyword accuracy.
        path_metrics = evaluate._aggregate_path_metrics(sample_results)

        log_data = {
            "step": state.global_step,
            "epoch": state.epoch,
            "accuracy": accuracy,
            "num_samples": len(sample_results),
            "num_correct": sum(r["correct"] for r in sample_results),
            "path_metrics": path_metrics,
            "samples": sample_results,
        }
        log_file = self._eval_log_dir / f"step_{state.global_step:06d}_epoch_{state.epoch:.3f}.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        if wandb.run is not None:
            wandb.save(str(log_file), base_path=str(self._eval_log_dir))

        wandb_metrics = {"eval/accuracy": accuracy, "epoch": state.epoch}
        for k, v in path_metrics.items():
            if v is not None:
                wandb_metrics[f"grep/path_{k}"] = v
        wandb.log(wandb_metrics)
        self.metrics = {"eval/accuracy": accuracy}
        self.metrics.update({f"grep/path_{k}": v for k, v in path_metrics.items() if v is not None})

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
    stats at every forward pass. Gradient norms are captured by
    ``_capture_grad_norms`` (called from ``GraphSFTTrainer.training_step``
    after backward, before zero_grad) so they reflect actual gradients.

    Architecture-aware: when pe_model is a GraphTransformer (rpearl_gt_llm),
    logs separate gradient norms for the inner R-PEARL, GT attention blocks,
    and GT output norm.
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

    def _install_hooks(self, model):
        """Install lightweight hooks that observe without duplicating logic.

        Unwraps PEFT so that the _augment_embeddings wrapper lives on the
        actual GraphAugmentedLLM instance (whose forward() calls
        self._augment_embeddings), not on the PeftModel wrapper.
        """
        if self._hooked:
            return
        callback = self
        inner = self._unwrap_peft(model)

        # Hook 1: capture PE norm from pe_proj output via a forward hook.
        def _pe_proj_hook(_module, _input, output):
            callback._pe_norm = output.detach().norm().item()
            callback._pe_has_nan = bool(output.detach().isnan().any())

        inner.pe_proj.register_forward_hook(_pe_proj_hook)

        # Hook 2: wrap _augment_embeddings on the actual GraphAugmentedLLM so
        # that self._augment_embeddings() inside forward() hits our wrapper.
        orig_augment = inner._augment_embeddings

        def _wrapped_augment(input_ids, graphs, injection_maps):
            # Capture base embedding norm before injection.
            with torch.no_grad():
                # Count injections directly from pre-computed injection maps.
                emb_table = inner.llm.get_input_embeddings()
                callback._emb_norm = emb_table(input_ids.to(emb_table.weight.device)).norm().item()
            total = 0
            for imap in injection_maps:
                total += sum(len(spans) for spans in imap.values())
            callback._num_injections = total
            # Call the real method — no duplicated injection logic.
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
        GraphSFTTrainer.training_step.  HF Trainer zeroes gradients before
        on_log fires, so reading .grad in on_log always returns 0.
        """
        self._captured_grad_norms["gnn"] = self._grad_norm(model.pe_model.parameters())
        self._captured_grad_norms["pe_proj"] = self._grad_norm(model.pe_proj.parameters())
        self._captured_grad_norms["pe_gain"] = self._grad_norm([model.pe_gain])
        lora_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]
        self._captured_grad_norms["lora"] = self._grad_norm(lora_params)

        if hasattr(model.pe_model, "blocks"):
            self._captured_grad_norms["rpearl"] = self._grad_norm(model.pe_model.pe_model.parameters())
            self._captured_grad_norms["gt_blocks"] = self._grad_norm(model.pe_model.blocks.parameters())
            self._captured_grad_norms["gt_output_norm"] = self._grad_norm(model.pe_model.output_norm.parameters())

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if hasattr(model, "pe_model"):
            self._install_hooks(model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not hasattr(model, "pe_model"):
            return

        lr = state.log_history[-1].get("learning_rate", float("nan")) if state.log_history else float("nan")
        metrics = {
            "debug/grad_norm_gnn": self._captured_grad_norms.get("gnn", 0.0),
            "debug/grad_norm_pe_proj": self._captured_grad_norms.get("pe_proj", 0.0),
            "debug/grad_norm_lora": self._captured_grad_norms.get("lora", 0.0),
            "debug/pe_output_norm": self._pe_norm,
            "debug/pe_has_nan": int(self._pe_has_nan),
            "debug/embedding_norm": self._emb_norm,
            "debug/num_injections": self._num_injections,
            "debug/lr": lr,
            "debug/pe_gain": model.pe_gain.item(),
            "debug/grad_norm_pe_gain": self._captured_grad_norms.get("pe_gain", 0.0),
        }

        # rpearl_gt_llm: split gradient norms by GT sub-component.
        if hasattr(model.pe_model, "blocks"):
            metrics["debug/grad_norm_rpearl"] = self._captured_grad_norms.get("rpearl", 0.0)
            metrics["debug/grad_norm_gt_blocks"] = self._captured_grad_norms.get("gt_blocks", 0.0)
            metrics["debug/grad_norm_gt_output_norm"] = self._captured_grad_norms.get("gt_output_norm", 0.0)

        if wandb.run is not None:
            wandb.log(metrics, step=state.global_step)


class AugGraphDebugCallback(TrainerCallback):
    """M11 — log augmented-graph diagnostics for the ``augmented_graph_gt`` model.

    Logs to W&B (E3/R6):
      - ``aug_graph/fiedler``     — λ₂ of the augmented Laplacian (sparse LOBPCG /
                                    eigsh; never densified). Trending → 0 means the
                                    sequence and scene layers are disconnecting.
      - ``aug_graph/scene_mass``  — fraction of a cross-linked token's k-hop mass
                                    landing on scene nodes. Collapsing → the scene
                                    is being swamped by the cycle.
      - ``grep/structural_gate``  — the M7 gate (forced up once edges are stripped).
      - ``grep/contrib_ratio``    — ‖gate·Y[V_Tx]‖ / ‖X‖, the true injected-signal
                                    energy (a scalar gate alone hides this).

    Path metrics (M10) are logged separately by ``EvalCallback``.

    Styled after ``GradientDebugCallback``: lightweight hooks capture the last
    augmented graph and the gated contribution during forward; ``on_log`` computes
    the (sparse) spectral quantities and logs. Only active for a model exposing
    the M7 ``injection`` gate (i.e. ``AugmentedGraphLLM``); a no-op otherwise.
    """

    def __init__(self, enable_visualizer: bool = False, visualizer_dir: str | None = None):
        self._last_aug = None
        self._contrib_ratio = float("nan")
        self._hooked = False
        self._enable_visualizer = enable_visualizer
        self._visualizer_dir = visualizer_dir
        self._visualized = False

    @staticmethod
    def _unwrap_peft(model):
        """Navigate PeftModel → LoraModel → AugmentedGraphLLM (which owns ``injection``)."""
        inner = model
        if hasattr(inner, "base_model"):
            inner = inner.base_model
        if hasattr(inner, "model") and hasattr(inner.model, "injection"):
            inner = inner.model
        return inner

    def _install_hooks(self, model):
        if self._hooked:
            return
        callback = self
        inner = self._unwrap_peft(model)

        # Hook 1: capture the last per-sample augmented graph so on_log can compute
        # Fiedler / scene-mass without duplicating the M4 build.
        orig_aug = inner._augmented_graph

        def _wrapped_aug(scene, injection_map, c, device, permutation=None):
            aug = orig_aug(scene, injection_map, c, device, permutation=permutation)
            callback._last_aug = aug
            return aug

        inner._augmented_graph = _wrapped_aug

        # Hook 2: contrib_ratio = ‖gate·Y[V_Tx]‖ / ‖X‖ from the gate's own inputs
        # (GatedInjection.forward(X, Y_tx)).
        def _inj_hook(module, inputs, _output):
            X, Y_tx = inputs[0], inputs[1]
            with torch.no_grad():
                num = (module.gate * Y_tx).detach().float().norm().item()
                den = X.detach().float().norm().item()
            callback._contrib_ratio = num / den if den > 0 else float("nan")

        inner.injection.register_forward_hook(_inj_hook)
        self._hooked = True

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if hasattr(model, "injection"):
            self._install_hooks(model)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not hasattr(model, "injection"):
            return
        inner = self._unwrap_peft(model)
        aug = self._last_aug

        if wandb.run is not None:
            metrics = {
                "grep/structural_gate": float(inner.injection.gate.detach().float().mean().item()),
                "grep/contrib_ratio": self._contrib_ratio,
            }
            if aug is not None:
                # Sparse solvers only (M4): fiedler() uses LOBPCG/eigsh, scene_mass()
                # uses sparse mat-mat — the N×N matrix is never densified.
                try:
                    metrics["aug_graph/fiedler"] = aug.fiedler()
                except Exception as e:
                    print(f"[M11] fiedler computation failed: {type(e).__name__}: {e}")
                try:
                    metrics["aug_graph/scene_mass"] = aug.scene_mass()
                except Exception as e:
                    print(f"[M11] scene_mass computation failed: {type(e).__name__}: {e}")
            wandb.log(metrics, step=state.global_step)

        # M12: one-shot augmented-graph + spectral-clustering render on the first
        # logging step after a graph is captured, when enabled.
        if self._enable_visualizer and not self._visualized and aug is not None:
            self._visualized = True
            try:
                from prism.eval import visualizer
                out_dir = self._visualizer_dir or (str(Path(args.output_dir) / "visuals"))
                visualizer.visualize(aug, out_dir,
                                     source=f"{Path(args.output_dir).name} @ step {state.global_step}")
            except Exception as e:
                print(f"[M12] visualizer failed: {type(e).__name__}: {e}")


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
