import json
import warnings

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
    """Logs per-component gradient norms, structural-signal magnitudes, and
    injection counts to W&B.

    Architecture-aware — covers both graph-augmented LLM families:

    - ``GraphAugmentedLLM`` (``rpearl_llm`` / ``rpearl_gt_llm``): a ``pe_model``
      (R-PEARL, optionally inside a GraphTransformer), a ``pe_proj`` head, a
      scalar ``pe_gain``, and ``_augment_embeddings`` for injection. Logs grad
      norms for the GNN/PE, projection, gain, LoRA, and — for the GT variant —
      the inner R-PEARL, GT attention blocks, and GT output norm, plus the
      ``pe_proj`` output magnitude.
    - ``CompositeGraphLLM`` (``composite_graph_gt``, M4–M9): a ``gt_model``
      (GraphTransformer holding R-PEARL), the M7 ``injection`` gate, and
      ``_fuse_embeddings`` for injection. Logs grad norms for the inner R-PEARL,
      GT blocks, GT output norm, the whole GT, the gate, and LoRA, plus the GT
      output ``Y`` magnitude and the gate value.

    Hooks observe without duplicating logic. Gradient norms are captured by
    ``_capture_grad_norms`` (called from ``GraphSFTTrainer.training_step`` after
    backward, before zero_grad) so they reflect real gradients — HF Trainer
    zeroes grads before ``on_log`` fires, so reading ``.grad`` there returns 0.
    The composite-graph spectral diagnostics (Fiedler, scene-mass, contrib-ratio)
    live in ``AugGraphDebugCallback``; this callback is the gradient/magnitude
    view and the two compose on the same ``composite_graph_gt`` run.
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
        """Navigate PeftModel → LoraModel → GraphAugmentedLLM / CompositeGraphLLM."""
        inner = model
        if hasattr(inner, 'base_model'):
            inner = inner.base_model
        if hasattr(inner, 'model') and (
            hasattr(inner.model, 'pe_proj') or hasattr(inner.model, 'gt_model')
        ):
            inner = inner.model
        return inner

    @staticmethod
    def _is_augmented(inner):
        """True for the M4–M9 ``CompositeGraphLLM`` (``gt_model`` + M7 ``injection``)."""
        return hasattr(inner, "gt_model") and hasattr(inner, "injection")

    @staticmethod
    def _supported(inner):
        """Either graph-augmented family — has a PE source we can introspect."""
        return hasattr(inner, "pe_model") or hasattr(inner, "gt_model")

    def _install_hooks(self, model):
        """Install lightweight hooks that observe without duplicating logic.

        Unwraps PEFT so the injection wrapper lives on the actual graph-augmented
        instance (whose ``forward`` calls it), not on the PeftModel wrapper.
        """
        if self._hooked:
            return
        inner = self._unwrap_peft(model)
        if self._is_augmented(inner):
            self._install_augmented_hooks(inner)
        else:
            self._install_legacy_hooks(inner)
        self._hooked = True

    def _install_legacy_hooks(self, inner):
        """GraphAugmentedLLM: pe_proj output norm + ``_augment_embeddings`` wrap."""
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

    def _install_augmented_hooks(self, inner):
        """CompositeGraphLLM: GT output (Y) norm + ``_fuse_embeddings`` wrap.

        The GT output ``Y`` is the structural signal the M7 gate scales into the
        LLM, so its magnitude (and any NaN) is the analogue of the legacy
        ``pe_proj`` output. ``_fuse_embeddings`` carries the same signature used
        at train and eval (``permutation`` keyword), so the wrapper must forward it.
        """
        callback = self

        def _gt_hook(_module, _input, output):
            callback._pe_norm = output.detach().norm().item()
            callback._pe_has_nan = bool(output.detach().isnan().any())

        inner.gt_model.register_forward_hook(_gt_hook)

        orig_fuse = inner._fuse_embeddings

        def _wrapped_fuse(input_ids, graphs, injection_maps, permutation=None, **kwargs):
            with torch.no_grad():
                emb_table = inner.llm.get_input_embeddings()
                callback._emb_norm = emb_table(input_ids.to(emb_table.weight.device)).norm().item()
            callback._num_injections = sum(
                len(spans) for imap in injection_maps for spans in imap.values()
            )
            # Forward extra kwargs (e.g. return_c_tok for the c_per_layer path) untouched.
            return orig_fuse(input_ids, graphs, injection_maps, permutation=permutation, **kwargs)

        inner._fuse_embeddings = _wrapped_fuse

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

        if self._is_augmented(inner):
            # CompositeGraphLLM: GraphTransformer (R-PEARL + blocks) + M7 gate.
            gt = inner.gt_model
            self._captured_grad_norms["gt"] = self._grad_norm(gt.parameters())
            self._captured_grad_norms["rpearl"] = self._grad_norm(gt.pe_model.parameters())
            self._captured_grad_norms["gt_blocks"] = self._grad_norm(gt.blocks.parameters())
            self._captured_grad_norms["gt_output_norm"] = self._grad_norm(gt.output_norm.parameters())
            self._captured_grad_norms["gate"] = self._grad_norm([inner.injection.gate])
            return

        # Legacy GraphAugmentedLLM: pe_model (+ optional GT) + pe_proj + pe_gain.
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

    @staticmethod
    def _filter_norm_metrics(inner):
        """STEP 3: log MEASURED ‖H(S)‖₂ (power iteration on the actual S̄) per R-PEARL
        filter vs target 1/F, the ratio, and the c_bias Ĉ scale. Fail-loud: re-run the
        bound check (``gcn.strict_filter_norm`` raises; lenient warns)."""
        out = {}
        gcn = getattr(getattr(getattr(inner, "gt_model", None), "pe_model", None), "pe_gcn", None)
        if gcn is not None and getattr(gcn, "_last_graph", None) is not None:
            ei, ew, n = gcn._last_graph
            try:
                report = gcn.filter_norm_report(ei, ew, n)        # true ‖H(S)‖₂ on S̄
                worst = 0.0
                for idx, d in report.items():
                    out[f"grep/filter_norm/layer{idx}_measured"] = d["measured"]
                    out[f"grep/filter_norm/layer{idx}_target"] = d["target"]
                    out[f"grep/filter_norm/layer{idx}_ratio"] = d["ratio"]
                    worst = max(worst, d["ratio"])
                out["grep/filter_norm/worst_ratio"] = worst       # ≤ 1 + tol if invariant holds
                gcn.assert_filter_bounds(strict=getattr(gcn, "strict_filter_norm", False),
                                         report=report)
            except Exception as e:  # never let a diagnostic kill training (unless strict raised)
                if isinstance(e, AssertionError):
                    raise
                warnings.warn(f"[filter_norm] measurement skipped: {e}")
        if getattr(inner, "c_bias", False):
            try:
                # analytic Ĉ from the R-PEARL taps (deterministic; c_bias positional kernel).
                # no_grad: logging only — Ĉ is built from grad-carrying taps, so without it
                # float(...) on those tensors warns (and would build a throwaway grad graph).
                with torch.no_grad():
                    C_hat, c_row = inner._analytic_c_tok(64, next(inner.parameters()).device)
                    out["grep/c_bias/c_hat_diag"] = float(C_hat.diagonal().max())  # =1 (normalized)
                    out["grep/c_bias/c_row_min"] = float(c_row.min())             # off-peak decay
                    out["grep/c_bias/lam_c"] = float(inner.lam_c)
                    out["grep/c_bias/lam_v"] = float(inner.lam_v)
            except Exception:
                pass
        return out

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None:
            return
        inner = self._unwrap_peft(model)
        if not self._supported(inner):
            return

        lr = state.log_history[-1].get("learning_rate", float("nan")) if state.log_history else float("nan")
        g = self._captured_grad_norms

        if self._is_augmented(inner):
            # CompositeGraphLLM (composite_graph_gt): R-PEARL + GT + M7 gate.
            metrics = {
                "debug/grad_norm_lora": g.get("lora", 0.0),
                "debug/grad_norm_gt": g.get("gt", 0.0),
                "debug/grad_norm_rpearl": g.get("rpearl", 0.0),
                "debug/grad_norm_gt_blocks": g.get("gt_blocks", 0.0),
                "debug/grad_norm_gt_output_norm": g.get("gt_output_norm", 0.0),
                "debug/grad_norm_gate": g.get("gate", 0.0),
                "debug/gt_output_norm": self._pe_norm,
                "debug/gt_has_nan": int(self._pe_has_nan),
                "debug/embedding_norm": self._emb_norm,
                "debug/num_injections": self._num_injections,
                "debug/gate_value": float(inner.injection.gate.detach().float().mean().item()),
                # Learnable tanh(g) output gates (GT output + R-PEARL output): the
                # effective magnitude multiplier the model has learned on each signal.
                "debug/gt_output_gain": float(inner.gt_model.output_gain.detach().tanh().item()),
                "debug/rpearl_output_gain": float(inner.gt_model.pe_model.output_gain.detach().tanh().item()),
                "debug/lr": lr,
            }
            # β=1/F R-PEARL filter-norm invariant (measured, fail-loud) + c_bias Ĉ scale.
            metrics.update(self._filter_norm_metrics(inner))
        else:
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


class AugGraphDebugCallback(TrainerCallback):
    """M11 — log composite-graph diagnostics for the ``composite_graph_gt`` model.

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
    composite graph and the gated contribution during forward; ``on_log`` computes
    the (sparse) spectral quantities and logs. Only active for a model exposing
    the M7 ``injection`` gate (i.e. ``CompositeGraphLLM``); a no-op otherwise.
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
        """Navigate PeftModel → LoraModel → CompositeGraphLLM (which owns ``injection``)."""
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

        # Hook 1: capture the last per-sample composite graph so on_log can compute
        # Fiedler / scene-mass without duplicating the M4 build.
        orig_aug = inner._composite_graph

        def _wrapped_aug(scene, injection_map, c, device, permutation=None):
            aug = orig_aug(scene, injection_map, c, device, permutation=permutation)
            callback._last_aug = aug
            return aug

        inner._composite_graph = _wrapped_aug

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

        # M12: one-shot composite-graph + spectral-clustering render on the first
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


class LoraWarmupCallback(TrainerCallback):
    """Freeze the LLM/LoRA parameters for the first ``warmup_steps`` optimizer
    steps so the structural path (GT, R-PEARL, gate) learns first (R6).

    With the LLM frozen, the loss gradient still flows to the structural params
    through ``inputs_embeds`` (which require grad), so the GT/gate get an
    isolated learning signal before LoRA starts competing for the objective.
    This counters the observed failure where the LLM content-fits the task in
    the first ~60 steps and the structural gradients then collapse to ~0. After
    ``warmup_steps`` the LoRA params are re-enabled and resume training (the
    optimizer already holds them, so updates simply resume).
    """

    def __init__(self, warmup_steps: int):
        self.warmup_steps = int(warmup_steps)
        self._frozen_params = []
        self._restored = False

    @staticmethod
    def _unwrap_peft(model):
        inner = model
        if hasattr(inner, "base_model"):
            inner = inner.base_model
        if hasattr(inner, "model") and (
            hasattr(inner.model, "injection") or hasattr(inner.model, "pe_proj")
        ):
            inner = inner.model
        return inner

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.warmup_steps <= 0 or model is None:
            return
        inner = self._unwrap_peft(model)
        # Capture the currently-trainable LLM params (the LoRA adapters), then
        # freeze them. Saved so we can re-enable exactly these after warmup.
        self._frozen_params = [p for p in inner.llm.parameters() if p.requires_grad]
        for p in self._frozen_params:
            p.requires_grad_(False)
        print(f"[train] LoRA warmup: froze {len(self._frozen_params)} LLM tensors "
              f"for the first {self.warmup_steps} steps (structure learns first)")

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        if (self._frozen_params and not self._restored
                and state.global_step >= self.warmup_steps):
            for p in self._frozen_params:
                p.requires_grad_(True)
            self._restored = True
            print(f"[train] LoRA warmup complete at step {state.global_step}: "
                  f"re-enabled {len(self._frozen_params)} LLM tensors")


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
