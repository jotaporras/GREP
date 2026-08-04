import json
import warnings

import torch
import wandb
from pathlib import Path
from transformers.trainer_callback import TrainerCallback

from prism.eval import evaluate


class EvalCallback(TrainerCallback):
    """Periodic train-time eval over one OR MANY held-out graphs.

    ``eval_samples_by_graph`` maps a graph name (file stem) to its list of
    `EvalSample`s; the callback runs `evaluate.eval_model_multiple_graphs` each
    interval and reports the sample-weighted micro-average of the per-graph keyword
    accuracy plus a per-graph breakdown. A single-graph dict reproduces the old
    single-graph `eval/accuracy` (same objective-keyword metric).
    """

    # Path-validity keys that ride under the top-level `eval/` namespace.
    _EVAL_PATH_KEYS = ("valid_path_rate", "path_optimality_rate", "hallucination_rate")

    def __init__(
        self,
        eval_samples_by_graph,
        *,
        tokenizer,
        use_icl: bool,
        include_edge_list: bool,
        eval_epoch_interval: float = 1.0,
        edge_weights: str = "gaussian",
        injection_scope: str = "full_sequence",
    ):
        self.eval_samples_by_graph = eval_samples_by_graph
        self.tokenizer = tokenizer
        self.use_icl = use_icl
        self.include_edge_list = include_edge_list
        self.eval_epoch_interval = eval_epoch_interval
        # "gaussian" | "binary"; must match the train-time data.edge_weights policy.
        self.edge_weights = edge_weights
        # Train-time injection scope; "decode_consistent" arms decode-time injection.
        self.injection_scope = injection_scope
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
        results = evaluate.eval_model_multiple_graphs(
            model,
            self.tokenizer,
            self.eval_samples_by_graph,
            include_edge_list=self.include_edge_list,
            use_icl=self.use_icl,
            permutation=None,
            on_graph_done=None,
            edge_weights=self.edge_weights,
            injection_scope=self.injection_scope,
        )
        model.train()

        # Sample-weighted micro-average keyword accuracy + aggregate path-validity metrics.
        all_samples = [s for r in results.values() for s in r.samples]
        num_total = sum(r.num_total for r in results.values())
        num_correct = sum(round(r.accuracy * r.num_total) for r in results.values())
        accuracy = (num_correct / num_total) if num_total else 0.0
        path_metrics = evaluate._aggregate_path_metrics(all_samples)

        log_data = {
            "step": state.global_step,
            "epoch": state.epoch,
            "accuracy": accuracy,
            "num_graphs": len(results),
            "num_samples": num_total,
            "num_correct": num_correct,
            "path_metrics": path_metrics,
            "per_graph": {name: {"accuracy": r.accuracy, "num_correct": r.num_correct,
                                 "num_total": r.num_total}
                          for name, r in results.items()},
            "samples": all_samples,
        }
        log_file = self._eval_log_dir / f"step_{state.global_step:06d}_epoch_{state.epoch:.3f}.json"
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2, default=str)

        if wandb.run is not None:
            wandb.save(str(log_file), base_path=str(self._eval_log_dir))

        # `eval/accuracy` is the overall micro-average; per-graph rates ride under
        # `eval/acc/<stem>`. Path keys: `valid_path_rate`, `path_optimality_rate`,
        # `hallucination_rate`.
        wandb_metrics = {"eval/accuracy": accuracy, "epoch": state.epoch}
        for name, r in results.items():
            wandb_metrics[f"eval/acc/{name}"] = r.accuracy
        for k, v in path_metrics.items():
            if v is not None and k not in self._EVAL_PATH_KEYS:
                wandb_metrics[f"grep/path_{k}"] = v
        for k in self._EVAL_PATH_KEYS:
            if path_metrics.get(k) is not None:
                wandb_metrics[f"eval/{k}"] = path_metrics[k]
        if wandb.run is not None:
            wandb.log(wandb_metrics)
        self.metrics = {"eval/accuracy": accuracy}
        self.metrics.update({f"grep/path_{k}": v for k, v in path_metrics.items()
                             if v is not None and k not in self._EVAL_PATH_KEYS})
        self.metrics.update({f"eval/{k}": path_metrics[k] for k in self._EVAL_PATH_KEYS
                             if path_metrics.get(k) is not None})

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

    Covers the graph-augmented LLM family: ``GraphAugmentedLLM`` (``rpearl_llm`` /
    ``rpearl_gt_llm`` / ``gt_llm``) — grad norms for the GNN/PE, pe_proj, pe_gain,
    LoRA, and (GT variant) inner R-PEARL + GT blocks; plus pe_proj output
    magnitude — and ``LearnableGraphMaskLLM`` (PE/GT norms only).

    Grad norms are captured by ``_capture_grad_norms`` (called after backward,
    before zero_grad) — HF Trainer zeroes grads before ``on_log``, so ``.grad``
    reads 0 there.
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
        """Navigate PeftModel → LoraModel → GraphAugmentedLLM / LearnableGraphMaskLLM."""
        inner = model
        if hasattr(inner, 'base_model'):
            inner = inner.base_model
        if hasattr(inner, 'model') and (
            hasattr(inner.model, 'pe_proj')
            or hasattr(inner.model, 'pe_model')  # LearnableGraphMaskLLM: GT but no pe_proj
        ):
            inner = inner.model
        return inner

    @staticmethod
    def _supported(inner):
        """Graph-augmented family — has a PE source we can introspect."""
        return hasattr(inner, "pe_model")

    def _install_hooks(self, model):
        """Install forward hooks on the unwrapped graph-augmented instance."""
        if self._hooked:
            return
        self._install_pe_hooks(self._unwrap_peft(model))
        self._hooked = True

    def _install_pe_hooks(self, inner):
        """GraphAugmentedLLM (pe_proj output norm + ``_augment_embeddings`` wrap) OR
        LearnableGraphMaskLLM (pe_model/GT output norm only — no pe_proj, and the PE feeds
        the attention mask rather than the embeddings, so the emb-norm/injection wrap is skipped).
        """
        callback = self

        # Hook 1: capture PE norm from the pe_proj output (GraphAugmentedLLM) or, when there is
        # no pe_proj (LearnableGraphMaskLLM), from the PE/GT module output directly.
        def _pe_norm_hook(_module, _input, output):
            callback._pe_norm = output.detach().norm().item()
            callback._pe_has_nan = bool(output.detach().isnan().any())

        pe_proj = getattr(inner, "pe_proj", None)
        (pe_proj if pe_proj is not None else inner.pe_model).register_forward_hook(_pe_norm_hook)

        # Hook 2 (GraphAugmentedLLM only): wrap _augment_embeddings to capture emb_norm and
        # injection count. LearnableGraphMaskLLM has no such method, so _emb_norm/_num_injections
        # stay at their NaN/0 defaults.
        if hasattr(inner, "_augment_embeddings"):
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

        # GraphAugmentedLLM / LearnableGraphMaskLLM: pe_model (+ optional inner GT/
        # R-PEARL). pe_proj + pe_gain exist only on GraphAugmentedLLM (embedding-injection path).
        self._captured_grad_norms["gnn"] = self._grad_norm(inner.pe_model.parameters())
        if hasattr(inner, "pe_proj"):
            self._captured_grad_norms["pe_proj"] = self._grad_norm(inner.pe_proj.parameters())
        if hasattr(inner, "pe_gain"):
            self._captured_grad_norms["pe_gain"] = self._grad_norm([inner.pe_gain])
        if hasattr(inner.pe_model, "blocks"):
            self._captured_grad_norms["gt_blocks"] = self._grad_norm(inner.pe_model.blocks.parameters())
            # rpearl_gt_llm wraps an R-PEARL inside the GT; gt_llm (SemanticGraphTransformer)
            # has no inner pe_model — guard so the debug callback doesn't crash.
            if hasattr(inner.pe_model, "pe_model"):
                self._captured_grad_norms["rpearl"] = self._grad_norm(inner.pe_model.pe_model.parameters())
        elif hasattr(inner.pe_model, "semantic_gt"):
            # TwoStagePE (legacy): two stacked GTs. Reported SEPARATELY because the aggregate
            # "gnn" norm hides the failure that matters here — one half (usually the
            # probe PE GT, behind the head) receiving no gradient at all.
            self._captured_grad_norms["nav_pe_gt"] = self._grad_norm(
                inner.pe_model.pe_gt.parameters())
            self._captured_grad_norms["nav_semantic_gt"] = self._grad_norm(
                inner.pe_model.semantic_gt.parameters())

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

        # GraphAugmentedLLM (rpearl_llm / rpearl_gt_llm / gt_llm) OR LearnableGraphMaskLLM.
        metrics = {
            "debug/grad_norm_gnn": g.get("gnn", 0.0),
            "debug/grad_norm_lora": g.get("lora", 0.0),
            "debug/pe_output_norm": self._pe_norm,
            "debug/pe_has_nan": int(self._pe_has_nan),
            "debug/embedding_norm": self._emb_norm,
            "debug/num_injections": self._num_injections,
            "debug/lr": lr,
        }
        # pe_proj / pe_gain exist only on GraphAugmentedLLM (embedding-injection path);
        # LearnableGraphMaskLLM has neither.
        if hasattr(inner, "pe_proj"):
            metrics["debug/grad_norm_pe_proj"] = g.get("pe_proj", 0.0)
        if hasattr(inner, "pe_gain"):
            metrics["debug/pe_gain"] = inner.pe_gain.item()
            metrics["debug/grad_norm_pe_gain"] = g.get("pe_gain", 0.0)
        # rpearl_gt_llm / learnable_graph_mask: split gradient norms by GT sub-component.
        if hasattr(inner.pe_model, "blocks"):
            metrics["debug/grad_norm_gt_blocks"] = g.get("gt_blocks", 0.0)
            if hasattr(inner.pe_model, "pe_model"):  # inner R-PEARL
                metrics["debug/grad_norm_rpearl"] = g.get("rpearl", 0.0)
        elif hasattr(inner.pe_model, "semantic_gt"):  # TwoStagePE (PE GT + Semantic GT)
            metrics["debug/grad_norm_nav_pe_gt"] = g.get("nav_pe_gt", 0.0)
            metrics["debug/grad_norm_nav_semantic_gt"] = g.get("nav_semantic_gt", 0.0)
        # wire_llm: raw vs effective sigma, the realised angle, and whether the clamp
        # engaged. Logged (not just warned) because a clamped sigma keeps growing with
        # NO effect on the model — the parameter silently stops meaning anything while
        # loss curves still look healthy. wire/sigma_raw_max diverging from
        # wire/sigma_eff_max is the signature.
        if hasattr(inner, "wire_telemetry"):
            metrics.update(inner.wire_telemetry())

        if wandb.run is not None:
            wandb.log(metrics, step=state.global_step)
