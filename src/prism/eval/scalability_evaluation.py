"""Post-hoc planning-eval driver for a trained PRISM checkpoint.

One script, two modes, switched by whether `--permutation-seed` is given:

* **Without seeds** (post-hoc cross_eval): writes one per-graph JSON to
  `<checkpoint>/eval_logs/cross_eval/<graph>.json` (or `--output`). Matches
  the layout consumed by `eval_viewer.html` and the judge-eval skill.

* **With seeds** (transferability sweep): for each seed, writes
  `<output>/perm_<seed>/<ckpt>_<graph>.json` per graph plus a
  `<ckpt>_summary.json`. Matches the layout consumed by the transferability
  summarisers and reports.

Both modes share the same loading + scoring path through
`prism.eval.evaluate.eval_model_multiple_graphs`. Output formatting is the
only branch.

Usage:
    python -m prism.eval.scalability_evaluation \
        --checkpoint outputs/.../e4_llm_..._xriur1bi \
        --graphs data/eval/e4_transferability/ \
        [--permutation-seed 32 42 58] \
        [--output results/e4_transferability/] \
        [--four-bit] [--device 0] [--text-edge-list present|none] \
        [--use-icl true|false]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


# CUDA_VISIBLE_DEVICES must be set before torch is imported by downstream
# modules; do an early --device parse to honour that.
_early_parser = argparse.ArgumentParser(add_help=False)
_early_parser.add_argument("--device", type=int, default=0)
_early_args, _ = _early_parser.parse_known_args()
if _early_args.device >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_early_args.device)


from prism.data import data
from prism.eval import checkpoint
from prism.eval import evaluate
from prism.models import utils as model_utils

# Checkpoint discovery + loading now lives in prism.eval.checkpoint (shared with the
# in-process post-train eval). These module-level aliases keep the historical private
# names resolvable for `main()` and as monkeypatch targets in the test suite.
_is_gnn_checkpoint = checkpoint.is_gnn_checkpoint
_resolve_text_edge_list = checkpoint.resolve_text_edge_list
_resolve_edge_weights = checkpoint.resolve_edge_weights
_resolve_injection_scope = checkpoint.resolve_injection_scope
_load_checkpoint = checkpoint.load_checkpoint


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Exactly one eval subject: an LLM checkpoint OR a NavigatorGT config.
    subject = p.add_mutually_exclusive_group(required=True)
    subject.add_argument("--checkpoint",
                   help="Checkpoint dir (run dir or subdir with adapter_config.json / train_config.json).")
    subject.add_argument("--navigator-config",
                   help="YAML config for a NavigatorGT eval (e.g. experiments/e9_navigator_gt.yaml). "
                        "When set, evaluates the GNN navigator instead of an LLM checkpoint: the route "
                        "is generated directly from explicit start/goal nodes.")
    p.add_argument("--graphs", required=True,
                   help="A test-graph JSON file, a directory of them, or a glob pattern.")
    p.add_argument("--permutation-seed", type=int, nargs="+", default=None,
                   help="One or more node-index permutation seeds. Omit to disable; "
                        "presence of this flag switches to transferability-style output layout.")
    p.add_argument("--output", default=None,
                   help="Output directory. Default: <checkpoint>/eval_logs/cross_eval/ "
                        "(without seeds) or results/ (with seeds).")
    p.add_argument("--four-bit", action="store_true", default=False,
                   help="Load model with 4-bit quantisation.")
    p.add_argument("--device", type=int, default=0,
                   help="Physical GPU index (default 0). Use -1 for device_map='auto'.")
    p.add_argument("--text-edge-list", choices=["present", "none"], default=None,
                   help="Override the textual-edge-list mode. Default: read from "
                        "train_config.json (legacy graph checkpoints: gnn_config.json) "
                        "for plain LLMs. Required if a plain-LLM checkpoint has neither.")
    p.add_argument("--use-icl", choices=["true", "false"], default="true",
                   help="Include SPINE in-context-learning examples in the planner prompt. "
                        "Default: true (matches historical behavior).")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------
# Output writers
# ----------------------------------------------------------------------------

def _write_cross_eval_result(
    result: evaluate.GraphEvalResultSummary,
    *,
    out_dir: str,
    checkpoint: str,
    graph_file: str,
    architecture: str,
    text_edge_list: str,
) -> str:
    """`<out_dir>/<graph>.json` shape used by eval_viewer.html and judge-eval."""
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
    out_file = os.path.join(out_dir, f"{result.name}.json")
    with open(out_file, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    return out_file


def _write_seeded_result(
    result: evaluate.GraphEvalResultSummary,
    *,
    out_dir: str,
    ckpt_name: str,
    graph_file: str,
) -> tuple[str, dict]:
    """`<out_dir>/<ckpt>_<graph>.json` shape used by the transferability summariser.

    Returns (out_path, trial_record) so the caller can aggregate the
    no-`samples` projection into the `<ckpt>_summary.json`.
    """
    trial_record = {
        "name": f"{result.name}.json",
        "path": graph_file,
        "num_total": result.num_total,
        "num_correct": result.num_correct,
        "accuracy": result.accuracy,
        "num_formatted": result.num_formatted,
        "num_keyword": result.num_keyword,
        "num_errors": result.num_errors,
        "elapsed_s": result.elapsed_s,
        "permutation": result.permutation,
        "samples": result.samples,
    }
    out_path = os.path.join(out_dir, f"{ckpt_name}_{result.name}.json")
    with open(out_path, "w") as f:
        json.dump(trial_record, f, indent=2, cls=_NumpyEncoder)
    return out_path, trial_record


def _write_seed_summary(
    *,
    out_dir: str,
    ckpt_name: str,
    checkpoint: str,
    graphs_arg: str,
    permutation,
    trial_records: list[dict],
) -> str:
    summary_path = os.path.join(out_dir, f"{ckpt_name}_summary.json")
    summary = {
        "checkpoint": checkpoint,
        "pattern": graphs_arg,
        "permutation": permutation.to_dict() if permutation is not None else None,
        "trials": [{k: v for k, v in r.items() if k != "samples"} for r in trial_records],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=_NumpyEncoder)
    return summary_path


# ----------------------------------------------------------------------------
# Per-graph progress printer (shared by both modes)
# ----------------------------------------------------------------------------

def _make_progress_printer(samples_by_graph: dict[str, list[evaluate.EvalSample]]):
    """Returns a closure to pass as `on_graph_done` to `evaluate.eval_model_multiple_graphs`."""
    grand_correct = [0]
    grand_total = [0]
    total_graphs = len(samples_by_graph)
    graph_idx = [0]

    def _on_done(name: str, result: evaluate.GraphEvalResultSummary) -> None:
        graph_idx[0] += 1
        grand_correct[0] += result.num_correct
        grand_total[0] += result.num_total
        overall_acc = grand_correct[0] / grand_total[0] if grand_total[0] else 0.0
        print(
            f"[{graph_idx[0]}/{total_graphs}] {name}: "
            f"{result.num_correct}/{result.num_total} ({result.accuracy:.1%}) "
            f"in {result.elapsed_s:.1f}s  |  "
            f"overall: {overall_acc:.1%} ({grand_correct[0]}/{grand_total[0]})"
        )

    return _on_done


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def _load_navigator(config_path: str):
    """Build a NavigatorGT + its eval policy from a YAML config (see experiments/e9_navigator_gt.yaml).

    Returns ``(model, edge_weights)``. The model is placed on CUDA when available (the sparse
    CSR attention kernels are CPU/CUDA-only — MPS is unsupported), else CPU.
    """
    import torch
    import yaml
    from prism.models.gt import NavigatorGT

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["navigator"]
    model = NavigatorGT.from_pretrained(
        cfg["weights"], gt_kwargs=cfg["gt"], semantic_kwargs=cfg["semantic"],
        max_length=cfg.get("max_length", 128))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, cfg.get("edge_weights", "binary")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    # The mutually-exclusive required group guarantees exactly one of these is set.
    is_navigator = args.navigator_config is not None

    samples_by_graph, graph_file_by_name = data.load_samples_by_graph(args.graphs)

    if is_navigator:
        # NavigatorGT: no LLM checkpoint. The route is generated from explicit start/goal
        # nodes; text_edge_list / injection_scope are LLM-only and unused here.
        checkpoint = os.path.abspath(args.navigator_config)
        ckpt_name = os.path.splitext(os.path.basename(checkpoint))[0]
        text_edge_list, injection_scope = "none", "full_sequence"
        print(f"Loading navigator: {checkpoint}")
        model, edge_weights = _load_navigator(args.navigator_config)
        tokenizer = None
        architecture = "navigator-gt"
    else:
        checkpoint = os.path.abspath(args.checkpoint.rstrip("/"))
        ckpt_name = os.path.basename(checkpoint)
        is_gnn = _is_gnn_checkpoint(checkpoint)
        text_edge_list = _resolve_text_edge_list(checkpoint, is_gnn, args.text_edge_list)
        edge_weights = _resolve_edge_weights(checkpoint)
        injection_scope = _resolve_injection_scope(checkpoint)
        print(f"Loading checkpoint: {checkpoint}")
        model, tokenizer, _ = _load_checkpoint(checkpoint, four_bit=args.four_bit, device=args.device)
        architecture = "graph-augmented" if is_gnn else "llm"
    print(f"  architecture: {architecture}  |  text_edge_list={text_edge_list}  |  "
          f"edge_weights={edge_weights}  |  injection_scope={injection_scope}  |  4bit={args.four_bit}")
    print(f"  {len(samples_by_graph)} graph file(s)\n")

    use_icl = args.use_icl == "true"
    permutations = (
        [model_utils.Permutation(s) for s in args.permutation_seed]
        if args.permutation_seed else [None]
    )
    has_seeds = args.permutation_seed is not None

    # Default output dir depends on mode. The navigator has no checkpoint dir to nest under,
    # so its default cross-eval output goes to results/<config-stem>/.
    if not has_seeds and not is_navigator:
        default_output = os.path.join(checkpoint, "eval_logs", "cross_eval")
    elif not has_seeds:
        default_output = os.path.join("results", ckpt_name)
    else:
        default_output = "results"
    base_output = args.output or default_output

    for permutation in permutations:
        if permutation is not None:
            out_dir = os.path.join(base_output, f"perm_{permutation.seed}")
            print(f"\n{'#'*60}\n  PERMUTATION SEED: {permutation.seed}\n{'#'*60}")
        else:
            out_dir = base_output
        os.makedirs(out_dir, exist_ok=True)

        progress = _make_progress_printer(samples_by_graph)
        results = evaluate.eval_model_multiple_graphs(
            model,
            tokenizer,
            samples_by_graph,
            include_edge_list=(text_edge_list == "present"),
            use_icl=use_icl,
            permutation=permutation,
            on_graph_done=progress,
            edge_weights=edge_weights,
            injection_scope=injection_scope,
        )

        if has_seeds:
            trial_records: list[dict] = []
            for name, result in results.items():
                out_path, trial_record = _write_seeded_result(
                    result,
                    out_dir=out_dir,
                    ckpt_name=ckpt_name,
                    graph_file=graph_file_by_name[name],
                )
                trial_records.append(trial_record)
                print(f"  Saved: {out_path}")
            summary_path = _write_seed_summary(
                out_dir=out_dir,
                ckpt_name=ckpt_name,
                checkpoint=args.checkpoint,
                graphs_arg=args.graphs,
                permutation=permutation,
                trial_records=trial_records,
            )
            print(f"\nSummary saved to {summary_path}")
        else:
            for name, result in results.items():
                out_file = _write_cross_eval_result(
                    result,
                    out_dir=out_dir,
                    checkpoint=checkpoint,
                    graph_file=graph_file_by_name[name],
                    architecture=architecture,
                    text_edge_list=text_edge_list,
                )
                print(f"  Saved: {out_file}")

        # Path-metrics figure for this run into <out_dir>/visuals/. ckpt_name carries
        # the trailing WandB tag; architecture is the graph-augmented/llm kind.
        fig_path = evaluate.render_path_metrics_figure(
            results, out_dir, ckpt_name, architecture=architecture)
        if fig_path:
            print(f"  Figure: {fig_path}")

        evaluate.print_summary_table(list(results.values()))


if __name__ == "__main__":
    main()
