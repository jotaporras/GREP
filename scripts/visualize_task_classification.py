#!/usr/bin/env python3
"""
Visualize BERT task classification results from eval/task_classification.json.

Produces one figure per (experiment_suite, permutation) combination showing:
  - Accuracy by task type across graph sizes for each model variant
  - Formatting rate by task type
  - Overall metrics annotated
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_LABELS = ["node_existence", "position_in_map", "reachability", "navigability"]
TASK_COLORS = {
    "node_existence": "#2196F3",
    "position_in_map": "#4CAF50",
    "reachability": "#FF9800",
    "navigability": "#E91E63",
}
MODEL_MARKERS = {
    "llm": "o",
    "rpearl_llm": "s",
    "rpearl_gt_llm": "D",
}
MODEL_LABELS = {
    "llm": "LLM (no graph)",
    "rpearl_llm": "R-PEARL + LLM",
    "rpearl_gt_llm": "R-PEARL GT + LLM",
}


def parse_source_file(src: str) -> dict:
    """Extract experiment suite, model variant, permutation, and graph size from source path."""
    basename = os.path.basename(src)
    dirname = os.path.dirname(src)

    perm_match = re.search(r"perm_(\d+)", dirname)
    perm = f"perm_{perm_match.group(1)}" if perm_match else None

    size_match = re.search(r"graph_unique_(\d+?)(?:_\d+)?\.json$", basename)
    graph_size = int(size_match.group(1)) if size_match else None

    if "multi_step" in basename:
        graph_size = None

    if "rpearl_gt_llm" in basename:
        model = "rpearl_gt_llm"
    elif "rpearl_llm" in basename:
        model = "rpearl_llm"
    elif "rpearl_improvement" in basename:
        model = "rpearl_llm"
    elif "_llm_" in basename:
        model = "llm"
    else:
        model = "unknown"

    exp_match = re.match(r"(e\d+)", basename)
    exp_prefix = exp_match.group(1) if exp_match else "unknown"

    if "transferability_betty" in dirname:
        suite = f"{exp_prefix}_transferability_betty"
    elif "transferability_n100" in dirname:
        suite = f"{exp_prefix}_transferability_n100"
    elif "transferability" in dirname:
        suite = f"{exp_prefix}_transferability"
    elif dirname.startswith("results/"):
        suite = f"{exp_prefix}_baseline"
    elif dirname.startswith("shared/results/perm"):
        suite = f"{exp_prefix}_shared"
    else:
        suite = exp_prefix

    return {
        "suite": suite,
        "model": model,
        "perm": perm,
        "graph_size": graph_size,
    }


def group_results(results: list[dict]) -> dict:
    """Group results by (suite, perm) -> model -> graph_size -> metrics."""
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        meta = parse_source_file(r["source_file"])
        if meta["graph_size"] is None:
            continue
        key = (meta["suite"], meta["perm"] or "none")
        grouped[key][meta["model"]][meta["graph_size"]].append(r)
    return grouped


def aggregate_metrics(result_list: list[dict]) -> dict:
    """Average metrics across multiple runs at the same graph size."""
    if not result_list:
        return {}

    agg = {
        "overall_accuracy": np.mean([r["overall_accuracy"] for r in result_list]),
        "overall_formatted": np.mean([r["overall_formatted"] for r in result_list]),
        "accuracy_by_type": {},
        "formatted_by_type": {},
        "type_distribution": defaultdict(float),
        "n_runs": len(result_list),
    }

    for label in TASK_LABELS:
        accs = [r["accuracy_by_type"].get(label, None) for r in result_list]
        accs = [a for a in accs if a is not None]
        if accs:
            agg["accuracy_by_type"][label] = np.mean(accs)

        fmts = [r["formatted_by_type"].get(label, None) for r in result_list]
        fmts = [f for f in fmts if f is not None]
        if fmts:
            agg["formatted_by_type"][label] = np.mean(fmts)

        dists = [r["type_distribution"].get(label, 0) for r in result_list]
        agg["type_distribution"][label] = np.mean(dists)

    return agg


def make_figure(suite: str, perm: str, model_data: dict, output_dir: Path):
    """Create a single figure for one (suite, perm) combination."""
    models = sorted(model_data.keys())
    if not models:
        return

    all_sizes = set()
    for model in models:
        all_sizes.update(model_data[model].keys())
    sizes = sorted(all_sizes)
    if not sizes:
        return

    n_types = len(TASK_LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Task Classification Analysis: {suite} / {perm}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for ax_idx, task_label in enumerate(TASK_LABELS):
        ax = axes[ax_idx // 2, ax_idx % 2]

        for model in models:
            size_metrics = {}
            for sz in sizes:
                runs = model_data[model].get(sz, [])
                if runs:
                    size_metrics[sz] = aggregate_metrics(runs)

            plot_sizes = sorted(size_metrics.keys())
            if not plot_sizes:
                continue

            accuracies = []
            formatted_rates = []
            for sz in plot_sizes:
                m = size_metrics[sz]
                accuracies.append(m["accuracy_by_type"].get(task_label, np.nan))
                formatted_rates.append(m["formatted_by_type"].get(task_label, np.nan))

            marker = MODEL_MARKERS.get(model, "x")
            label_str = MODEL_LABELS.get(model, model)

            ax.plot(
                plot_sizes,
                accuracies,
                marker=marker,
                label=f"{label_str} (acc)",
                linewidth=2,
                markersize=7,
            )
            ax.plot(
                plot_sizes,
                formatted_rates,
                marker=marker,
                label=f"{label_str} (fmt)",
                linewidth=1,
                linestyle="--",
                alpha=0.5,
                markersize=5,
            )

        ax.set_title(
            task_label.replace("_", " ").title(),
            fontsize=12,
            fontweight="bold",
            color=TASK_COLORS[task_label],
        )
        ax.set_xlabel("Graph Size (nodes)")
        ax.set_ylabel("Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")

        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes], fontsize=8)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    total_samples = 0
    for model in models:
        for sz, runs in model_data[model].items():
            for r in runs:
                total_samples += r["total_samples"]

    fig.text(
        0.5, 0.01,
        f"Total samples: {total_samples} | Models: {', '.join(MODEL_LABELS.get(m, m) for m in models)} | "
        f"Solid = accuracy, Dashed = formatting rate",
        ha="center", fontsize=9, style="italic",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    safe_name = f"{suite}_{perm}".replace("/", "_")
    out_path = output_dir / f"task_classification_{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def make_distribution_figure(suite: str, perm: str, model_data: dict, output_dir: Path):
    """Create a stacked bar chart showing task type distribution across graph sizes."""
    models = sorted(model_data.keys())
    if not models:
        return

    all_sizes = set()
    for model in models:
        all_sizes.update(model_data[model].keys())
    sizes = sorted(all_sizes)
    if not sizes:
        return

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    fig.suptitle(
        f"Task Type Distribution: {suite} / {perm}",
        fontsize=13,
        fontweight="bold",
    )

    for m_idx, model in enumerate(models):
        ax = axes[0, m_idx]
        bottoms = np.zeros(len(sizes))

        for task_label in TASK_LABELS:
            counts = []
            for sz in sizes:
                runs = model_data[model].get(sz, [])
                if runs:
                    avg = np.mean([r["type_distribution"].get(task_label, 0) for r in runs])
                    counts.append(avg)
                else:
                    counts.append(0)

            ax.bar(
                range(len(sizes)),
                counts,
                bottom=bottoms,
                label=task_label.replace("_", " ").title(),
                color=TASK_COLORS[task_label],
                alpha=0.85,
            )
            bottoms += np.array(counts)

        ax.set_title(MODEL_LABELS.get(model, model), fontsize=11)
        ax.set_xticks(range(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes], fontsize=8)
        ax.set_xlabel("Graph Size")
        ax.set_ylabel("Avg Task Count")
        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    safe_name = f"{suite}_{perm}".replace("/", "_")
    out_path = output_dir / f"task_distribution_{safe_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main():
    input_path = PROJECT_ROOT / "eval" / "task_classification.json"
    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run bert_task_classifier.py first.")
        sys.exit(1)

    output_dir = PROJECT_ROOT / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_path) as f:
        data = json.load(f)

    results = data["results"]
    print(f"Loaded {len(results)} classified result files.\n")

    grouped = group_results(results)
    print(f"Found {len(grouped)} (suite, permutation) combinations:\n")

    for (suite, perm), model_data in sorted(grouped.items()):
        models = sorted(model_data.keys())
        total = sum(
            len(runs)
            for m in models
            for runs in model_data[m].values()
        )
        print(f"  {suite} / {perm}: {len(models)} models, {total} result files")
        make_figure(suite, perm, model_data, output_dir)
        make_distribution_figure(suite, perm, model_data, output_dir)

    print(f"\nAll figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
