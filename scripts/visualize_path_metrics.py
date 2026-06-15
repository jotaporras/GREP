#!/usr/bin/env python3
"""
Visualize path-accuracy metrics from a PRISM eval results directory.

For every graph (data_gen_NNN) in a results directory, the per-sample
``path_metrics`` block is averaged into a single rate per metric, then rendered
as:

  - a grouped bar chart of the headline path metrics per graph, plus a trailing
    "AVG" group holding the cumulative (dataset-wide) average, and a secondary-axis
    bar in every group showing the length of the longest correct path, and
  - a heatmap of every available path metric per graph.

Source selection mirrors how the eval results are graded: for each graph the
``.gemma.json`` file is used when present (Gemma judge / rescue applied),
otherwise the plain ``.json`` is used. ``.judged.json`` variants are ignored.

Usage:
    python scripts/visualize_path_metrics.py [RESULTS_DIR ...]

With no arguments every run under results/models is visualized and the figures
are written to results/e7_architecture_experiments.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "results" / "models"
OUTPUT_DIR = PROJECT_ROOT / "results" / "e7_architecture_experiments"
LONGEST_PATH_COLOR = "#00363a"  # dark teal, drawn on the secondary axis

# Metrics shown in the grouped bar chart (the headline path-validity signals).
HEADLINE_METRICS = [
    "nodes_exist_rate",
    "edge_validity_rate",
    "full_path_valid",
    "structured_correct",
    "path_rescued",
    "correct",
]

# Full set of path metrics (and their preferred display order) for the heatmap.
# ``correct`` is the answer-key correctness pulled from the sample, not from
# path_metrics; it is appended for context.
ALL_METRICS = [
    "nodes_exist_rate",
    "edge_validity_rate",
    "start_goal_ok",
    "full_path_valid",
    "waypoints_ok",
    "avoid_ok",
    "required_edges_present",
    "path_from_reasoning",
    "path_rescued",
    "structured_correct",
    "structured",
    "correct",
]

METRIC_COLORS = {
    "nodes_exist_rate": "#2196F3",
    "edge_validity_rate": "#00BCD4",
    "start_goal_ok": "#9C27B0",
    "full_path_valid": "#FF9800",
    "waypoints_ok": "#8BC34A",
    "avoid_ok": "#CDDC39",
    "required_edges_present": "#795548",
    "path_from_reasoning": "#607D8B",
    "path_rescued": "#E91E63",
    "structured_correct": "#4CAF50",
    "structured": "#9E9E9E",
    "correct": "#212121",
}


def _graph_index(path: Path) -> int:
    m = re.search(r"data_gen_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def select_graph_files(results_dir: Path) -> dict[str, Path]:
    """Map graph name -> source file, preferring .gemma.json over plain .json."""
    selected: dict[str, Path] = {}
    for f in results_dir.glob("data_gen_*.json"):
        # Skip the judged variant entirely.
        if f.name.endswith(".judged.json"):
            continue
        m = re.match(r"(data_gen_\d+)", f.name)
        if not m:
            continue
        graph = m.group(1)
        is_gemma = f.name.endswith(".gemma.json")
        current = selected.get(graph)
        # Prefer .gemma.json; only fall back to plain .json when no gemma exists.
        if current is None or is_gemma:
            if current is None or not current.name.endswith(".gemma.json"):
                selected[graph] = f
    return selected


def graph_metric_means(source_file: Path) -> tuple[dict[str, float], int]:
    """Average each path metric across all samples of one graph.

    Booleans average as a 0-1 rate; ``None`` values are ignored so metrics that
    are entirely null (e.g. cost_optimality) simply drop out. Also returns the
    length (in nodes) of the longest *correct* path found in the graph, where a
    correct path is a sample with ``full_path_valid`` true.
    """
    with open(source_file) as fh:
        data = json.load(fh)

    buckets: dict[str, list[float]] = defaultdict(list)
    longest_correct = 0
    for sample in data.get("samples", []):
        pm = sample.get("path_metrics") or {}
        for key in ALL_METRICS:
            if key == "correct":
                val = sample.get("correct")
            else:
                val = pm.get(key)
            if isinstance(val, bool):
                buckets[key].append(1.0 if val else 0.0)
            elif isinstance(val, (int, float)):
                buckets[key].append(float(val))
            # strings / None are skipped

        if pm.get("full_path_valid") is True:
            length = pm.get("num_parsed") or len(pm.get("parsed_nodes") or [])
            longest_correct = max(longest_correct, int(length))

    means = {k: float(np.mean(v)) for k, v in buckets.items() if v}
    return means, longest_correct


def collect_run(
    results_dir: Path,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, int]]:
    """Return (sorted graph names, {graph: {metric: mean}}, {graph: longest})."""
    files = select_graph_files(results_dir)
    per_graph: dict[str, dict[str, float]] = {}
    per_graph_longest: dict[str, int] = {}
    for g, p in files.items():
        means, longest = graph_metric_means(p)
        per_graph[g] = means
        per_graph_longest[g] = longest
    graphs = sorted(per_graph, key=lambda g: _graph_index(Path(g)))
    return graphs, per_graph, per_graph_longest


def make_figure(
    run_name: str,
    graphs: list[str],
    per_graph: dict,
    per_graph_longest: dict,
    output_dir: Path,
):
    if not graphs:
        print(f"  No graphs found for {run_name}; skipping.")
        return

    # Determine which metrics actually have data in this run.
    present = {m for g in graphs for m in per_graph[g]}
    headline = [m for m in HEADLINE_METRICS if m in present]
    heat_metrics = [m for m in ALL_METRICS if m in present]

    # Dataset-wide (cumulative) averages for the trailing "AVG" group.
    overall = {
        m: float(np.nanmean([per_graph[g].get(m, np.nan) for g in graphs]))
        for m in heat_metrics
    }
    # Longest correct path: per graph, plus the run-wide maximum for the AVG group.
    longest_per_graph = [per_graph_longest.get(g, 0) for g in graphs]
    longest_overall = max(longest_per_graph) if longest_per_graph else 0

    fig = plt.figure(figsize=(max(12, 1.6 * len(graphs) + 5), 11))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.32)
    fig.suptitle(
        f"Path Metrics by Graph — {run_name}",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )

    # Graphs followed by the cumulative-average group.
    groups = graphs + ["__avg__"]
    short = [g.replace("data_gen_", "g") for g in graphs] + ["AVG"]

    # ---- Panel A: grouped bar chart of headline metrics + longest path ----
    ax_bar = fig.add_subplot(gs[0])
    ax_len = ax_bar.twinx()  # secondary axis for the longest-correct-path bar
    x = np.arange(len(groups))
    n = len(headline)
    slots = n + 1  # rate metrics + one slot for the longest-path bar
    width = 0.8 / slots

    def slot_offset(i: int) -> float:
        return (i - (slots - 1) / 2) * width

    for i, metric in enumerate(headline):
        vals = [
            overall[metric] if g == "__avg__" else per_graph[g].get(metric, np.nan)
            for g in groups
        ]
        ax_bar.bar(
            x + slot_offset(i),
            vals,
            width,
            label=metric.replace("_", " "),
            color=METRIC_COLORS.get(metric, None),
            alpha=0.9,
        )

    longest_vals = longest_per_graph + [longest_overall]
    bar_len = ax_len.bar(
        x + slot_offset(n),
        longest_vals,
        width,
        label="longest correct path",
        color=LONGEST_PATH_COLOR,
        alpha=0.95,
        hatch="//",
        edgecolor="white",
        linewidth=0.4,
    )
    for xi, v in zip(x + slot_offset(n), longest_vals):
        if v > 0:
            ax_len.text(xi, v, f"{int(v)}", ha="center", va="bottom", fontsize=7.5,
                        color=LONGEST_PATH_COLOR, fontweight="bold")

    # Divider separating the per-graph groups from the cumulative average.
    ax_bar.axvline(len(graphs) - 0.5, color="0.4", linestyle="--", linewidth=1)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(short, fontsize=9)
    ax_bar.set_ylabel("Average rate")
    ax_bar.set_ylim(0, 1.08)
    ax_len.set_ylabel("Longest correct path (nodes)", color=LONGEST_PATH_COLOR)
    ax_len.tick_params(axis="y", labelcolor=LONGEST_PATH_COLOR)
    ax_len.set_ylim(0, max(longest_overall * 1.25, 1))
    ax_bar.set_title(
        "Headline path metrics (per graph + cumulative average)",
        fontsize=11, fontweight="bold",
    )
    ax_bar.grid(axis="y", alpha=0.3)

    handles, labels = ax_bar.get_legend_handles_labels()
    handles.append(bar_len)
    labels.append("longest correct path")
    ax_bar.legend(handles, labels, fontsize=8, ncol=min(slots, 3),
                  loc="upper right", framealpha=0.9)

    # ---- Panel B: heatmap of all available metrics (inverted colormap) ----
    ax_heat = fig.add_subplot(gs[1])
    matrix = np.array(
        [[per_graph[g].get(m, np.nan) for g in graphs] for m in heat_metrics],
        dtype=float,
    )
    im = ax_heat.imshow(matrix, aspect="auto", cmap="viridis_r", vmin=0.0, vmax=1.0)

    ax_heat.set_xticks(np.arange(len(graphs)))
    ax_heat.set_xticklabels([g.replace("data_gen_", "g") for g in graphs], fontsize=9)
    ax_heat.set_yticks(np.arange(len(heat_metrics)))
    ax_heat.set_yticklabels([m.replace("_", " ") for m in heat_metrics], fontsize=9)
    ax_heat.set_title("All path metrics (per-graph average)", fontsize=11, fontweight="bold")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            if np.isnan(v):
                continue
            ax_heat.text(
                c, r, f"{v:.2f}",
                ha="center", va="center", fontsize=7.5,
                color="white" if v >= 0.5 else "black",
            )

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.015)
    cbar.set_label("Average rate", fontsize=9)

    # Footer: dataset-wide means.
    summary = "  ".join(
        f"{m.replace('_', ' ')}={overall[m]:.2f}"
        for m in ("full_path_valid", "structured_correct", "correct")
        if m in overall
    )
    fig.text(
        0.5, 0.012,
        f"{len(graphs)} graphs  |  source: .gemma.json where available, else .json  |  "
        f"longest correct path (run): {longest_overall}  |  dataset mean — {summary}",
        ha="center", fontsize=9, style="italic",
    )

    out_path = output_dir / f"path_metrics_{run_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    if sys.argv[1:]:
        dirs = [Path(a) for a in sys.argv[1:]]
    else:
        dirs = sorted(d for d in MODELS_DIR.iterdir() if d.is_dir())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for results_dir in dirs:
        if not results_dir.exists():
            print(f"ERROR: {results_dir} not found; skipping.")
            continue
        run_name = results_dir.name
        print(f"Processing {results_dir} ...")
        graphs, per_graph, per_graph_longest = collect_run(results_dir)
        make_figure(run_name, graphs, per_graph, per_graph_longest, OUTPUT_DIR)


if __name__ == "__main__":
    main()
