"""Path-metrics figure renderer for PRISM planning eval.

Turns per-graph eval results into a single publication-style figure:

  - a grouped bar chart of the headline path metrics per graph, plus a trailing
    "Overall" group holding the dataset-wide mean, and a secondary-axis bar in
    every group showing the length of the longest valid path, and
  - a heatmap of every surfaced path metric per graph.

Two entry points share one rendering core (:func:`make_figure`):

  * :func:`render_from_samples` — in-process: feed live
    ``GraphEvalResultSummary.samples`` straight from ``prism.eval.evaluate`` /
    ``scalability_evaluation`` (no JSON round-trip). This is what the eval drivers
    call, writing into ``<output>/visuals/``.
  * :func:`render_dir` — on-disk: read the per-graph ``<graph>.json`` files a run
    already exported. This is what ``scripts/visualize_path_metrics.py`` calls.

The model architecture and run tag (the WandB id parsed from the run name) are
shown as a discrete badge in the figure corner.
"""

from __future__ import annotations

import json
import re
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Publication-style defaults: serif type, restrained spines, consistent sizing.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
})

LONGEST_PATH_COLOR = "#00363a"  # dark teal, drawn on the secondary axis

# Every metric below is a per-sample key whose mean is a metric the eval modules
# actually surface to the user (evaluate.print_summary_table PATH-VALIDITY block +
# callbacks.EvalCallback eval/* and grep/path_* logging, both fed by
# path_validator.aggregate_path_metrics). Per-sample keys map to run-level reports:
#   edge_validity_rate     -> edge_validity_rate
#   hallucination_rate     -> hallucination_rate          (eval/*)
#   valid_path_ab          -> valid_path_rate             (eval/*)
#   path_from_reasoning    -> num_from_reasoning
#   path_rescued           -> num_rescued
#   waypoints_ok           -> waypoints_ok_rate
#   avoid_ok               -> avoid_ok_rate
#   required_edges_present -> required_edges_rate
#   structured_correct     -> structured_pass_rate
#   llm_judge_pass         -> llm_judge_accuracy
#   correct (sample-level) -> accuracy / num_correct
# Non-[0,1] optimality ratios (cost_optimality, hop_optimality/path_optimality_rate,
# both >= 1) are surfaced too but excluded here — they break the shared [0,1] scale.
# hallucination_rate is path_validator's exact complement of edge_validity_rate
# (= 1 - edge_validity_rate over routes with >= 1 hop) and is the ONLY lower-is-better
# quantity shown — it is included alongside Edge Validity as requested.
# Deprecated per-sample keys that the eval modules NO LONGER aggregate or report
# (nodes_exist_rate, full_path_valid, start_goal_ok, the `structured` flag) are not
# shown.

# Metrics shown in the grouped bar chart (the headline path-validity signals).
HEADLINE_METRICS = [
    "edge_validity_rate",
    "hallucination_rate",
    "valid_path_ab",
    "structured_correct",
    "correct",
]

# Full set of surfaced path metrics (and their preferred display order) for the heatmap.
# ``correct`` is the answer-key correctness pulled from the sample top-level, not from
# path_metrics; it is appended for context.
ALL_METRICS = [
    "edge_validity_rate",
    "hallucination_rate",
    "valid_path_ab",
    "path_from_reasoning",
    "path_rescued",
    "waypoints_ok",
    "avoid_ok",
    "required_edges_present",
    "structured_correct",
    "llm_judge_pass",
    "correct",
]

# Formal, publication-style display names (Title Case) for every surfaced metric.
# The correctness labels are qualified by scope so the figure reads as the
# strictness hierarchy edge_validity ⊂ valid_path_ab ⊂ structured_correct, with
# `correct` the broadest (defined for every sample, subsumes structured_correct).
METRIC_LABELS = {
    "edge_validity_rate": "Edge Validity (Per-Hop)",
    "hallucination_rate": "Hallucination Rate (↓)",
    "valid_path_ab": "Valid Path (Start→Goal)",
    "path_from_reasoning": "Path From Reasoning",
    "path_rescued": "Path Rescued",
    "waypoints_ok": "Waypoints Satisfied",
    "avoid_ok": "Avoid-Set Satisfied",
    "required_edges_present": "Required Edges Present",
    "structured_correct": "Structured Correct (All Constraints)",
    "llm_judge_pass": "Subjective Accuracy (Judge)",
    "correct": "Objective Accuracy (Overall)",
}

METRIC_COLORS = {
    "edge_validity_rate": "#00BCD4",
    "hallucination_rate": "#F44336",
    "valid_path_ab": "#3F51B5",
    "path_from_reasoning": "#607D8B",
    "path_rescued": "#E91E63",
    "waypoints_ok": "#8BC34A",
    "avoid_ok": "#CDDC39",
    "required_edges_present": "#795548",
    "structured_correct": "#4CAF50",
    "llm_judge_pass": "#9C27B0",
    "correct": "#212121",
}


def _graph_index(path: Path) -> int:
    m = re.search(r"data_gen_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def _metric_label(metric: str) -> str:
    """Formal Title-Case display name for a metric key."""
    return METRIC_LABELS.get(metric, metric.replace("_", " ").title())


def _graph_label(graph: str) -> str:
    """Compact axis label for a graph, e.g. ``data_gen_004`` -> ``G4``."""
    idx = _graph_index(Path(graph))
    return f"G{idx}" if idx >= 0 else graph


# A WandB run id is the trailing token: short, alphanumeric, mixing letters+digits.
_WANDB_ID = re.compile(r"^(?=.*[a-z])(?=.*\d)[a-z0-9]{6,12}$")


def _split_run(run_name: str) -> tuple[str, str | None]:
    """Split a run dir name into (model_name, wandb_id).

    Eval checkpoints are named ``<model>_..._<wandbid>`` (e.g. ``e4_llm_..._xriur1bi``);
    the trailing token is the WandB run id when it looks like one. Returns
    ``(run_name, None)`` when no id-shaped suffix is present.
    """
    parts = run_name.split("_")
    if len(parts) > 1 and _WANDB_ID.match(parts[-1]):
        return "_".join(parts[:-1]), parts[-1]
    return run_name, None


def _run_label(run_name: str) -> str:
    """Capitalised model name with the WandB id in parentheses, e.g.
    ``COMPOSITE GRAPH GT (3xk9p2af)``."""
    model, wandb_id = _split_run(run_name)
    label = model.replace("_", " ").upper()
    return f"{label} ({wandb_id})" if wandb_id else label


def _arch_tag_badge(run_name: str, architecture: str | None) -> str:
    """Discrete ``<arch> · <wandb-tag>`` provenance string for the figure corner.

    Pairs the model architecture (when the caller supplies it) with the run tag —
    the WandB id parsed out of ``run_name``. Either part may be absent.
    """
    _, tag = _split_run(run_name)
    bits = [str(architecture)] if architecture else []
    if tag:
        bits.append(tag)
    return "  ·  ".join(bits)


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


def graph_metric_means_from_samples(
    samples: list[dict],
) -> tuple[dict[str, float], int]:
    """Average each path metric across one graph's per-sample dicts.

    ``samples`` is the ``GraphEvalResultSummary.samples`` list (the same objects
    written verbatim into the cross-eval JSONs), so this is the single source of
    truth for both the on-disk and in-memory entry points.

    Booleans average as a 0-1 rate; ``None`` values are ignored so metrics that
    are entirely null (e.g. an unjudged ``llm_judge_pass``) simply drop out. Also
    returns the length (in nodes) of the longest *valid* path found in the graph,
    where a valid path is a sample with ``valid_path_ab`` true (the per-sample
    source of the surfaced ``valid_path_rate``).
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    longest_correct = 0
    for sample in samples:
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

        if pm.get("valid_path_ab") is True:
            length = pm.get("num_parsed") or len(pm.get("parsed_nodes") or [])
            longest_correct = max(longest_correct, int(length))

    means = {k: float(np.mean(v)) for k, v in buckets.items() if v}
    return means, longest_correct


def graph_metric_means(source_file: Path) -> tuple[dict[str, float], int]:
    """Disk wrapper for :func:`graph_metric_means_from_samples` (reads one JSON)."""
    with open(source_file) as fh:
        data = json.load(fh)
    return graph_metric_means_from_samples(data.get("samples", []))


def _dir_architecture(results_dir: Path) -> str | None:
    """Read the ``architecture`` field from the run's first per-graph JSON."""
    for f in select_graph_files(results_dir).values():
        try:
            with open(f) as fh:
                return json.load(fh).get("architecture")
        except Exception:
            return None
    return None


def collect_from_samples(
    samples_by_graph: dict[str, list[dict]],
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, int]]:
    """In-memory variant of :func:`collect_run`: average pre-loaded eval samples.

    ``samples_by_graph`` maps graph name -> its per-sample dict list. Returns the
    same ``(sorted graph names, {graph: {metric: mean}}, {graph: longest})`` triple.
    """
    per_graph: dict[str, dict[str, float]] = {}
    per_graph_longest: dict[str, int] = {}
    for g, samples in samples_by_graph.items():
        means, longest = graph_metric_means_from_samples(samples)
        per_graph[g] = means
        per_graph_longest[g] = longest
    graphs = sorted(per_graph, key=lambda g: _graph_index(Path(g)))
    return graphs, per_graph, per_graph_longest


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


def render_from_samples(
    run_name: str,
    samples_by_graph: dict[str, list[dict]],
    output_dir,
    *,
    architecture: str | None = None,
):
    """Render the figure straight from in-memory eval samples (no JSON round-trip).

    ``samples_by_graph`` maps graph name -> ``GraphEvalResultSummary.samples``;
    ``architecture`` is stamped into the figure's provenance badge. Writes
    ``<output_dir>/path_metrics_<run_name>.png`` and returns its Path, or None when
    no graph carried samples. The eval drivers call this with
    ``output_dir = <results>/visuals``.
    """
    output_dir = Path(output_dir)
    graphs, per_graph, per_graph_longest = collect_from_samples(samples_by_graph)
    if not graphs:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    return make_figure(run_name, graphs, per_graph, per_graph_longest, output_dir,
                       architecture=architecture)


def render_dir(results_dir, output_dir, run_name: str | None = None):
    """Render one figure from a run's exported per-graph JSONs (CLI entry point).

    Reads the ``architecture`` from the JSONs for the provenance badge; ``run_name``
    defaults to the results-dir name (which usually carries the ``_<wandbid>`` tag).
    """
    results_dir = Path(results_dir)
    graphs, per_graph, per_graph_longest = collect_run(results_dir)
    if not graphs:
        print(f"  No graphs found in {results_dir}; skipping.")
        return None
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return make_figure(run_name or results_dir.name, graphs, per_graph,
                       per_graph_longest, output_dir,
                       architecture=_dir_architecture(results_dir))


def make_figure(
    run_name: str,
    graphs: list[str],
    per_graph: dict,
    per_graph_longest: dict,
    output_dir: Path,
    *,
    architecture: str | None = None,
):
    if not graphs:
        print(f"  No graphs found for {run_name}; skipping.")
        return None

    # Determine which metrics actually have data in this run.
    present = {m for g in graphs for m in per_graph[g]}
    headline = [m for m in HEADLINE_METRICS if m in present]
    heat_metrics = [m for m in ALL_METRICS if m in present]

    # Dataset-wide (cumulative) averages for the trailing "Overall" group.
    overall = {
        m: float(np.nanmean([per_graph[g].get(m, np.nan) for g in graphs]))
        for m in heat_metrics
    }
    # Longest valid path: per graph, plus the run-wide maximum for the Overall group.
    longest_per_graph = [per_graph_longest.get(g, 0) for g in graphs]
    longest_overall = max(longest_per_graph) if longest_per_graph else 0

    fig_w = max(12, 1.6 * len(graphs) + 5)
    fig = plt.figure(figsize=(fig_w, 11))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.34)
    fig.suptitle(
        "Planner Path-Validity and Answer Correctness Across Evaluation Graphs",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )

    # Discrete provenance badge (model arch · run tag) in the top-right corner.
    badge = _arch_tag_badge(run_name, architecture)
    if badge:
        fig.text(
            0.995, 0.997, badge, ha="right", va="top",
            fontsize=9, family="monospace", color="#37474F",
            bbox=dict(boxstyle="round,pad=0.35", fc="#ECEFF1",
                      ec="#90A4AE", lw=0.8, alpha=0.9),
        )

    # Graphs followed by the dataset-mean group.
    groups = graphs + ["__avg__"]
    short = [_graph_label(g) for g in graphs] + ["Overall"]

    # ---- Panel A: grouped bar chart of headline metrics + longest path ----
    ax_bar = fig.add_subplot(gs[0])
    ax_len = ax_bar.twinx()  # secondary axis for the longest-valid-path bar
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
            label=_metric_label(metric),
            color=METRIC_COLORS.get(metric, None),
            alpha=0.9,
        )

    longest_vals = longest_per_graph + [longest_overall]
    bar_len = ax_len.bar(
        x + slot_offset(n),
        longest_vals,
        width,
        label="Longest Valid Path",
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
    ax_bar.set_xticklabels(short)
    ax_bar.set_ylabel("Mean Rate")
    ax_bar.set_ylim(0, 1.08)
    ax_len.set_ylabel("Longest Valid Path (Nodes)", color=LONGEST_PATH_COLOR)
    ax_len.tick_params(axis="y", labelcolor=LONGEST_PATH_COLOR)
    ax_len.set_ylim(0, max(longest_overall * 1.25, 1))
    ax_bar.set_title("(a) Headline Metrics — Per Graph and Dataset Mean")
    ax_bar.grid(axis="y", alpha=0.3)

    handles, labels = ax_bar.get_legend_handles_labels()
    handles.append(bar_len)
    labels.append("Longest Valid Path")
    ax_bar.legend(handles, labels, ncol=min(slots, 3),
                  loc="upper right", framealpha=0.9)

    # ---- Panel B: heatmap of all available metrics ----
    ax_heat = fig.add_subplot(gs[1])
    matrix = np.array(
        [[per_graph[g].get(m, np.nan) for g in graphs] for m in heat_metrics],
        dtype=float,
    )
    im = ax_heat.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    ax_heat.set_xticks(np.arange(len(graphs)))
    ax_heat.set_xticklabels([_graph_label(g) for g in graphs])
    ax_heat.set_yticks(np.arange(len(heat_metrics)))
    ax_heat.set_yticklabels([_metric_label(m) for m in heat_metrics])
    ax_heat.set_title("(b) All Surfaced Metrics — Per-Graph Mean")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            v = matrix[r, c]
            if np.isnan(v):
                continue
            ax_heat.text(
                c, r, f"{v:.2f}",
                ha="center", va="center", fontsize=7.5,
                # viridis: low values render dark (white text), high values bright (black text).
                color="white" if v < 0.5 else "black",
            )

    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.015)
    cbar.set_label("Mean Rate")

    # Formal, paper-style caption defining each headline quantity.
    n_graphs = len(graphs)
    last = _graph_label(graphs[-1]) if graphs else "G0"
    caption = (
        f"Figure 1.  Deterministic path-validity and answer correctness of the "
        f"{_run_label(run_name)} planner over {n_graphs} held-out evaluation graphs "
        f"(G1–{last}), with the dataset mean (Overall).  (a) Headline metrics per "
        f"graph; hatched bars (right axis) report the longest graph-valid A→B route "
        f"recovered (run maximum: {longest_overall} nodes).  (b) Per-graph means for the "
        f"full surfaced metric set.  Edge Validity: fraction of emitted route hops that "
        f"are real graph edges.  Hallucination Rate (↓): its complement, 1 − Edge "
        f"Validity over routes with ≥1 hop.  Valid Path (A→B): fraction of reachability "
        f"/ navigability tasks solved by a graph-valid route from start to goal.  "
        f"Structured Correct: deterministic NetworkX verdict on structural tasks.  "
        f"Objective Accuracy: judge-free headline correctness.  Subjective Accuracy "
        f"(Judge): the separate Gemma-judge verdict over judged samples.  All quantities "
        f"are means in [0, 1]; higher is better except Hallucination Rate (↓)."
    )
    wrap_width = max(80, int(fig_w / 0.085))
    fig.text(
        0.5, 0.0, "\n".join(textwrap.wrap(caption, wrap_width)),
        ha="center", va="top", fontsize=9,
    )

    out_path = output_dir / f"path_metrics_{run_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path
