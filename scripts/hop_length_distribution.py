"""All-pairs shortest-path (hop) length distributions for scene-graph datasets.

Parses every graph JSON in each ``--dataset NAME=DIR`` with the pipeline's own
``safe_parse_graph``, aggregates unordered-pair BFS hop counts over the whole
dataset, and writes one histogram PNG per dataset plus a normalised comparison
PNG and a JSON summary.

Usage:
    uv run python scripts/hop_length_distribution.py \
        --dataset n_100=data/n_100/gen/nav_n100_gemma_data/test_graphs \
        --dataset n_30=data/n_30/gen/nav100_n30_gemma_data/split/test_graphs \
        --out results/hop_length_distributions
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from prism.data.utils import safe_parse_graph

# Validated categorical slots 1/2 (dataviz reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834"]
SURFACE = "#fcfcfb"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def hop_counts(graph_dir: Path) -> tuple[Counter, dict]:
    """Aggregate unordered-pair hop counts over every graph JSON in ``graph_dir``."""
    hist, meta = Counter(), {"n_graphs": 0, "nodes": [], "edges": [], "unreachable_pairs": 0}
    for path in sorted(graph_dir.glob("*.json")):
        data = json.loads(path.read_text())
        G, _ = safe_parse_graph(data.get("graph", data))
        n = G.number_of_nodes()
        reachable = 0
        for _, lengths in nx.all_pairs_shortest_path_length(G):
            for d in lengths.values():
                if d > 0:
                    hist[d] += 1  # counted twice (u,v) and (v,u)
                    reachable += 1
        meta["n_graphs"] += 1
        meta["nodes"].append(n)
        meta["edges"].append(G.number_of_edges())
        meta["unreachable_pairs"] += n * (n - 1) // 2 - reachable // 2
    return Counter({d: c // 2 for d, c in hist.items()}), meta


def summarise(hist: Counter, meta: dict) -> dict:
    hops = np.repeat(sorted(hist), [hist[d] for d in sorted(hist)])
    return {
        "n_graphs": meta["n_graphs"],
        "nodes_per_graph": {"min": min(meta["nodes"]), "max": max(meta["nodes"]),
                            "mean": float(np.mean(meta["nodes"]))},
        "edges_per_graph": {"min": min(meta["edges"]), "max": max(meta["edges"]),
                            "mean": float(np.mean(meta["edges"]))},
        "connected_pairs": int(hops.size),
        "unreachable_pairs": meta["unreachable_pairs"],
        "mean_hops": float(hops.mean()),
        "median_hops": float(np.median(hops)),
        "p95_hops": float(np.percentile(hops, 95)),
        "max_hops": int(hops.max()),
        "counts_by_hop": {int(d): int(hist[d]) for d in sorted(hist)},
    }


def _style(ax, title: str, subtitle: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.xaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set(color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2, length=0, labelsize=10)
    ax.set_xlabel("shortest-path length (hops)", color=INK_2, fontsize=10, labelpad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10, labelpad=10)
    ax.set_title(title, color=INK, fontsize=14, loc="left", pad=24, fontweight="semibold")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=INK_2, fontsize=10, va="bottom")


def plot_single(hist: Counter, stats: dict, name: str, color: str, out: Path) -> None:
    xs = sorted(hist)
    ys = [hist[d] for d in xs]
    fig, ax = plt.subplots(figsize=(8, 4.8), facecolor=SURFACE)
    ax.bar(xs, ys, width=0.82, color=color, linewidth=0)
    # Direct-label the mode only; the axis carries the rest.
    peak = int(np.argmax(ys))
    ax.annotate(f"{ys[peak]:,} pairs at {xs[peak]} hops", (xs[peak], ys[peak]),
                textcoords="offset points", xytext=(0, 8), ha="center",
                color=INK, fontsize=10)
    ax.set_xticks(xs)
    ax.set_ylim(0, max(ys) * 1.16)
    unreachable = stats["unreachable_pairs"]
    note = f" · {unreachable:,} pairs unreachable (excluded)" if unreachable else ""
    _style(ax, f"Hop-length distribution — {name} test graphs",
           f"{stats['n_graphs']} graphs · {stats['connected_pairs']:,} connected node pairs · "
           f"median {stats['median_hops']:.0f}, mean {stats['mean_hops']:.2f}, "
           f"max {stats['max_hops']}{note}",
           "node pairs")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_comparison(datasets: dict, out: Path) -> None:
    """Grouped bars on relative frequency — pair counts differ ~10x across datasets."""
    xs = sorted({d for h, _ in datasets.values() for d in h})
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURFACE)
    w = 0.82 / len(datasets)
    for i, (name, (hist, stats)) in enumerate(datasets.items()):
        total = stats["connected_pairs"]
        pos = np.array(xs) + (i - (len(datasets) - 1) / 2) * w
        ax.bar(pos, [100 * hist.get(d, 0) / total for d in xs], width=w * 0.92,
               color=SERIES[i], linewidth=0, label=f"{name}  (median {stats['median_hops']:.0f})")
    ax.set_xticks(xs)
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=10, loc="upper right")
    _style(ax, "Hop-length distribution — n_100 vs n_30 test graphs",
           "share of connected node pairs at each shortest-path length",
           "% of node pairs")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", action="append", required=True, metavar="NAME=DIR",
                   help="repeatable; label and directory of graph JSONs")
    p.add_argument("--out", required=True, type=Path, help="output directory for PNGs + summary")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results, summary = {}, {}
    for i, spec in enumerate(args.dataset):
        name, _, dir_ = spec.partition("=")
        hist, meta = hop_counts(Path(dir_))
        stats = summarise(hist, meta)
        stats["source"] = dir_
        results[name], summary[name] = (hist, stats), stats
        plot_single(hist, stats, name, SERIES[i % len(SERIES)],
                    args.out / f"hop_distribution_{name}.png")
        print(f"{name}: {stats['n_graphs']} graphs, {stats['connected_pairs']:,} pairs, "
              f"mean {stats['mean_hops']:.2f}, max {stats['max_hops']}, "
              f"unreachable {stats['unreachable_pairs']}")

    if len(results) > 1:
        plot_comparison(results, args.out / "hop_distribution_comparison.png")
    (args.out / "hop_distribution_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
