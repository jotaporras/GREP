"""Render the M4 composite graph (directed cycle ⊕ scene ⊕ cross-links) from a
scene-graph JSON, via prism.eval.visualizer.

Models scripts/render_scene_graph.py, but instead of drawing the bare scene
graph it (1) parses the scene graph, (2) assembles the composite graph with a
directed token cycle and synthetic cross-links (since no tokenized prompt is
provided here, each scene label gets a few evenly-spread "mentions" on the
cycle, so the E2 mention→node and multi-mention clique edges are exercised),
then (3) writes the two visualizer artifacts:

  • composite_graph.html             — interactive vis.js (dark-green cycle ring,
                                        gold mention↔node cross-links, purple
                                        same-label cliques, red scene edges)
  • composite_graph_spectral.png     — Fiedler spectral clustering, colored edges

Usage:
    python scripts/render_composite_graph.py --graph data/grep_training_data/graphs/data_gen_000.json
    python scripts/render_composite_graph.py --graph data/eval/eval_1_multi_step.json \
        --out outputs/visuals/aug_eval1 --cycle 256 --mentions 2 --open
"""

import argparse
import os
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch  # noqa: E402

from prism.data.utils import scene_graph_dict_to_pyg  # noqa: E402
from prism.models.composite_graph import build_composite_graph  # noqa: E402
from prism.eval import visualizer  # noqa: E402


def load_graph(path: str) -> dict:
    import json
    with open(path) as f:
        data = json.load(f)
    return data.get("graph", data)   # bare graph dict or eval JSON wrapping "graph"


def synthetic_injection_map(num_scene: int, c: int, mentions: int, span: int,
                            window: int = 12) -> dict:
    """Place `mentions` short token spans per scene node in a LOCAL window.

    Stands in for the M3 build_injection_map output when no tokenized prompt is
    available. Each label's mentions are clustered within a small `window` of
    cycle positions (not spread across the diameter), so the multi-mention (E2b)
    clique edges are short local arcs and cross-links are short spokes — keeping
    the rendered graph legible rather than a center-spanning hairball.
    """
    inj = {}
    base_step = max(1, c // max(1, num_scene))
    gap = max(span + 1, window // max(1, mentions))
    for j in range(num_scene):
        base = (j * base_step) % c
        spans = []
        for m in range(mentions):
            start = min(base + m * gap, c - span)
            spans.append((start, start + span))
        inj[j] = spans
    return inj


def main():
    parser = argparse.ArgumentParser(
        description="Render the composite graph (M4) from a scene-graph JSON.",
    )
    parser.add_argument("--graph", required=True, help="Path to a scene-graph JSON file.")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: outputs/visuals/aug_<graph_stem>).")
    parser.add_argument("--cycle", type=int, default=256,
                        help="Token cycle length c = |V_Tx| (default: 256).")
    parser.add_argument("--mentions", type=int, default=3,
                        help="Synthetic mentions per scene label (default: 3).")
    parser.add_argument("--span", type=int, default=2,
                        help="Token span length per mention (default: 2).")
    parser.add_argument("--window", type=int, default=48,
                        help="Cycle window the mentions of a label are spread over "
                             "(default: 48; larger => longer, more visible clique arcs).")
    parser.add_argument("--max-cycle", type=int, default=80,
                        help="Max token nodes drawn from the cycle (subsample, default: 80).")
    parser.add_argument("--open", action="store_true",
                        help="Open the HTML in the default browser after saving.")
    args = parser.parse_args()

    graph_dict = load_graph(args.graph)
    stem = Path(args.graph).stem
    out_dir = args.out or os.path.join("outputs", "visuals", f"aug_{stem}")

    scene = scene_graph_dict_to_pyg(graph_dict)
    n_scene = scene.num_nodes
    scene_ew = (scene.edge_weight if getattr(scene, "edge_weight", None) is not None
                else torch.ones(scene.edge_index.shape[1]))
    inj = synthetic_injection_map(n_scene, args.cycle, args.mentions, args.span, args.window)

    aug = build_composite_graph(args.cycle, scene.edge_index, scene_ew, n_scene, inj)
    print(f"Composite graph from {Path(args.graph).name}: "
          f"c={args.cycle} token + {n_scene} scene = {aug.num_nodes} nodes, "
          f"{aug.edge_index.shape[1]} directed edges; Fiedler={aug.fiedler():.4f}")

    os.makedirs(out_dir, exist_ok=True)
    src = Path(args.graph).name
    html = visualizer.render_composite_graph_html(
        aug, os.path.join(out_dir, "composite_graph.html"),
        max_cycle=args.max_cycle, source=src)
    png = visualizer.render_spectral_clustering(
        aug, os.path.join(out_dir, "composite_graph_spectral.png"),
        max_cycle=args.max_cycle, source=src)
    print(f"Saved → {html}\nSaved → {png}")

    if args.open:
        webbrowser.open(Path(html).resolve().as_uri())


if __name__ == "__main__":
    main()
