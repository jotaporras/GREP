"""M12 — visualize the augmented graph and its spectral clustering (R10).

Two artifacts for a single augmented graph (from M4):
  1. an interactive vis.js HTML laying out the two layers — token (directed
     cycle) nodes vs. scene nodes — with cross-links highlighted;
  2. a spectral-clustering PNG coloring nodes by the Fiedler vector of the
     (diagnostic, symmetric) augmented Laplacian.

Both **reuse the existing scripts** rather than reimplementing: the vis.js
HTML shell from ``scripts/render_scene_graph.py`` and the GSO eigendecomposition
+ panel drawing from ``scripts/spectral_clustering.py``. The c=8192 cycle is
subsampled for legibility (all scene nodes + all cross-linked token nodes + a
strided sample of the rest), as the spec recommends.

Gated behind ``enable_visualizer``; see ``visualize`` (the entry point).
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict


def _import_scripts():
    """Import the reusable rendering + spectral helpers from ``scripts/`` (lazy)."""
    scripts_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "scripts")
    )
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import render_scene_graph
    import spectral_clustering
    return render_scene_graph, spectral_clustering


def _subsample(aug, max_cycle: int = 80):
    """Reduce the augmented graph for legible rendering.

    Keeps every scene node and every cross-linked token node, plus a strided
    sample of the remaining cycle for context. Returns the kept global ids, a
    {global→local} remap, the per-node is_token flags, edges classified into
    cycle/scene/crosslink (local ids), positions (token ring, scene core), and c.
    """
    c = aug.num_token_nodes
    N = aug.num_nodes
    ei = aug.edge_index.cpu()
    is_tok = aug.is_token.cpu()
    src, dst = ei[0], ei[1]

    crosslinked_tokens = src[is_tok[src] & ~is_tok[dst]].unique().tolist()
    keep_tokens = set(int(t) for t in crosslinked_tokens)
    stride = max(1, c // max_cycle)
    keep_tokens.update(range(0, c, stride))

    keep = sorted(keep_tokens) + list(range(c, N))   # tokens then scene
    remap = {g: i for i, g in enumerate(keep)}
    keepset = set(keep)
    node_is_token = [g < c for g in keep]

    # clique: token↔token that is NOT a cycle step (same-label multi-mention E2b);
    # scene: scene↔scene; crosslink: token↔scene. Real i→(i+1) cycle steps are
    # dropped here and replaced by a coarse ring below (subsampling breaks most
    # consecutive adjacencies, so the real cycle edges would be invisible).
    edges = {"cycle": [], "clique": [], "scene": [], "crosslink": []}
    for s, d in ei.t().tolist():
        if s in keepset and d in keepset:
            st, dt = s < c, d < c
            if st and dt:
                if d == (s + 1) % c:
                    continue
                edges["clique"].append((remap[s], remap[d]))
            elif not st and not dt:
                edges["scene"].append((remap[s], remap[d]))
            else:
                edges["crosslink"].append((remap[s], remap[d]))

    # Coarse cycle ring: connect consecutive kept token nodes (in global cycle
    # order) so the subsampled sequence layer renders as a visible ring.
    kept_tokens = [remap[g] for g in keep if g < c]
    for i in range(len(kept_tokens)):
        a, b = kept_tokens[i], kept_tokens[(i + 1) % len(kept_tokens)]
        if a != b:
            edges["cycle"].append((a, b))

    pos = {}
    # Token nodes on the outer ring (angle = cycle index).
    for g, i in remap.items():
        if g < c:
            ang = 2 * math.pi * g / max(1, c)
            pos[i] = (math.cos(ang), math.sin(ang))

    # Place each scene node radially inward from the *mean angle of its own
    # cross-linked tokens*, so cross-links are short local spokes instead of all
    # converging on one central point (the cause of the hairball).
    scene_tokens: dict = {}
    for a, b in edges["crosslink"]:
        tok, sc = (a, b) if node_is_token[a] else (b, a)
        scene_tokens.setdefault(sc, []).append(tok)
    scene_locals = [i for i, is_t in enumerate(node_is_token) if not is_t]
    for rank, i in enumerate(scene_locals):
        toks = scene_tokens.get(i)
        if toks:
            mx = sum(pos[t][0] for t in toks) / len(toks)
            my = sum(pos[t][1] for t in toks) / len(toks)
            r = math.hypot(mx, my) or 1.0
            pos[i] = (0.6 * mx / r, 0.6 * my / r)
        else:
            ang = 2 * math.pi * rank / max(1, len(scene_locals))
            pos[i] = (0.45 * math.cos(ang), 0.45 * math.sin(ang))
    return keep, remap, node_is_token, edges, pos, c


def render_augmented_graph_html(aug, out_path: str, max_cycle: int = 80, source: str = None) -> str:
    """Write the interactive augmented-graph HTML (reuses render_scene_graph's vis.js shell)."""
    import json
    rsg, _ = _import_scripts()
    keep, remap, node_is_token, edges, pos, c = _subsample(aug, max_cycle)

    crosslinked = {i for e in edges["crosslink"] for i in e if node_is_token[i]}
    SCALE = 600
    vis_nodes = []
    for i, is_tok in enumerate(node_is_token):
        x, y = pos[i][0] * SCALE, -pos[i][1] * SCALE
        if is_tok:   # token (cycle) node — blue dot, gold border if cross-linked
            vis_nodes.append({
                "id": i, "label": "", "x": round(x, 1), "y": round(y, 1),
                "shape": "dot", "size": 10 if i in crosslinked else 5,
                "color": {"background": "#4a90d9",
                          "border": "#b8860b" if i in crosslinked else "#2c6fad"},
            })
        else:        # scene node — red diamond
            vis_nodes.append({
                "id": i, "label": f"s{keep[i] - c}", "x": round(x, 1), "y": round(y, 1),
                "shape": "diamond", "size": 14,
                "color": {"background": "#e05c5c", "border": "#a32929"},
                "font": {"size": 11, "color": "#1a1a2e"},
            })
    vis_edges = []
    for a, b in edges["cycle"]:
        vis_edges.append({"from": a, "to": b, "color": {"color": "#006400"}, "width": 2.5, "arrows": "to"})
    for a, b in edges["scene"]:
        vis_edges.append({"from": a, "to": b, "color": {"color": "#e05c5c"}, "width": 1.0})
    for a, b in edges["crosslink"]:   # mention↔scene-node — gold
        vis_edges.append({"from": a, "to": b, "color": {"color": "#FFD700", "highlight": "#b8860b"}, "width": 0.8})
    for a, b in edges["clique"]:      # same-label multi-mention clique (E2b) — purple
        vis_edges.append({"from": a, "to": b, "color": {"color": "#9b59b6", "highlight": "#6c3483"}, "width": 0.8})

    source_label = f"augmented graph (M4) — source: {source}" if source else "augmented graph (M4)"
    html = rsg.HTML_TEMPLATE.format(
        title="Augmented Graph", source_file=source_label,
        n_nodes=len(keep), n_edges=len(vis_edges), robot_location="—",
        vis_nodes=json.dumps(vis_nodes), vis_edges=json.dumps(vis_edges),
    )
    # Relabel the reused legend to the augmented-graph layers + edge classes.
    html = (html.replace("</div> Region", "</div> Token node (cycle)")
                .replace("</div> Object", "</div> Scene node")
                .replace("Robot start (",
                         "Gold=mention↔node, Purple=same-label clique, Red=scene ("))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# Per-class edge styling (color, width) shared by both artifacts' intent.
# (color, width, alpha). The directed cycle is the structural backbone, so it
# stays thick/opaque; the dense cross-link/clique edges are kept thin and faint
# so the figure reads instead of blooming into a hairball.
_EDGE_STYLE = {
    "cycle": ("#006400", 2.5, 0.95),   # directed sequence cycle (dark green)
    "scene": ("#e05c5c", 1.0, 0.7),    # scene affinity edges
    "crosslink": ("#FFD700", 0.6, 0.4),  # mention ↔ scene node (E2a)
    "clique": ("#9b59b6", 1.1, 0.6),   # same-label multi-mention clique (E2b)
}


def render_spectral_clustering(aug, out_path: str, max_cycle: int = 80, source: str = None) -> str:
    """Write the spectral-clustering PNG: nodes colored by the Fiedler vector
    (reusing spectral_clustering's eigendecomposition), edges colored by class.
    """
    import torch
    import networkx as nx
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from torch_geometric.data import Data
    _, sc = _import_scripts()

    keep, remap, node_is_token, edges, pos, c = _subsample(aug, max_cycle)
    K = len(keep)
    # Symmetric (undirected) view for the diagnostic spectral clustering.
    undirected = [(a, b) for cls in edges for (a, b) in edges[cls]]
    undirected += [(b, a) for (a, b) in undirected]
    ei = (torch.tensor(undirected, dtype=torch.long).t()
          if undirected else torch.zeros(2, 0, dtype=torch.long))
    evals, evecs = sc.compute_gso_eigen(Data(edge_index=ei, num_nodes=K))  # reused
    fiedler = evecs[:, 1] if evecs.shape[1] > 1 else evecs[:, 0]

    G = nx.Graph()
    G.add_nodes_from(range(K))
    fig, ax = plt.subplots(figsize=(11, 10))
    # colored edges per class: faint cross-links first, then cliques on top of
    # them, scene edges, and the cycle ring last.
    for cls in ("crosslink", "clique", "scene", "cycle"):
        color, width, alpha = _EDGE_STYLE[cls]
        if edges[cls]:
            nx.draw_networkx_edges(G, pos=pos, ax=ax, edgelist=edges[cls],
                                   edge_color=color, width=width, alpha=alpha)
    # nodes colored by the Fiedler vector (same cmap/normalization as _draw_panel)
    vals = fiedler.numpy()
    norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())
    cmap = plt.cm.RdYlBu
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=[cmap(norm(v)) for v in vals],
                           node_size=160, edgecolors="k", linewidths=0.4)
    nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=5, font_color="black")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75, pad=0.02,
                 label="Fiedler value")
    ax.legend(handles=[Line2D([0], [0], color=col, lw=2, label=name)
                       for name, (col, _w, _a) in _EDGE_STYLE.items()],
              loc="upper left", fontsize=8, title="edge type")
    ax.set_title("Augmented graph — spectral clustering (Fiedler) + colored edges", fontsize=11)
    ax.axis("off")
    if source:
        fig.text(0.5, 0.01, f"source: {source}", ha="center", fontsize=8, color="#555")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize(aug, out_dir: str, config=None, source: str = None) -> Dict[str, str]:
    """M12 entry point: write both artifacts for one augmented graph.

    ``source`` is an optional provenance label (e.g. the scene-graph file or
    checkpoint) stamped onto both artifacts. Returns paths {"html",
    "spectral_png"}. Intended to be gated by the caller on ``enable_visualizer``
    (see callbacks.AugGraphDebugCallback).
    """
    os.makedirs(out_dir, exist_ok=True)
    html = render_augmented_graph_html(aug, os.path.join(out_dir, "augmented_graph.html"), source=source)
    png = render_spectral_clustering(aug, os.path.join(out_dir, "augmented_graph_spectral.png"), source=source)
    print(f"[M12] visualizer wrote: {html} , {png}")
    return {"html": html, "spectral_png": png}
