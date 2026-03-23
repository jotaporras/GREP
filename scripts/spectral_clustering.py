"""Spectral clustering analysis of R-PEARL / Graph Transformer filter banks.

Computes the MIMO frequency response and spectral clustering of the TAG
convolutional filter banks inside a trained GNN checkpoint and contrasts
them against a randomly-initialized model of the same architecture.

Outputs a 2x2 figure:
    ┌──────────────────┬──────────────────┐
    │  Trained cluster │ Untrained cluster│
    ├──────────────────┼──────────────────┤
    │  Trained S(λ)    │ Untrained S(λ)   │
    └──────────────────┴──────────────────┘

Usage:
    python scripts/spectral_clustering.py <checkpoint_path>
    python scripts/spectral_clustering.py <checkpoint_path> --data data/eval/gpt_gen_formatted.json
    python scripts/spectral_clustering.py <checkpoint_path> --layer 2 --output-feature 3
    python scripts/spectral_clustering.py <checkpoint_path> --data data/eval/gpt_gen_formatted.json --graph-index 42
"""
import argparse
import json
import os
import random
import re
import sys
from ast import literal_eval

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.utils import to_dense_adj, to_networkx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from prism.data.utils import scene_graph_dict_to_pyg
from prism.models.gt import GraphTransformer
from prism.models.r_pearl import RandomGNNPositionalEncodings


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spectral clustering analysis of R-PEARL / Graph Transformer filter banks.",
    )
    parser.add_argument(
        "checkpoint",
        help="Checkpoint directory containing gnn_config.json and gnn_weights.pt.",
    )
    parser.add_argument(
        "--data",
        default="data/eval/eval_1_multi_step.json",
        help=(
            "Path to a data JSON. Accepts two formats: "
            "(1) eval JSON with a top-level 'graph' key, or "
            "(2) training JSON (list of conversation dicts) — a graph will be "
            "randomly selected unless --graph-index is specified."
        ),
    )
    parser.add_argument(
        "--graph-index",
        type=int,
        default=None,
        help="For training data: index of the conversation to use. Random if omitted.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="GCN conv layer index (0-based). Negative indices count from the end. Default: -1 (last layer).",
    )
    parser.add_argument(
        "--output-feature",
        type=int,
        default=0,
        help="Output feature index f for spectral clustering (default: 0).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Default: outputs/visuals/spectral_<checkpoint_name>.png.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def _build_pe_model(gnn_cfg: dict) -> torch.nn.Module:
    """Build a PE model from gnn_config (random weights, no checkpoint loaded)."""
    architecture = gnn_cfg.get("architecture", "rpearl_llm")
    if architecture == "rpearl_gt_llm":
        return GraphTransformer(
            num_layers=gnn_cfg["gt_num_layers"],
            pe_hidden_channels=gnn_cfg["pe_hidden_channels"],
            pe_num_layers=gnn_cfg["pe_num_layers"],
            d_model=gnn_cfg["d_model"],
            heads=gnn_cfg["gt_heads"],
            num_samples=gnn_cfg["num_samples"],
            dropout=gnn_cfg["dropout"],
            k_pe=gnn_cfg["k_pe"],
            k_gt=gnn_cfg["k_gt"],
            eps=gnn_cfg["gt_eps"],
            use_layer_norm=gnn_cfg["use_layer_norm"],
        )
    return RandomGNNPositionalEncodings(
        pe_hidden_channels=gnn_cfg["pe_hidden_channels"],
        pe_num_layers=gnn_cfg["pe_num_layers"],
        d_model=gnn_cfg["d_model"],
        num_samples=gnn_cfg["num_samples"],
        dropout=gnn_cfg["dropout"],
        k=gnn_cfg["k_pe"],
        use_layer_norm=gnn_cfg["use_layer_norm"],
    )


def _get_gcn(pe_model):
    """Return the inner GCN regardless of whether pe_model is R-PEARL or GraphTransformer."""
    if isinstance(pe_model, GraphTransformer):
        return pe_model.pe_model.pe_gcn
    return pe_model.pe_gcn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_SG_PATTERN = re.compile(r"[Ss]cene[_ ]graph:\s*(\{.*\})", re.DOTALL)


def _extract_scene_graph_from_conversation(conversation: list[dict]) -> dict:
    """Pull the scene graph dict from the user turn of a training conversation."""
    for turn in conversation:
        role = turn.get("from") or turn.get("role", "")
        text = turn.get("value") or turn.get("content", "")
        if role not in ("human", "user"):
            continue
        m = _SG_PATTERN.search(text)
        if m:
            return literal_eval(m.group(1))
    raise ValueError("No scene graph found in conversation.")


def load_graph(data_path: str, graph_index: int | None):
    """Load a PyG graph from either an eval JSON or a training JSON.

    Returns (pyg_graph, description_str).
    """
    with open(data_path) as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "graph" in raw:
        graph = scene_graph_dict_to_pyg(raw["graph"])
        return graph, "eval graph"

    # Training format: list of conversation dicts
    if graph_index is None:
        graph_index = random.randint(0, len(raw) - 1)
    print(f"Selected training example {graph_index}/{len(raw) - 1}")
    sg_dict = _extract_scene_graph_from_conversation(raw[graph_index]["conversations"])
    graph = scene_graph_dict_to_pyg(sg_dict)
    return graph, f"training example {graph_index}"


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def compute_gso_eigen(graph):
    """Eigendecompose the normalized adjacency S = D^{-1/2} A D^{-1/2}."""
    adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes)[0]
    deg = adj.sum(dim=1)
    deg_inv_sqrt = torch.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = 1.0 / torch.sqrt(deg[mask])
    S = torch.diag(deg_inv_sqrt) @ adj @ torch.diag(deg_inv_sqrt)
    eigenvalues, eigenvectors = torch.linalg.eigh(S)
    return eigenvalues, eigenvectors


def compute_freq_response(gcn, layer_idx, eigenvalues, device):
    r"""Compute the MIMO frequency response H̃^{(l)}(λ_i) for each eigenvalue.

    Returns a tensor of shape ``[n_eigenvalues, F_in, F_out]`` on *device*.
    """
    lins = gcn.convs[layer_idx].lins
    F_out, F_in = lins[0].weight.shape
    n = eigenvalues.shape[0]
    result = torch.zeros(n, F_in, F_out, device=device)
    for k, lin in enumerate(lins):
        H_k = lin.weight.detach().T                # [F_in, F_out], already on device
        for i in range(n):
            result[i] += (eigenvalues[i].item() ** k) * H_k
    return result


def compute_clustering(freq_response, eigenvectors, output_feature):
    r"""Compute spectral clustering color vector c_f from the frequency response.

    .. math::
        Q_f = \operatorname{diag}(\|H̃(λ_i) e_f\|^2)
        c_f = (V \odot V^*) Q_f \mathbf{1}

    Returns CPU tensor for plotting.
    """
    sensitivity = torch.linalg.norm(freq_response, dim=1) ** 2   # [n, F_out]
    clusters = (eigenvectors * eigenvectors.conj()).real @ sensitivity  # [n, F_out]
    return clusters[:, output_feature].cpu()


def compute_selectivity_curve(gcn, layer_idx, eigenvalues, output_feature, n_points=200):
    r"""Evaluate S(λ) = \|H̃(λ) e_f\|^2 on a dense grid for plotting.

    Returns CPU tensors for plotting.
    """
    lins = gcn.convs[layer_idx].lins
    lambdas = torch.linspace(eigenvalues.min().item(), eigenvalues.max().item(), n_points)
    selectivity = []
    for lam in lambdas:
        H = sum(lin.weight.detach().T * (lam.item() ** k) for k, lin in enumerate(lins))
        selectivity.append((H[:, output_feature] ** 2).sum().cpu().item())
    return lambdas, selectivity


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_panel(fig, ax, nx_graph, pos, node_values, title):
    """Draw a spectral-clustering-colored graph into a matplotlib axis."""
    vals = node_values.numpy()
    norm = mcolors.Normalize(vmin=vals.min(), vmax=vals.max())
    cmap = plt.cm.RdYlBu
    node_colors = [cmap(norm(v)) for v in vals]
    nx.draw_networkx_edges(nx_graph, pos=pos, ax=ax, edge_color="#888888", width=0.6, alpha=0.7)
    nx.draw_networkx_nodes(nx_graph, pos=pos, ax=ax, node_color=node_colors, node_size=260, edgecolors="k", linewidths=0.4)
    nx.draw_networkx_labels(nx_graph, pos=pos, ax=ax, font_size=5, font_color="black")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75, pad=0.02)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _draw_freq(ax, lambdas, selectivity, eigenvalues, title):
    """Draw spectral selectivity S(λ) with eigenvalue markers."""
    ax.plot(lambdas.numpy(), selectivity, linewidth=1.2)
    for ev in eigenvalues.cpu():
        ax.axvline(x=ev.item(), color="red", alpha=0.15, linewidth=1)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("λ (graph frequency)")
    ax.set_ylabel("S(λ)")
    ax.set_title(title, fontsize=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load config and build models
    # ------------------------------------------------------------------
    gnn_config_path = os.path.join(args.checkpoint, "gnn_config.json")
    with open(gnn_config_path) as f:
        gnn_cfg = json.load(f)

    architecture = gnn_cfg.get("architecture", "rpearl_llm")
    print(f"Architecture: {architecture}")

    trained_pe = _build_pe_model(gnn_cfg).to(device)
    untrained_pe = _build_pe_model(gnn_cfg).to(device)

    weights = torch.load(
        os.path.join(args.checkpoint, "gnn_weights.pt"), map_location=device,
    )
    key = "gt_model" if architecture == "rpearl_gt_llm" else "pe_model"
    trained_pe.load_state_dict(weights[key])

    trained_pe.eval()
    untrained_pe.eval()

    trained_gcn = _get_gcn(trained_pe)
    untrained_gcn = _get_gcn(untrained_pe)

    # ------------------------------------------------------------------
    # 2. Resolve layer index
    # ------------------------------------------------------------------
    n_convs = len(trained_gcn.convs)
    layer = args.layer if args.layer >= 0 else n_convs + args.layer
    print(f"GCN layer {layer}/{n_convs - 1}, output feature {args.output_feature}")

    # ------------------------------------------------------------------
    # 3. Load scene graph and eigendecompose the GSO
    # ------------------------------------------------------------------
    print(f"Loading graph from {args.data}")
    graph, desc = load_graph(args.data, args.graph_index)
    print(f"Graph ({desc}): {graph.num_nodes} nodes, {graph.edge_index.shape[1]} directed edges")

    graph = graph.to(device)
    eigenvalues, eigenvectors = compute_gso_eigen(graph)

    # ------------------------------------------------------------------
    # 4. Spectral analysis (trained vs untrained)
    # ------------------------------------------------------------------
    results = {}
    with torch.no_grad():
        for label, gcn in [("Trained", trained_gcn), ("Untrained", untrained_gcn)]:
            freq = compute_freq_response(gcn, layer, eigenvalues, device)
            clusters = compute_clustering(freq, eigenvectors, args.output_feature)
            lambdas, selectivity = compute_selectivity_curve(
                gcn, layer, eigenvalues, args.output_feature
            )
            results[label] = (clusters, lambdas, selectivity)
            print(f"  {label:>9s}  cluster range: [{clusters.min():.4f}, {clusters.max():.4f}]")

    # Move eigenvalues to CPU for plotting
    eigenvalues_cpu = eigenvalues.cpu()

    # ------------------------------------------------------------------
    # 5. Build networkx graph for drawing
    # ------------------------------------------------------------------
    graph_cpu = graph.cpu()
    nx_graph = to_networkx(graph_cpu, to_undirected=True)
    mapping = {i: name for i, name in enumerate(graph_cpu.node_names)}
    nx_graph = nx.relabel_nodes(nx_graph, mapping)

    if hasattr(graph_cpu, "coords") and graph_cpu.coords is not None:
        pos = {name: graph_cpu.coords[i].tolist() for i, name in enumerate(graph_cpu.node_names)}
    else:
        pos = nx.spring_layout(nx_graph, seed=42)

    # ------------------------------------------------------------------
    # 6. Plot 2×2 figure
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for col, label in enumerate(["Trained", "Untrained"]):
        clusters, lambdas, selectivity = results[label]
        _draw_panel(
            fig, axes[0, col], nx_graph, pos, clusters,
            f"{label} — Spectral Clustering (layer {layer}, f={args.output_feature})",
        )
        _draw_freq(
            axes[1, col], lambdas, selectivity, eigenvalues_cpu,
            f"{label} — Spectral Selectivity S(λ) (layer {layer}, f={args.output_feature})",
        )

    ckpt_name = os.path.basename(args.checkpoint.rstrip("/"))
    fig.suptitle(f"Spectral Analysis — {ckpt_name}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # ------------------------------------------------------------------
    # 7. Save
    # ------------------------------------------------------------------
    if args.output is not None:
        output_path = args.output
    else:
        output_path = f"outputs/visuals/spectral_{ckpt_name}.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
