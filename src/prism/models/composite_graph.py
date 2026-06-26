"""Assemble the composite graph G and its graph shift operator S.

The composite graph glues a directed-cycle "sequence layer" (one node per token,
nodes 0..c-1) to the scene graph (nodes c..c+|V_Sc|-1) via cross-links:

  - sequence layer: directed cycle  i -> (i+1) mod c, weight cycle_weight;
  - scene layer:     edges from G_Sc with the scene affinity weight, directedness
                     preserved (the scene edge_index is used as given);
  - cross-links:     (a) every token of every mention of a label <=> that
                     label's scene node, and (b) all mention-tokens of a label
                     form a clique. Binary crosslink_weight, both directions.

R-PEARL consumes the directed (edge_index, edge_weight). The graph shift operator
S is the two-sided degree normalization Ŝ = D^{-1/2}(A+I)D^{-1/2} of the
*directed* adjacency: it scales but does not symmetrize (no reverse edges, no
``to_undirected``), preserving the directed circulant of the sequence cycle so
its spectrum stays complex. The symmetric Laplacian (``laplacian``/``fiedler``)
is a connectivity diagnostic (Fiedler, spectral clustering) only and must
never feed back as the operator. Everything stays sparse; the N×N matrix
(N ≈ 8200) is never densified.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch_geometric.utils import add_self_loops, coalesce


@dataclass
class CompositeGraph:
    """Assembled composite graph G plus its sparse GSO S.

    Attributes:
        edge_index: [2, E] directed edges of G (cycle is one-directional).
        edge_weight: [E] matching weights (cycle/affinity/crosslink).
        is_token: [N] bool mask, True on sequence-layer (token) nodes.
        num_token_nodes: c = |V_Tx|.
        num_scene_nodes: |V_Sc|.
        gso: sparse [N, N] two-sided degree-normalized adjacency of the DIRECTED
            graph (with self-loops); not symmetrized, so the directed circulant
            of the cycle is preserved.
    """

    edge_index: Tensor
    edge_weight: Tensor
    is_token: Tensor
    num_token_nodes: int
    num_scene_nodes: int
    gso: Tensor

    @property
    def num_nodes(self) -> int:
        return self.num_token_nodes + self.num_scene_nodes

    def laplacian(self):
        """Combinatorial Laplacian L = D - A of the symmetrized graph.

        Sparse, no self-loops; stays on the graph's own device.
        """
        N = self.num_nodes
        device = self.edge_index.device
        sym_index, sym_weight = _symmetrize(self.edge_index, self.edge_weight, N)
        deg = torch.zeros(N, device=device).index_add_(0, sym_index[0], sym_weight)
        diag = torch.arange(N, device=device).unsqueeze(0).expand(2, -1)
        lap_index = torch.cat([diag, sym_index], dim=1)
        lap_weight = torch.cat([deg, -sym_weight])
        return torch.sparse_coo_tensor(lap_index, lap_weight, (N, N)).coalesce()

    def fiedler(self) -> float:
        """Fiedler value λ₂ of the combinatorial Laplacian; > 0 iff G is connected.

        Uses ``torch.lobpcg`` on sparse L; falls back to scipy ``eigsh`` on failure.
        """
        L = self.laplacian()
        try:
            # Two smallest eigenvalues; λ₁≈0 (constant vector), λ₂ is the Fiedler.
            vals = torch.lobpcg(L.to_sparse_csr(), k=2, largest=False)[0]
            return float(vals.max())
        except Exception:
            from scipy.sparse import coo_matrix
            from scipy.sparse.linalg import eigsh

            Lc = L.coalesce().cpu()
            idx = Lc.indices().numpy()
            Ls = coo_matrix((Lc.values().numpy(), (idx[0], idx[1])),
                            shape=(self.num_nodes, self.num_nodes)).tocsr()
            return float(sorted(eigsh(Ls, k=2, which="SM", return_eigenvectors=False))[1])

    def scene_mass(self, k: int = 2) -> float:
        """Fraction of a cross-linked token's k-hop propagated mass on scene nodes.

        Drops unit mass on each cross-linked token, diffuses k steps through S, and
        averages the scene-node share. Collapses to ≈0 if cross-links are absent.
        """
        N = self.num_nodes
        device = self.gso.device
        dtype = self.gso.dtype
        src = self.edge_index[0]
        dst = self.edge_index[1]
        token_sources = src[self.is_token[src] & ~self.is_token[dst]].unique()
        if token_sources.numel() == 0:
            return 0.0

        cols = torch.arange(token_sources.numel(), device=device)
        V = torch.zeros(N, token_sources.numel(), device=device, dtype=dtype)
        V[token_sources, cols] = 1.0
        for _ in range(k):
            V = torch.sparse.mm(self.gso, V)

        scene_mask = (~self.is_token).to(device=device, dtype=dtype).unsqueeze(1)
        total = V.sum(0)
        scene = (V * scene_mask).sum(0)
        valid = total > 0
        return float((scene[valid] / total[valid]).mean()) if valid.any() else 0.0


def _symmetrize(edge_index: Tensor, edge_weight: Tensor, num_nodes: int):
    """Undirected view of a directed weighted edge set (dedup with max)."""
    sym_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    sym_weight = torch.cat([edge_weight, edge_weight])
    return coalesce(sym_index, sym_weight, num_nodes=num_nodes, reduce="max")


def build_composite_graph(
    num_token_nodes: int,
    scene_edge_index: Tensor,
    scene_edge_weight: Tensor,
    num_scene_nodes: int,
    injection_map: dict[int, list[tuple[int, int]]],
    cycle_weight: float = 1.0,
    cycle_directed: bool = True,
    crosslink_weight: float = 1.0,
    crosslink_mention_to_node: bool = True,
    crosslink_mention_clique: bool = True,
) -> CompositeGraph:
    """Assemble the composite graph G and its GSO S.

    Args:
        num_token_nodes: c, the sequence-layer length.
        scene_edge_index: [2, E_sc] scene edges in local 0..|V_Sc|-1 indexing.
        scene_edge_weight: [E_sc] scene affinity weights aligned to scene_edge_index.
        num_scene_nodes: |V_Sc|.
        injection_map: {scene_node_idx: [(start, end), ...]} token spans of each
            scene node's mentions, already scoped to the last graph.
        cycle_weight / cycle_directed / crosslink_*: see GREPConfig.

    Returns:
        CompositeGraph with directed (edge_index, edge_weight), is_token mask, and
        the sparse two-sided degree-normalized GSO of the directed graph.
    """
    c = num_token_nodes
    n_scene = num_scene_nodes
    N = c + n_scene
    device = scene_edge_index.device
    f32 = torch.float32

    rows: list[Tensor] = []
    cols: list[Tensor] = []
    vals: list[Tensor] = []

    def _add(src: Tensor, dst: Tensor, w: float):
        rows.append(src)
        cols.append(dst)
        vals.append(torch.full((src.numel(),), float(w), dtype=f32, device=device))

    # --- sequence layer: directed cycle i -> (i+1) mod c (wraparound included) ---
    i = torch.arange(c, device=device)
    nxt = (i + 1) % c
    _add(i, nxt, cycle_weight)
    if not cycle_directed:
        _add(nxt, i, cycle_weight)

    # --- scene layer: shift local indices by +c, keep affinity weights as given ---
    if scene_edge_index.numel() > 0:
        rows.append(scene_edge_index[0] + c)
        cols.append(scene_edge_index[1] + c)
        vals.append(scene_edge_weight.to(device=device, dtype=f32))

    # --- cross-links, restricted to last-graph mentions via injection_map ---
    for node_idx, spans in injection_map.items():
        scene_global = c + node_idx
        token_ids = sorted({t for start, end in spans for t in range(start, end) if t < c})
        if not token_ids:
            continue
        tok = torch.tensor(token_ids, device=device, dtype=torch.long)
        node = torch.full_like(tok, scene_global)
        if crosslink_mention_to_node:
            _add(tok, node, crosslink_weight)   # token -> scene node
            _add(node, tok, crosslink_weight)   # scene node -> token
        if crosslink_mention_clique and tok.numel() > 1:
            a, b = torch.meshgrid(tok, tok, indexing="ij")
            off_diag = a != b
            _add(a[off_diag], b[off_diag], crosslink_weight)

    edge_index = torch.stack([torch.cat(rows), torch.cat(cols)])
    edge_weight = torch.cat(vals)

    is_token = torch.zeros(N, dtype=torch.bool, device=device)
    is_token[:c] = True

    gso = _build_gso(edge_index, edge_weight, N)

    return CompositeGraph(
        edge_index=edge_index,
        edge_weight=edge_weight,
        is_token=is_token,
        num_token_nodes=c,
        num_scene_nodes=n_scene,
        gso=gso,
    )


def _build_gso(edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
    """Two-sided degree-normalized GSO Ŝ = D^{-1/2}(A+I)D^{-1/2} on the DIRECTED A.

    Does NOT symmetrize (no reverse edges, no ``to_undirected``), so a forward-only
    cycle edge i→(i+1) leaves S[i+1, i] == 0. D = row-degree of (A+I).
    """
    sl_index, sl_weight = add_self_loops(
        edge_index, edge_weight, fill_value=1.0, num_nodes=num_nodes
    )
    deg = torch.zeros(num_nodes, device=edge_index.device).index_add_(0, sl_index[0], sl_weight)
    dinv = deg.pow(-0.5)
    dinv[torch.isinf(dinv)] = 0.0
    norm = dinv[sl_index[0]] * sl_weight * dinv[sl_index[1]]
    return torch.sparse_coo_tensor(sl_index, norm, (num_nodes, num_nodes)).coalesce()


def composite_graph_gnn_rebuild_params(config) -> dict:
    """GNN checkpoint rebuild params for ``composite_graph_gt`` (read back by loaders for eval)."""
    if config.gnn.arch != "composite_graph_gt":
        return {}
    return {
        "k_gt": config.gnn.k_gt,
        "gt_num_layers": config.gnn.gt_num_layers,
        "gt_heads": config.gnn.gt_heads,
        "probe_distribution": config.gnn.probe_distribution,
        "max_gather_rows": config.gnn.max_gather_rows,
        "fixed_seed_mode": config.gnn.fixed_seed_mode,
        "fixed_seed_value": config.gnn.fixed_seed_value,
        "injection_mode": config.gnn.injection_mode,
        "gate_init": config.gnn.gate_init,
        "gate_per_dim": config.gnn.gate_per_dim,
        "disable_rope": config.model.disable_rope,
        "structural_lr_mult": config.gnn.structural_lr_mult,
        "pe_readout": config.gnn.pe_readout,
        "pe_center_moment": config.gnn.pe_center_moment,
        "cycle_weight": config.gnn.cycle_weight,
        "cycle_directed": config.gnn.cycle_directed,
        "crosslink_weight": config.gnn.crosslink_weight,
        "crosslink_mention_to_node": config.gnn.crosslink_mention_to_node,
        "crosslink_mention_clique": config.gnn.crosslink_mention_clique,
        "pe_qk_injection": config.gnn.pe_qk_injection,
        "pe_inject_v": config.gnn.pe_inject_v,
        "c_per_layer": config.gnn.c_per_layer,
        "c_bias": config.gnn.c_bias,
        "use_scene_bias": config.gnn.use_scene_bias,
        "c_kernel": config.gnn.c_kernel,
    }
