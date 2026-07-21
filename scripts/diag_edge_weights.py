"""Premise diagnostic: what does the Gaussian affinity do to SBM+geometry scene graphs?

For each eval graph: per-edge-class (object_connections vs region_connections)
distance and Gaussian-weight stats, share of total weight vs share of edge count,
and the Fiedler value (algebraic connectivity, normalized Laplacian) of the
weighted vs unweighted graph. Uses the SAME graph parse as training
(prism.data.utils.safe_parse_graph) and the same sigma = median edge length.

Usage:
    PYTHONPATH=src python scripts/diag_edge_weights.py \
        'data/revised/gen/nav100_n30_gemma_data/split/test_graphs/*.json'
(quote the glob — the script expands argv[1] itself).
"""
import glob
import json
import math
import sys

import networkx as nx
import numpy as np

from prism.data import utils


def fiedler(G, weight):
    """Second-smallest eigenvalue of the normalized Laplacian on the largest CC."""
    H = G.subgraph(max(nx.connected_components(G), key=len))
    L = nx.normalized_laplacian_matrix(H, weight=weight).toarray()
    return float(np.sort(np.linalg.eigvalsh(L))[1])


rows = []
for path in sorted(glob.glob(sys.argv[1])):
    payload = json.load(open(path))
    g_dict = payload["graph"] if "graph" in payload else payload
    G, _ = utils.safe_parse_graph(g_dict)

    dists = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    sigma = np.median(dists)
    for u, v, d in G.edges(data=True):
        d["gauss"] = math.exp(-d["weight"] ** 2 / (2 * sigma**2)) if sigma > 0 else 1.0

    by_type = {}
    for u, v, d in G.edges(data=True):
        by_type.setdefault(d["type"], []).append((d["weight"], d["gauss"]))

    total_w = sum(d["gauss"] for _, _, d in G.edges(data=True))
    n_edges = G.number_of_edges()
    row = {"graph": path.split("/")[-1], "n_nodes": G.number_of_nodes(),
           "n_edges": n_edges, "sigma": sigma}
    for t, pairs in sorted(by_type.items()):
        dd = np.array([p[0] for p in pairs])
        ww = np.array([p[1] for p in pairs])
        row[f"{t}: n"] = len(pairs)
        row[f"{t}: dist med"] = float(np.median(dd))
        row[f"{t}: w med"] = float(np.median(ww))
        row[f"{t}: w max"] = float(ww.max())
        row[f"{t}: share of count"] = len(pairs) / n_edges
        row[f"{t}: share of weight"] = float(ww.sum()) / total_w
    row["fiedler unweighted"] = fiedler(G, weight=None)
    row["fiedler gaussian"] = fiedler(G, weight="gauss")
    rows.append(row)

keys = ["n_nodes", "n_edges", "sigma",
        "object: n", "object: dist med", "object: w med",
        "region: n", "region: dist med", "region: w med", "region: w max",
        "region: share of count", "region: share of weight",
        "fiedler unweighted", "fiedler gaussian"]
print(f"{'graph':28s}" + "".join(f"{k:>18s}" for k in keys))
for r in rows:
    print(f"{r['graph']:28s}" + "".join(
        f"{r.get(k, float('nan')):18.4g}" if isinstance(r.get(k), (int, float))
        else f"{'—':>18s}" for k in keys))

agg = {k: np.mean([r[k] for r in rows if k in r]) for k in keys if any(k in r for r in rows)}
print("\nMEANS over graphs:")
for k in keys:
    if k in agg:
        print(f"  {k:24s} {agg[k]:.6g}")
