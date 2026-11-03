import argparse
import json

import networkx as nx
import numpy as np
from scipy import spatial

# e22 spatial layouts (docs/2026-11-02 e22_geometric_scale_design.md §2). The
# SBM gives dense, clustered graphs whose diameter grows with log N; the robot's
# hospital graph is a near-tree of rooms (mean degree ~2.2, diameter 20 at 37
# regions). `floorplan` is a BSP floor plan (rooms = recursive rectangle
# splits, doors = spanning tree of shared walls + a few extra doors);
# `emst` is the fallback (Delaunay EMST + extra edges, no zones).
LAYOUTS = ("sbm", "floorplan", "emst")
PLAN_ASPECT = 1.4  # width / height of the BSP rectangle (notebook §8b: 1.0 x 0.7)
MIN_WALL = 0.02    # shared wall (in plan-width units) that can hold a door
# §2 acceptance band: mean over >= 20 seeds at 100 regions, keys of
# layout_statistics. Read by tests/test_floorplan_generator.py and
# scripts/e22_corpus_audit.py.
TARGET_BAND_100 = {
    "mean_degree": (2.1, 2.6),
    "dead_end_frac": (0.15, 0.30),
    "bridge_frac": (0.3, 0.7),
    "cycles_per_region": (0.05, 0.25),
    "diameter": (18, 28),
    "clustering": (0.0, 0.15),
}


def build_probability_matrix(n_communities, intra_prob, inter_prob) -> list[list[float]]:
    """Diagonal = intra_prob, off-diagonal = inter_prob."""
    prob = np.full((n_communities, n_communities), inter_prob)
    np.fill_diagonal(prob, intra_prob)
    return prob.tolist()


def generate_region_graph(n_communities, nodes_per_community, intra_prob, inter_prob, rng, max_retries=100) -> tuple[nx.Graph, list[int]]:
    """Build a connected SBM region graph, retrying with fresh seeds until connected."""
    prob = build_probability_matrix(n_communities, intra_prob, inter_prob)
    sizes = [nodes_per_community] * n_communities

    for _ in range(max_retries):
        seed = int(rng.integers(0, 2**31))
        region_graph = nx.generators.community.stochastic_block_model(
            sizes, prob, seed=seed
        )
        if nx.is_connected(region_graph):
            community_assignments = [region_graph.nodes[i]["block"] for i in range(len(region_graph))]
            return region_graph, community_assignments

    raise RuntimeError(
        f"Could not generate a connected region graph after {max_retries} retries. "
        f"Try increasing intra_prob or inter_prob."
    )


def assign_coordinates(G, community_assignments, n_communities, rng) -> dict[int, list[float]]:
    """Place community centroids on a circle with Gaussian jitter per node."""
    scale = 5.0 * len(G.nodes)
    radius, jitter_std = scale / 2, scale / (4 * n_communities)
    angles = np.linspace(0, 2 * np.pi, n_communities, endpoint=False)
    centroids = {c: (radius * np.cos(a), radius * np.sin(a)) for c, a in enumerate(angles)}

    coords = {}
    for node in G.nodes:
        cx, cy = centroids[community_assignments[node]]
        x = round(float(cx + rng.normal(0, jitter_std)), 1)
        y = round(float(cy + rng.normal(0, jitter_std)), 1)
        coords[node] = [x, y]

    return coords


def bsp_rooms(n_regions, corridor_bias, zone_depth, rng) -> list[dict]:
    """Recursively split a PLAN_ASPECT rectangle into ``n_regions`` rooms.

    Port of ``bsp_floorplan`` (notebook 2026-11-01 §8b): split the largest
    room, split position U[0.35, 0.65], long axis preferred (forced once the
    room is more than 1/0.35 wide). Each room carries its split-tree ``path``
    (one entry per ancestor split) so zones can be read off at ``zone_depth``.

    ``corridor_bias``: a split above ``zone_depth`` (building-scale) carves,
    with this probability, a thin corridor strip (aspect >= 4) between its two
    halves; every room later cut from either half can open onto it. Corridors
    are never split again.
    """
    rooms = [{"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0 / PLAN_ASPECT,
              "path": (), "corridor": False}]
    while len(rooms) < n_regions:
        splittable = [j for j, r in enumerate(rooms) if not r["corridor"]]
        i = max(splittable, key=lambda j: rooms[j]["w"] * rooms[j]["h"])
        r = rooms.pop(i)
        x, y, w, h, path = r["x"], r["y"], r["w"], r["h"], r["path"]
        f = float(rng.uniform(0.35, 0.65))
        split_x = w / h > 1 / 0.35 or (w >= h and rng.random() < 0.8)
        carve = (
            len(path) < zone_depth
            and len(rooms) + 3 <= n_regions
            and rng.random() < corridor_bias
        )
        # Corridor strip width: thin against the room (10% of the cut side)
        # and at most a quarter of its length, so its aspect is >= 4.
        c = (min(0.1 * w, h / 4) if split_x else min(0.1 * h, w / 4)) if carve else 0.0
        if split_x:
            a = {"x": x, "y": y, "w": w * f - c / 2, "h": h}
            b = {"x": x + w * f + c / 2, "y": y, "w": w * (1 - f) - c / 2, "h": h}
            mid = {"x": x + w * f - c / 2, "y": y, "w": c, "h": h}
        else:
            a = {"x": x, "y": y, "w": w, "h": h * f - c / 2}
            b = {"x": x, "y": y + h * f + c / 2, "w": w, "h": h * (1 - f) - c / 2}
            mid = {"x": x, "y": y + h * f - c / 2, "w": w, "h": c}
        rooms += [{**a, "path": path + (0,), "corridor": False},
                  {**b, "path": path + (1,), "corridor": False}]
        if carve:
            rooms.append({**mid, "path": path + (2,), "corridor": True})
    return rooms


def _shared_wall(a, b) -> float:
    """Length of the wall segment rooms ``a`` and ``b`` share (0 if none)."""
    eps = 1e-9
    if abs(a["x"] + a["w"] - b["x"]) < eps or abs(b["x"] + b["w"] - a["x"]) < eps:
        return min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
    if abs(a["y"] + a["h"] - b["y"]) < eps or abs(b["y"] + b["h"] - a["y"]) < eps:
        return min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
    return 0.0


def spanning_tree_plus(candidates: nx.Graph, extra_frac, rng) -> nx.Graph:
    """Random spanning tree of ``candidates`` + ``extra_frac`` of its other edges."""
    for u, v in candidates.edges:
        candidates[u][v]["w"] = float(rng.random())
    G = nx.Graph(nx.minimum_spanning_tree(candidates, weight="w"))
    G.add_nodes_from(candidates.nodes)
    others = [e for e in candidates.edges if not G.has_edge(*e)]
    k = int(extra_frac * len(others))
    G.add_edges_from(others[j] for j in sorted(rng.choice(len(others), size=k, replace=False)))
    return G


def generate_floorplan_graph(n_regions, door_extra, corridor_bias, zone_depth, region_area,
                             rng, max_retries=100) -> tuple[nx.Graph, dict, dict]:
    """BSP floor plan -> (door graph, room centroids in metres, zone per node).

    Candidate doors are shared walls >= MIN_WALL; doors = a random spanning
    tree of the candidates + ``door_extra`` of the rest. A plan whose
    candidate graph is disconnected (a sliver room with no usable wall) is
    redrawn, like generate_region_graph's connectivity retries.
    """
    for _ in range(max_retries):
        rooms = bsp_rooms(n_regions, corridor_bias, zone_depth, rng)
        W = nx.Graph()
        W.add_nodes_from(range(len(rooms)))
        W.add_edges_from(
            (i, j) for i in range(len(rooms)) for j in range(i + 1, len(rooms))
            if _shared_wall(rooms[i], rooms[j]) >= MIN_WALL
        )
        if nx.is_connected(W):
            break
    else:
        raise RuntimeError(f"no connected floor plan after {max_retries} retries")
    G = nx.Graph()
    G.add_nodes_from(range(len(rooms)))
    G.add_edges_from(spanning_tree_plus(W, door_extra, rng).edges)

    # Scale the unit-width plan so the mean room is `region_area` m^2.
    plan_area = 1.0 / PLAN_ASPECT
    scale = float(np.sqrt(region_area * n_regions / plan_area))
    coords = {
        i: [round((r["x"] + r["w"] / 2) * scale, 1), round((r["y"] + r["h"] / 2) * scale, 1)]
        for i, r in enumerate(rooms)
    }
    # Zone = the split-tree prefix at zone_depth (2**zone_depth zones); a
    # corridor carved above that depth is its own zone.
    prefixes = sorted({r["path"][:zone_depth] for r in rooms})
    zone_of = {p: z for z, p in enumerate(prefixes)}
    zones = {i: zone_of[r["path"][:zone_depth]] for i, r in enumerate(rooms)}
    return G, coords, zones


def generate_emst_graph(n_regions, extra_frac, region_area, rng) -> tuple[nx.Graph, dict]:
    """Delaunay EMST + ``extra_frac`` of the other Delaunay edges (§2 fallback).

    Port of ``mst_plus`` (notebook §8b): uniform points in a PLAN_ASPECT
    rectangle, minimum spanning tree by Euclidean length.
    """
    plan_area = 1.0 / PLAN_ASPECT
    scale = float(np.sqrt(region_area * n_regions / plan_area))
    pos = rng.random((n_regions, 2)) * np.array([1.0, 1.0 / PLAN_ASPECT])
    D = nx.Graph()
    D.add_nodes_from(range(n_regions))
    for s in spatial.Delaunay(pos).simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                D.add_edge(int(s[a]), int(s[b]))
    for u, v in D.edges:
        D[u][v]["w"] = float(np.linalg.norm(pos[u] - pos[v]))
    G = nx.Graph()
    G.add_nodes_from(range(n_regions))
    G.add_edges_from(nx.minimum_spanning_tree(D, weight="w").edges)
    others = [e for e in D.edges if not G.has_edge(*e)]
    k = int(extra_frac * len(others))
    G.add_edges_from(others[j] for j in sorted(rng.choice(len(others), size=k, replace=False)))
    coords = {i: [round(float(pos[i, 0]) * scale, 1), round(float(pos[i, 1]) * scale, 1)]
              for i in range(n_regions)}
    return G, coords


def layout_statistics(G: nx.Graph) -> dict:
    """The §2 target statistics of a connected region graph.

    Same definitions as the notebook's ``gen_stats`` / ``graph_stats``:
    dead ends = degree-1 fraction, bridge fraction over edges, cycles per
    region = (E - N + 1) / N.
    """
    deg = np.array([d for _, d in G.degree()])
    n, e = G.number_of_nodes(), G.number_of_edges()
    return {
        "n_regions": n,
        "n_edges": e,
        "mean_degree": float(deg.mean()),
        "dead_end_frac": float(np.mean(deg == 1)),
        "bridge_frac": len(list(nx.bridges(G))) / e,
        "cycles_per_region": (e - n + 1) / n,
        "diameter": nx.diameter(G),
        "clustering": nx.average_clustering(G),
    }


def generate_objects(region_coords, object_rate, rng) -> tuple[list[dict], list[list[str]]]:
    """Poisson-sample objects per region with jittered coords."""
    objects = []
    object_connections = []
    obj_counter = 1

    for region_idx, region_coord in region_coords.items():
        n_objects = int(rng.poisson(object_rate))
        region_name = f"region_{region_idx + 1}"
        for _ in range(n_objects):
            obj_name = f"object_{obj_counter}"
            ox = round(float(region_coord[0] + rng.normal(0, 2.0)), 1)
            oy = round(float(region_coord[1] + rng.normal(0, 2.0)), 1)
            objects.append({"name": obj_name, "coords": [ox, oy], "description": ""})
            object_connections.append([obj_name, region_name])
            obj_counter += 1

    return objects, object_connections


def assign_descriptions(objects, description_prob, rng) -> list[dict]:
    """Mark a random subset of objects with __FILL__ placeholder descriptions."""
    for obj in objects:
        if rng.random() < description_prob:
            obj["description"] = "__FILL__"
    return objects


def build_skeleton_json(G, region_coords, community_assignments, objects,
                        object_connections, n_tasks, params) -> dict:
    """Assemble the full skeleton JSON matching eval_1_multi_step.json schema."""
    regions = []
    for node in G.nodes:
        regions.append({
            "name": f"region_{node + 1}",
            "coords": region_coords[node],
            "description": "",
        })

    region_connections = []
    for u, v in G.edges:
        region_connections.append([f"region_{u + 1}", f"region_{v + 1}"])

    community_map = {f"region_{i + 1}": community_assignments[i] for i in range(len(G))}

    return {
        "graph": {
            "objects": objects,
            "regions": regions,
            "object_connections": object_connections,
            "region_connections": region_connections,
            "robot_location": "region_1",
        },
        "tasks": [],
        "_metadata": {
            "n_communities": params["n_communities"],
            "nodes_per_community": params["nodes_per_community"],
            "intra_community_prob": params["intra_community_prob"],
            "inter_community_prob": params["inter_community_prob"],
            "object_rate": params["object_rate"],
            "description_prob": params["description_prob"],
            "community_assignments": community_map,
            "n_tasks": n_tasks,
            "seed": params["seed"],
            "n_regions": len(G),
            "n_objects": len(objects),
        },
    }


def build_spatial_skeleton_json(G, region_coords, zones, objects, object_connections,
                                n_tasks, robot_location, params) -> dict:
    """Skeleton for the floorplan/emst layouts: same graph schema as
    build_skeleton_json, layout-specific ``_metadata``.

    ``zones`` (region -> zone id) is written both as ``_metadata.zones`` (the
    populate prompt's ``<zone>_<kind>_<k>`` naming hint) and as
    ``community_assignments`` / ``n_communities``, which the populate prompt
    already reads; emst has no zones (one community, no ``zones`` key).
    """
    regions = [
        {"name": f"region_{node + 1}", "coords": region_coords[node], "description": ""}
        for node in G.nodes
    ]
    region_connections = [[f"region_{u + 1}", f"region_{v + 1}"] for u, v in G.edges]
    zone_map = {f"region_{i + 1}": zones[i] for i in range(len(G))}
    metadata = {
        "layout": params["layout"],
        "n_regions": len(G),
        "door_extra": params["door_extra"],
    }
    if params["layout"] == "floorplan":
        metadata.update({
            "corridor_bias": params["corridor_bias"],
            "zone_depth": params["zone_depth"],
            "zones": zone_map,
        })
    metadata.update({
        "n_communities": len(set(zones.values())),
        "community_assignments": zone_map,
        "region_area_m2": params["region_area"],
        "object_rate": params["object_rate"],
        "description_prob": params["description_prob"],
        "robot_location_mode": params["robot_location"],
        "n_tasks": n_tasks,
        "seed": params["seed"],
        "n_objects": len(objects),
    })
    return {
        "graph": {
            "objects": objects,
            "regions": regions,
            "object_connections": object_connections,
            "region_connections": region_connections,
            "robot_location": robot_location,
        },
        "tasks": [],
        "_metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a structurally valid eval graph skeleton for PRISM."
    )
    parser.add_argument("--layout", choices=LAYOUTS, default="sbm",
                        help="sbm = stochastic block model (historical, byte-identical); "
                             "floorplan = e22 BSP floor plan; emst = Delaunay EMST + extra edges.")
    parser.add_argument("--n-regions", type=int, default=100,
                        help="floorplan/emst: number of regions (sbm uses communities x nodes).")
    parser.add_argument("--door-extra-min", type=float, default=0.05,
                        help="floorplan/emst: per-graph fraction of the non-tree candidate "
                             "edges added as extra doors is drawn from U[min, max].")
    parser.add_argument("--door-extra-max", type=float, default=0.20)
    parser.add_argument("--corridor-bias", type=float, default=0.3,
                        help="floorplan: probability that a split above --zone-depth carves "
                             "a corridor room between its halves.")
    parser.add_argument("--zone-depth", type=int, default=3,
                        help="floorplan: zones = BSP subtrees at this depth (2**depth zones).")
    parser.add_argument("--region-area", type=float, default=25.0,
                        help="floorplan/emst: mean room area in m^2 (hospital_v4: 24.6); "
                             "sets the coordinate scale.")
    parser.add_argument("--robot-location", choices=("region_1", "random"), default="region_1",
                        help="region_1 = historical; random = a uniformly drawn region "
                             "(drawn last, so the rest of the skeleton is unchanged).")
    parser.add_argument("--n-communities", type=int, default=3)
    parser.add_argument("--nodes-per-community", type=int, default=5)
    parser.add_argument("--intra-community-prob", type=float, default=0.6)
    parser.add_argument("--inter-community-prob", type=float, default=0.05)
    parser.add_argument("--object-rate", type=float, default=0.3)
    parser.add_argument("--description-prob", type=float, default=0.05)
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--output", type=str, default="data/eval/generated_graph.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.layout != "sbm":
        door_extra = float(rng.uniform(args.door_extra_min, args.door_extra_max))
        if args.layout == "floorplan":
            G, region_coords, zones = generate_floorplan_graph(
                args.n_regions, door_extra, args.corridor_bias, args.zone_depth,
                args.region_area, rng,
            )
        else:
            G, region_coords = generate_emst_graph(args.n_regions, door_extra, args.region_area, rng)
            zones = {i: 0 for i in G.nodes}
        objects, object_connections = generate_objects(region_coords, args.object_rate, rng)
        objects = assign_descriptions(objects, args.description_prob, rng)
        robot_location = (
            f"region_{int(rng.integers(len(G))) + 1}" if args.robot_location == "random"
            else "region_1"
        )
        params = {
            "layout": args.layout,
            "door_extra": round(door_extra, 4),
            "corridor_bias": args.corridor_bias,
            "zone_depth": args.zone_depth,
            "region_area": args.region_area,
            "object_rate": args.object_rate,
            "description_prob": args.description_prob,
            "robot_location": args.robot_location,
            "seed": args.seed,
        }
        skeleton = build_spatial_skeleton_json(
            G, region_coords, zones, objects, object_connections,
            args.n_tasks, robot_location, params,
        )
        with open(args.output, "w") as f:
            json.dump(skeleton, f, indent=2)
        stats = layout_statistics(G)
        print(f"Generated {args.layout} skeleton: {len(G)} regions, {len(objects)} objects, "
              f"{skeleton['_metadata']['n_communities']} zones, door_extra {door_extra:.3f}, "
              f"degree {stats['mean_degree']:.2f}, diameter {stats['diameter']}")
        print(f"Saved to {args.output}")
        return

    G, community_assignments = generate_region_graph(
        args.n_communities, args.nodes_per_community,
        args.intra_community_prob, args.inter_community_prob, rng,
    )
    region_coords = assign_coordinates(G, community_assignments, args.n_communities, rng)
    objects, object_connections = generate_objects(region_coords, args.object_rate, rng)
    objects = assign_descriptions(objects, args.description_prob, rng)

    params = {
        "n_communities": args.n_communities,
        "nodes_per_community": args.nodes_per_community,
        "intra_community_prob": args.intra_community_prob,
        "inter_community_prob": args.inter_community_prob,
        "object_rate": args.object_rate,
        "description_prob": args.description_prob,
        "seed": args.seed,
    }
    skeleton = build_skeleton_json(
        G, region_coords, community_assignments, objects,
        object_connections, args.n_tasks, params,
    )
    if args.robot_location == "random":
        skeleton["graph"]["robot_location"] = f"region_{int(rng.integers(len(G))) + 1}"

    with open(args.output, "w") as f:
        json.dump(skeleton, f, indent=2)

    n_regions = len(G)
    n_objects = len(objects)
    print(f"Generated skeleton: {n_regions} regions, {n_objects} objects, "
          f"{args.n_communities} communities")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
