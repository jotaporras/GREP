import argparse
import json

import networkx as nx
import numpy as np


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


def main():
    parser = argparse.ArgumentParser(
        description="Generate a structurally valid eval graph skeleton for PRISM."
    )
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

    with open(args.output, "w") as f:
        json.dump(skeleton, f, indent=2)

    n_regions = len(G)
    n_objects = len(objects)
    print(f"Generated skeleton: {n_regions} regions, {n_objects} objects, "
          f"{args.n_communities} communities")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
