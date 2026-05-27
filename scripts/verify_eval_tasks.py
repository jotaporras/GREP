"""Deterministically solve each task in e6_transferability eval graphs and verify
that (a) the task is solvable and (b) the answer regex matches the computed answer.

Usage:
    python scripts/verify_eval_tasks.py [--dir data/eval/e6_transferability]
"""

import argparse
import json
import re
from pathlib import Path

import networkx as nx


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(graph_data: dict) -> tuple[nx.Graph, dict[str, str]]:
    """Return (G over regions, obj2region mapping).

    Nodes are region names.  Edges come from region_connections.
    obj2region maps each object name to its containing region via
    object_connections.
    """
    G = nx.Graph()
    for r in graph_data["regions"]:
        G.add_node(r["name"])
    for u, v in graph_data["region_connections"]:
        G.add_edge(u, v)

    obj2region: dict[str, str] = {}
    for obj_name, region_name in graph_data["object_connections"]:
        obj2region[obj_name] = region_name

    return G, obj2region


def region2objects(obj2region: dict[str, str]) -> dict[str, list[str]]:
    r2o: dict[str, list[str]] = {}
    for obj, reg in obj2region.items():
        r2o.setdefault(reg, []).append(obj)
    return r2o


# ---------------------------------------------------------------------------
# Acceptance-criterion parsers  (each returns a computed answer string or None)
# ---------------------------------------------------------------------------

def solve_object_location(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'identifies X as the region containing Y'"""
    m = re.search(r"identifies (\S+) as the region containing (\S+?)\.?$", criterion)
    if not m:
        return None
    expected_region, obj = m.group(1), m.group(2)
    actual = obj2region.get(obj)
    if actual != expected_region:
        return f"MISMATCH: {obj} is in {actual}, criterion says {expected_region}"
    return expected_region


def solve_max_degree(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'identifies X as the area with the most direct connections (N neighbors)'"""
    m = re.search(r"identifies (\S+) as the area with the most direct connections \((\d+) neighbors", criterion)
    if not m:
        return None
    expected_node, expected_deg = m.group(1), int(m.group(2))
    max_deg = max(G.degree(n) for n in G.nodes)
    max_nodes = [n for n in G.nodes if G.degree(n) == max_deg]
    if expected_node not in max_nodes:
        return f"MISMATCH: max degree nodes are {max_nodes} (deg={max_deg}), criterion says {expected_node}({expected_deg})"
    if max_deg != expected_deg:
        return f"MISMATCH: {expected_node} has degree {G.degree(expected_node)}, criterion says {expected_deg}"
    if len(max_nodes) > 1:
        return f"TIE: {expected_node} is one of {len(max_nodes)} nodes with degree {max_deg}: {max_nodes}"
    return expected_node


def solve_min_degree(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'identifies X as the area with the fewest neighbors (N connections: ...)'"""
    m = re.search(r"identifies (\S+) as the area with the fewest neighbors", criterion)
    if not m:
        return None
    expected_node = m.group(1)
    actual_node = min(G.nodes, key=lambda n: G.degree(n))
    if actual_node != expected_node:
        return f"MISMATCH: min degree is {actual_node}({G.degree(actual_node)}), criterion says {expected_node}"
    return expected_node


def solve_colocated_objects(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'identifies X as the area containing Y and Z' or 'containing exactly N objects (list)'"""
    # Pattern: "identifies X as the area containing obj1 and obj2"
    m = re.search(r"identifies (\S+) as the (?:only )?area containing (?:both )?(\S+) and (\S+?)\.?$", criterion)
    if m:
        expected_region = m.group(1)
        obj1, obj2 = m.group(2), m.group(3)
        r1 = obj2region.get(obj1)
        r2 = obj2region.get(obj2)
        if r1 == r2 == expected_region:
            return expected_region
        return f"MISMATCH: {obj1}->{r1}, {obj2}->{r2}, criterion says {expected_region}"

    # Pattern: "identifies X as the only area containing exactly three objects (a, b, c)"
    m = re.search(r"identifies (\S+) as the only area containing exactly (\w+) objects \(([^)]+)\)", criterion)
    if m:
        expected_region = m.group(1)
        r2o = region2objects(obj2region)
        objs_listed = [o.strip() for o in m.group(3).split(",")]
        actual_objs = sorted(r2o.get(expected_region, []))
        if sorted(objs_listed) == actual_objs:
            return expected_region
        return f"MISMATCH: {expected_region} has {actual_objs}, criterion lists {objs_listed}"

    return None


def solve_count_unique_regions(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'states N, since obj1 (reg1), obj2 (reg2), ... each occupy a unique area'"""
    m = re.search(r"states (\d+), since (.+) each occupy a unique area", criterion)
    if not m:
        return None
    expected_count = int(m.group(1))
    r2o = region2objects(obj2region)
    unique_regions = [r for r, objs in r2o.items() if len(objs) == 1]
    if len(unique_regions) != expected_count:
        return f"MISMATCH: {len(unique_regions)} unique-object regions, criterion says {expected_count}"
    return str(expected_count)


def solve_count_regions_with_n_objects(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'states N, since X and Y are the only areas with exactly K objects each'"""
    m = re.search(r"states (\d+), since (.+) are the only areas with exactly (\w+) objects each", criterion)
    if not m:
        return None
    expected_count = int(m.group(1))
    r2o = region2objects(obj2region)
    word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
    k_str = m.group(3)
    k = word_to_num.get(k_str, int(k_str) if k_str.isdigit() else None)
    if k is None:
        return f"PARSE_ERROR: cannot parse object count '{k_str}'"
    matching = [r for r, objs in r2o.items() if len(objs) == k]
    if len(matching) != expected_count:
        return f"MISMATCH: {len(matching)} regions with {k} objects, criterion says {expected_count}"
    return str(expected_count)


def solve_adjacency_yes(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'states yes because X is directly connected to Y' or 'states yes, since X is directly connected to Y'"""
    m = re.search(r"states yes[,.]? (?:because|since) (\S+)(?: \([^)]*\))? is directly connected to (\S+?)(?:\s*\([^)]*\))?\.?$", criterion)
    if not m:
        return None
    u, v = m.group(1), m.group(2)
    u_region = obj2region.get(u, u)
    v_region = obj2region.get(v, v)
    if G.has_edge(u_region, v_region):
        return "yes"
    return f"MISMATCH: {u_region} and {v_region} are NOT adjacent, criterion says yes"


def solve_adjacency_no(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'states no because X is not directly connected to Y' or 'states no, since ...'"""
    m = re.search(r"states no[,.]? (?:because|since) (\S+) is not directly connected to (\S+?)\.?$", criterion)
    if not m:
        return None
    u, v = m.group(1), m.group(2)
    u_region = obj2region.get(u, u)
    v_region = obj2region.get(v, v)
    if not G.has_edge(u_region, v_region):
        return "no"
    return f"MISMATCH: {u_region} and {v_region} ARE adjacent, criterion says no"


def solve_path(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'confirms a route exists and outputs a path whose first hop leaves X
    and whose last hop reaches Y'"""
    m = re.search(
        r"confirms a route exists and outputs a path whose first hop leaves (\S+) "
        r"and whose last hop reaches (\S+?)\.?$",
        criterion,
    )
    if not m:
        return None
    src, dst = m.group(1), m.group(2)
    if not G.has_node(src) or not G.has_node(dst):
        return f"MISMATCH: {src} or {dst} not in graph"
    if not nx.has_path(G, src, dst):
        return f"MISMATCH: no path from {src} to {dst}"
    path = nx.shortest_path(G, src, dst)
    return " -> ".join(path)


def solve_path_through(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'confirms a route exists and outputs a path from X through Y to Z'"""
    m = re.search(r"(?:confirms|states).*(?:path|route) from (\S+) through (\S+) to (\S+?)\.?$", criterion)
    if not m:
        return None
    src, via, dst = m.group(1), m.group(2), m.group(3)
    for node in (src, via, dst):
        if not G.has_node(node):
            return f"MISMATCH: {node} not in graph"
    if not nx.has_path(G, src, via) or not nx.has_path(G, via, dst):
        return f"MISMATCH: no path from {src} through {via} to {dst}"
    p1 = nx.shortest_path(G, src, via)
    p2 = nx.shortest_path(G, via, dst)
    full = p1 + p2[1:]
    return " -> ".join(full)


def solve_path_avoid(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'outputs a path from X to Y avoiding Z' or 'that does not include Z' or 'that avoids Z, such as ...'"""
    m = re.search(r"(?:path|route).*from (\S+) to (\S+) (?:that (?:does not include|avoids)|avoiding) (\S+?)[\.,]", criterion)
    if not m:
        m = re.search(r"(?:path|route) from (\S+) to (\S+) (?:that (?:does not include|avoids)|avoiding) (\S+?)\.?$", criterion)
    if not m:
        return None
    src, dst, avoid = m.group(1), m.group(2), m.group(3)
    # Remove the avoided node temporarily
    H = G.copy()
    if avoid in H:
        H.remove_node(avoid)
    if not H.has_node(src) or not H.has_node(dst):
        return f"MISMATCH: {src} or {dst} removed or not in graph"
    if nx.has_path(H, src, dst):
        path = nx.shortest_path(H, src, dst)
        return " -> ".join(path)
    return f"MISMATCH: no path from {src} to {dst} avoiding {avoid}, but criterion expects one"


def solve_no_path_avoid(criterion: str, G: nx.Graph, obj2region: dict) -> str | None:
    """'states no because all paths from X to Y must pass through Z'"""
    m = re.search(r"states no because all paths from (\S+) to (\S+) must pass through (\S+?)\.?$", criterion)
    if not m:
        return None
    src, dst, required = m.group(1), m.group(2), m.group(3)
    H = G.copy()
    if required in H:
        H.remove_node(required)
    if not H.has_node(src) or not H.has_node(dst):
        return "no"
    if nx.has_path(H, src, dst):
        alt = nx.shortest_path(H, src, dst)
        return f"MISMATCH: path exists avoiding {required}: {' -> '.join(alt)}"
    return "no"


SOLVERS = [
    solve_object_location,
    solve_max_degree,
    solve_min_degree,
    solve_colocated_objects,
    solve_count_unique_regions,
    solve_count_regions_with_n_objects,
    solve_adjacency_yes,
    solve_adjacency_no,
    solve_path,
    solve_path_through,
    solve_path_avoid,
    solve_no_path_avoid,
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def verify_task(task: dict, G: nx.Graph, obj2region: dict) -> dict:
    criterion = task["acceptance_criterion"]
    answer_regex = task["answer"]

    for solver in SOLVERS:
        result = solver(criterion, G, obj2region)
        if result is not None:
            break
    else:
        return {
            "task": task["task"],
            "status": "UNRECOGNIZED",
            "detail": f"No solver matched criterion: {criterion}",
        }

    if result.startswith("MISMATCH"):
        return {
            "task": task["task"],
            "status": "WRONG_ANSWER",
            "detail": result,
            "criterion": criterion,
        }

    if re.search(answer_regex, result):
        return {
            "task": task["task"],
            "status": "PASS",
            "computed": result,
        }
    else:
        return {
            "task": task["task"],
            "status": "REGEX_MISMATCH",
            "computed": result,
            "regex": answer_regex,
            "criterion": criterion,
        }


def verify_file(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    G, obj2region = build_graph(data["graph"])
    results = []
    for task in data["tasks"]:
        results.append(verify_task(task, G, obj2region))
    return results


def main():
    parser = argparse.ArgumentParser(description="Verify e6 transferability eval tasks")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("data/eval/e6_transferability"),
        help="Directory containing eval graph JSON files",
    )
    args = parser.parse_args()

    files = sorted(args.dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in {args.dir}")
        return

    total_pass = 0
    total_fail = 0
    total_tasks = 0

    for f in files:
        results = verify_file(f)
        passes = sum(1 for r in results if r["status"] == "PASS")
        fails = len(results) - passes
        total_pass += passes
        total_fail += fails
        total_tasks += len(results)

        tag = "OK" if fails == 0 else "FAIL"
        print(f"[{tag}] {f.name}: {passes}/{len(results)} passed")

        for r in results:
            if r["status"] != "PASS":
                print(f"  {r['status']}: {r['task']}")
                for k in ("detail", "computed", "regex", "criterion"):
                    if k in r:
                        print(f"    {k}: {r[k]}")

    print(f"\nTotal: {total_pass}/{total_tasks} passed, {total_fail} failed")


if __name__ == "__main__":
    main()
