"""Deterministic smoke test: are a graph file's tasks actually solvable?

Builds the undirected scene graph from a `graph.json` (regions + objects as
nodes; region_connections + object_connections as edges) and, for every task,
checks the claim encoded in its ``acceptance_criterion`` against ground-truth
connectivity via plain BFS — no LLM, no networkx, fully deterministic.

Claim types recognised from the acceptance_criterion text:
  * reachability   : "<T> is reachable from <S>."            -> path S->T exists
  * via-path       : "path from <S> to <T> via <V1> and <V2>" -> S,T,Vi all in
                     one component (an undirected walk through them exists)
  * avoid-node     : "path to <T> that does not use <X> as an intermediate"
                     -> T reachable from a neighbour of S without revisiting X

Also sanity-checks that each task's ``answer`` regex matches a correct answer.

Usage:  python scripts/smoke_test_graph_solvable.py [path/to/graph.json]
Exits nonzero if any task is unsolvable or any answer-regex check fails.
"""

import json
import re
import sys
from collections import deque


def build_adjacency(graph: dict):
    nodes = {n["name"] for n in graph["objects"]} | {n["name"] for n in graph["regions"]}
    adj = {n: set() for n in nodes}
    bad_edges = []
    for key in ("object_connections", "region_connections"):
        for a, b in graph.get(key, []):
            if a not in nodes or b not in nodes:
                bad_edges.append((a, b))
                continue
            adj[a].add(b)
            adj[b].add(a)
    return nodes, adj, bad_edges


def bfs_path(adj, src, dst, blocked=frozenset()):
    """BFS shortest path src->dst avoiding `blocked` nodes; None if unreachable."""
    if src not in adj or dst not in adj or src in blocked:
        return None
    prev = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        if cur == dst:
            path = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            return path[::-1]
        for nxt in adj[cur]:
            if nxt not in prev and nxt not in blocked:
                prev[nxt] = cur
                q.append(nxt)
    return None


def reachable(adj, src, dst, blocked=frozenset()):
    return bfs_path(adj, src, dst, blocked) is not None


def same_component(adj, nodes_needed):
    """True if every node in `nodes_needed` is mutually connected."""
    nodes_needed = [n for n in nodes_needed if n]
    if not nodes_needed:
        return False
    root = nodes_needed[0]
    return all(reachable(adj, root, n) for n in nodes_needed)


NODE_RE = re.compile(r"[a-z][a-z_]*_\d+")


def classify_and_check(adj, task):
    crit = task.get("acceptance_criterion", "")
    src = task.get("init_node")
    ids = NODE_RE.findall(crit)

    # avoid-node: "...path to <T> that does not use <X> as an intermediate..."
    m = re.search(r"path to (\w+_\d+) that does not use (\w+_\d+)", crit)
    if m:
        target, excluded = m.group(1), m.group(2)
        witness = None
        for nb in adj.get(src, ()):
            if nb == excluded:
                continue
            p = bfs_path(adj, nb, target, blocked={excluded})
            if p:
                witness = [src, *p]
                break
        return "avoid-node", f"{src}->{target} avoiding {excluded}", witness is not None, witness or []

    # via-path: "...path from <S> to <T> via ..."
    m = re.search(r"path from (\w+_\d+) to (\w+_\d+)\s+via\s+(.+)", crit)
    if m:
        s, target, via_str = m.group(1), m.group(2), m.group(3)
        vias = NODE_RE.findall(via_str)
        ok = same_component(adj, [s, target, *vias])
        return "via-path", f"{s}->{target} via {vias}", ok, [s, *vias, target]

    # reachability: "...<T> is reachable from <S>."
    m = re.search(r"(\w+_\d+) is reachable from (\w+_\d+)", crit)
    if m:
        target, s = m.group(1), m.group(2)
        p = bfs_path(adj, s, target)
        return "reachability", f"{s}->{target}", p is not None, p or []

    # Fallback: treat first two ids as src/target reachability.
    if len(ids) >= 2:
        p = bfs_path(adj, ids[-1], ids[0])
        return "reachability?", f"{ids[-1]}->{ids[0]}", p is not None, p or []
    return "UNPARSED", crit[:60], False, []


def answer_regex_ok(task, witness_nodes):
    """Does the answer regex match a correct response for a solvable task?

    A correct answer is either affirmative ("yes") or names a node on an actual
    valid path (the BFS witness). Testing against the real path — not the
    criterion text — avoids false negatives when the regex lists intermediate
    nodes rather than the source/target.
    """
    pattern = task.get("answer", "")
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return False, f"bad regex: {e}"
    if rx.search("yes"):
        return True, "matches 'yes'"
    matched = [n for n in witness_nodes if rx.search(n)]
    if matched:
        return True, f"matches route node ({matched[0]})"
    return False, "matches neither 'yes' nor any node on a valid path"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/graph.json"
    with open(path) as f:
        doc = json.load(f)
    graph, tasks = doc["graph"], doc["tasks"]

    nodes, adj, bad_edges = build_adjacency(graph)
    n_edges = sum(len(v) for v in adj.values()) // 2
    # connected-component count
    seen, comps = set(), 0
    for n in nodes:
        if n not in seen:
            comps += 1
            seen.add(n)
            q = deque([n])
            while q:
                c = q.popleft()
                for nb in adj[c]:
                    if nb not in seen:
                        seen.add(nb)
                        q.append(nb)

    print(f"GRAPH: {len(graph['regions'])} regions, {len(graph['objects'])} objects, "
          f"{n_edges} undirected edges, {comps} connected component(s)")
    if bad_edges:
        print(f"  ⚠ {len(bad_edges)} edge(s) reference unknown nodes: {bad_edges}")
    print(f"TASKS: {len(tasks)}\n")

    n_fail = 0
    for i, task in enumerate(tasks):
        ttype, claim, graph_ok, witness = classify_and_check(adj, task)
        rx_ok, rx_msg = answer_regex_ok(task, witness)
        solvable = graph_ok and rx_ok
        n_fail += not solvable
        flag = "PASS" if solvable else "FAIL"
        print(f"[{flag}] task {i:2d} | {ttype:13s} | {claim}")
        if not graph_ok:
            print(f"         ↳ GRAPH: claim is FALSE in the graph")
        if not rx_ok:
            print(f"         ↳ REGEX: {rx_msg}")

    print(f"\nSUMMARY: {len(tasks) - n_fail}/{len(tasks)} tasks deterministically solvable")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
