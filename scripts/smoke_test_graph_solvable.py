"""Deterministic, graph-agnostic smoke test: are a graph file's tasks solvable & verifiable?

Works on ANY ``{graph, tasks}`` JSON regardless of how the ``acceptance_criterion``
is phrased or which answer-regex dialect is used. All connectivity is computed with
NetworkX on the undirected scene graph (regions + objects as nodes; object_connections
+ region_connections as edges). No LLM, fully deterministic.

For every task two independent checks run:

  GRAPH  (solvability)   – every entity the answer depends on exists in the graph and
                           is reachable from ``init_node`` (honouring any "without /
                           avoid / does not use" constraint named in the criterion);
                           every route encoded in a path-style answer is a real ordered
                           edge-walk; deny ("cannot reach …") claims are verified when a
                           constraint is parseable, otherwise reported as unverified.

  REGEX  (verifiability) – the ``answer`` regex ACCEPTS a battery of plausible correct
                           answers and REJECTS a battery of plausible wrong ones
                           (wrong polarity, premise echo, or an invalid listed route).

Verdict per task:
  PASS  – solvable and the regex cleanly separates correct from incorrect answers.
  WARN  – solvable, no leak, but the regex is narrow (misses some correct phrasings)
          or a deny-claim could not be verified from the graph alone.
  FAIL  – an entity is missing/unreachable, a route is non-traversable, the claim is
          false in the graph, or the regex rewards a wrong answer.

Usage:
  python scripts/smoke_test_graph_solvable.py <graph.json | dir-of-data_gen_*.json>

Exits nonzero if any task FAILs.
"""

import glob
import json
import os
import re
import sys
from collections import deque

try:
    import networkx as nx           # canonical engine (the project env has it)
    HAVE_NX = True
except ImportError:                 # graceful fallback so the tool runs anywhere
    nx = None
    HAVE_NX = False

# ---------------------------------------------------------------------------- graph
NODE_RE = re.compile(r"[a-z][a-z_]*_\d+")


class SceneGraph:
    """Undirected scene graph. Uses NetworkX for connectivity when available,
    otherwise an equivalent pure-Python BFS — identical results either way."""

    def __init__(self, nodes, edges):
        self.adj = {n: set() for n in nodes}
        for a, b in edges:
            self.adj[a].add(b)
            self.adj[b].add(a)
        self._nx = None
        if HAVE_NX:
            self._nx = nx.Graph()
            self._nx.add_nodes_from(nodes)
            self._nx.add_edges_from(edges)

    def __contains__(self, n):
        return n in self.adj

    def has_edge(self, a, b):
        return b in self.adj.get(a, ())

    def num_edges(self):
        return sum(len(v) for v in self.adj.values()) // 2

    def num_nodes(self):
        return len(self.adj)

    def has_path(self, src, dst, blocked=frozenset()):
        if src not in self.adj or dst not in self.adj or src in blocked or dst in blocked:
            return False
        if self._nx is not None:
            if not blocked:
                return nx.has_path(self._nx, src, dst)
            H = self._nx.subgraph([n for n in self._nx if n not in blocked])
            return nx.has_path(H, src, dst)
        seen, q = {src}, deque([src])      # pure-Python BFS
        while q:
            cur = q.popleft()
            if cur == dst:
                return True
            for nb in self.adj[cur]:
                if nb not in seen and nb not in blocked:
                    seen.add(nb); q.append(nb)
        return False

    def num_components(self):
        if self._nx is not None:
            return nx.number_connected_components(self._nx) if self.num_nodes() else 0
        seen, comps = set(), 0
        for n in self.adj:
            if n not in seen:
                comps += 1
                seen.add(n); q = deque([n])
                while q:
                    cur = q.popleft()
                    for nb in self.adj[cur]:
                        if nb not in seen:
                            seen.add(nb); q.append(nb)
        return comps


def build_graph(graph: dict):
    nodes = {n["name"] for n in graph.get("objects", [])} | {n["name"] for n in graph.get("regions", [])}
    edges, bad_edges = [], []
    for key in ("object_connections", "region_connections"):
        for a, b in graph.get(key, []):
            if a not in nodes or b not in nodes:
                bad_edges.append((a, b))
                continue
            edges.append((a, b))
    return SceneGraph(nodes, edges), nodes, bad_edges


def reachable(G, src, dst, blocked=frozenset()):
    return G.has_path(src, dst, blocked)


def walk_valid(G, seq):
    """True iff `seq` is an ordered walk along real edges."""
    return len(seq) >= 2 and all(G.has_edge(seq[i], seq[i + 1]) for i in range(len(seq) - 1))


# ---------------------------------------------------------------------------- parsing
DENY_CUES = re.compile(
    r"\b(?:den(?:y|ies|ied)|cannot|can't|not\s+reachable|unreachable|no\s+path|"
    r"no\s+route|impossible|not\s+possible|is\s+itself)\b", re.I)
ALREADY_CUES = re.compile(r"\balready\b|already\s+(?:there|at)", re.I)
CONSTRAINT_RE = re.compile(
    r"(?:without\s+(?:using|passing\s+through|going\s+through)|does\s+not\s+use|"
    r"avoid(?:s|ing)?|excluding|not\s+via|not\s+using)\b(.*?)"
    # stop at a clause boundary so illustrative "..., such as via X" route hints
    # are not swept into the avoid-set (which would over-block reachability).
    r"(?:[.,;]|\bsuch\s+as\b|\bfor\s+example\b|\be\.g\.|$)", re.I)


def ids_in(text, nodes):
    """Graph node-ids referenced in `text`, in order, de-duplicated."""
    seen, out = set(), []
    for t in NODE_RE.findall(text or ""):
        if t in nodes and t not in seen:
            seen.add(t); out.append(t)
    return out


def phantom_ids(text, nodes):
    return sorted({t for t in NODE_RE.findall(text or "") if t not in nodes})


def route_alternatives(answer, nodes):
    """Node-sequences encoded by a path-style regex (handles '->' and '.*')."""
    alts = []
    for chunk in (answer or "").split("|"):
        seq = [t for t in NODE_RE.findall(chunk) if t in nodes]
        if len(seq) >= 2:
            alts.append(seq)
    return alts


def excluded_nodes(text, nodes):
    ex = set()
    for m in CONSTRAINT_RE.finditer(text or ""):
        for t in NODE_RE.findall(m.group(1)):
            if t in nodes:
                ex.add(t)
    return ex


def task_type(answer, criterion, nodes):
    if route_alternatives(answer, nodes):
        return "path"
    if DENY_CUES.search(criterion or ""):
        return "deny"
    if ALREADY_CUES.search(criterion or ""):
        return "already"
    return "affirm"


# ---------------------------------------------------------------------------- batteries
def affirm_battery(target):
    t = target or "the area"
    correct = [
        f"Yes, the robot can reach {t}.",
        f"{t} is reachable from the start.",
        f"Yes, a path to {t} exists.",
        f"The robot can reach {t}.",
        f"Yes, it is possible to reach {t}.",
        f"Affirmative, there is a route to {t}.",
    ]
    wrong = [
        f"No, the robot cannot reach {t}.",
        f"No, it is not possible to reach {t}.",
        f"No, there is no path to {t}.",
        f"No, {t} is unreachable.",
        f"It is impossible to reach {t}.",
        f"The robot is unable to reach {t}.",
    ]
    return correct, wrong


def already_battery(target):
    c, w = affirm_battery(target)
    t = target or "the area"
    return c + [f"Yes, the robot is already there.", f"The robot is already at {t}."], w


def path_battery(valid_route, invalid_routes):
    arrow = " -> ".join(valid_route)
    correct = [
        f"The shortest path is {arrow}.",
        "Go " + " then ".join(valid_route) + ".",
        "Route: " + ", ".join(valid_route) + ".",
    ]
    wrong = [
        "No, there is no path.",
        "The robot cannot reach it.",
        "Path: " + " -> ".join(reversed(valid_route)) + ".",
    ]
    for bad in invalid_routes:                       # a listed-but-non-traversable route
        wrong.append("The route is " + " -> ".join(bad) + ".")
    return correct, wrong


def regex_scores(pattern, correct, wrong):
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return None, None, f"regex won't compile: {e}"
    acc = sum(bool(rx.search(s)) for s in correct) / len(correct)
    rej = sum(not rx.search(s) for s in wrong) / len(wrong)
    return acc, rej, ""


# ---------------------------------------------------------------------------- per-task
def check_task(G, nodes, task):
    answer = task.get("answer", "")
    crit = task.get("acceptance_criterion", "") or ""
    text = crit or task.get("task", "")
    init = task.get("init_node")
    ttype = task_type(answer, crit, nodes)

    phantoms = phantom_ids(crit, nodes)
    excluded = excluded_nodes(text, nodes)
    refs = [n for n in ids_in(text, nodes) if n != init and n not in excluded]
    target = refs[0] if refs else None

    graph_ok, graph_note, verified = True, "", True

    if init is not None and init not in G:
        return verdict(ttype, False, f"init_node '{init}' not in graph", None, None, "")
    if phantoms:
        return verdict(ttype, False, f"phantom entities {phantoms}", None, None, "")

    if ttype == "path":
        alts = route_alternatives(answer, nodes)
        valid = [a for a in alts if walk_valid(G, a)]
        invalid = [a for a in alts if not walk_valid(G, a)]
        if not valid:
            return verdict(ttype, False, "no listed route is a valid edge-walk", None, None, "")
        target = valid[0][-1]
        correct, wrong = path_battery(valid[0], invalid)
        if invalid:
            graph_note = f"regex also lists {len(invalid)} non-traversable route(s)"
    elif ttype == "deny":
        # entities must exist & be observable; verify the denial only if a constraint is parseable.
        if excluded and refs:
            blocked_targets = ids_in(text, nodes)
            tgt = next((n for n in blocked_targets if n != init and n not in excluded), None)
            if tgt is not None:
                graph_ok = not reachable(G, init, tgt, excluded)
                target = tgt
                graph_note = "" if graph_ok else "deny claim is FALSE: target IS reachable under the stated constraint"
        else:
            verified = False
            graph_note = "deny claim not graph-verifiable (no explicit constraint); checked entity existence only"
            for n in ids_in(text, nodes):
                if n != init and not reachable(G, init, n):
                    graph_ok, graph_note = False, f"referenced entity '{n}' unreachable from {init}"
                    break
        correct, wrong = affirm_battery(target)
        correct, wrong = wrong, correct                 # deny: a "no" answer is correct
    else:  # affirm / already
        for n in refs:
            if not reachable(G, init, n, excluded):
                graph_ok = False
                graph_note = (f"'{n}' unreachable from {init}"
                              + (f" avoiding {sorted(excluded)}" if excluded else ""))
                break
        correct, wrong = (already_battery(target) if ttype == "already" else affirm_battery(target))

    acc, rej, rx_err = regex_scores(answer, correct, wrong)
    return verdict(ttype, graph_ok, graph_note, acc, rej, rx_err, verified,
                   invalid_route=(ttype == "path" and bool(graph_note)))


def verdict(ttype, graph_ok, graph_note, acc, rej, rx_err, verified=True, invalid_route=False):
    notes = []
    if rx_err:
        return dict(typ=ttype, verdict="FAIL", acc=acc, rej=rej,
                    note=rx_err if not graph_note else f"{graph_note}; {rx_err}")
    leak = (rej is not None and rej < 1.0)
    narrow = (acc is not None and acc < 1.0)
    if not graph_ok:
        v = "FAIL"
    elif leak or invalid_route:
        v = "FAIL"
    elif narrow or not verified:
        v = "WARN"
    else:
        v = "PASS"
    if graph_note:
        notes.append(graph_note)
    if leak:
        notes.append(f"regex matches a WRONG answer (reject={rej:.2f})")
    if narrow:
        notes.append(f"regex misses correct phrasings (accept={acc:.2f})")
    if not verified and v == "WARN" and not graph_note:
        notes.append("deny claim assumed (not graph-verified)")
    return dict(typ=ttype, verdict=v, acc=acc, rej=rej, note="; ".join(notes))


# ---------------------------------------------------------------------------- driver
def audit_file(path):
    with open(path) as f:
        doc = json.load(f)
    G, nodes, bad_edges = build_graph(doc["graph"])
    tasks = doc.get("tasks", [])
    print(f"\n=== {os.path.basename(path)} ===")
    print(f"GRAPH: {len(doc['graph'].get('regions', []))} regions, "
          f"{len(doc['graph'].get('objects', []))} objects, {G.num_edges()} edges, "
          f"{G.num_components()} component(s) | TASKS: {len(tasks)}")
    if bad_edges:
        print(f"  ⚠ {len(bad_edges)} edge(s) reference unknown nodes: {bad_edges}")
    n_fail = 0
    for i, task in enumerate(tasks):
        r = check_task(G, nodes, task)
        n_fail += (r["verdict"] == "FAIL")
        acc = "-" if r["acc"] is None else f"{r['acc']:.2f}"
        rej = "-" if r["rej"] is None else f"{r['rej']:.2f}"
        line = f"[{r['verdict']:4s}] task {i:2d} | {r['typ']:7s} | acc {acc} rej {rej}"
        if r["note"]:
            line += f"  | {r['note']}"
        print(line)
    return n_fail, len(tasks)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "data/graph.json"
    if os.path.isdir(arg):
        files = sorted(glob.glob(os.path.join(arg, "data_gen_*.json")))
        if not files:
            files = sorted(glob.glob(os.path.join(arg, "*.json")))
    else:
        files = [arg]
    print(f"connectivity engine: {'NetworkX ' + nx.__version__ if HAVE_NX else 'builtin BFS (networkx not found)'}")
    total_fail = total = 0
    for f in files:
        try:
            nf, nt = audit_file(f)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"\n=== {os.path.basename(f)} ===\n  SKIP: not a graph/tasks file ({e})")
            continue
        total_fail += nf; total += nt
    print(f"\nSUMMARY: {total - total_fail}/{total} tasks PASS/WARN (no FAIL); {total_fail} FAIL")
    sys.exit(1 if total_fail else 0)


if __name__ == "__main__":
    main()
