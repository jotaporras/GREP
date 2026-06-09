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
# Node ids may carry several numeric segments (e.g. grid names like ``bay_3_26_1``),
# so match one-or-more ``_<int>`` tails — a single-segment ``_\d+`` would truncate
# ``bay_3_26_1`` to ``bay_3`` and wrongly report a phantom entity.
NODE_RE = re.compile(r"[a-z][a-z_]*(?:_\d+)+")


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


def strip_regex(answer):
    """Drop regex word-boundaries / inline flags so NODE_RE reads clean node ids.
    Without this, ``\\bhangar_1`` is scanned as the bogus token ``bhangar_1``."""
    return re.sub(r"\\b|\(\?[a-z]+\)", " ", answer or "")


def answer_node_ids(answer, nodes):
    """Real graph node-ids named (in order) by an answer regex."""
    return [t for t in NODE_RE.findall(strip_regex(answer)) if t in nodes]


def route_alternatives(answer, nodes):
    """Node-sequences encoded by a path-style regex (handles '->' and '.*')."""
    alts = []
    for chunk in strip_regex(answer).split("|"):
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
    """Classify by what the ANSWER is shaped to accept (authoritative), falling back
    to criterion cues only for the affirm/already distinction:

      deny / affirm / already – a yes/no polarity answer
      route                    – >=2 node ids joined by a '.*' wildcard (endpoints /
                                 waypoints only; connectivity, not a literal edge-walk)
      path                     – >=2 node ids with NO wildcard (a full ordered edge-walk)
      identify_node            – a single bare node id ("Where is …", "Which area …")
      identify_count           – a bare integer ("How many …")
    """
    cleaned = strip_regex(answer)
    if re.search(r"(?i)\bno\b", cleaned):
        return "deny"
    if re.search(r"(?i)\byes\b", cleaned):
        return "already" if ALREADY_CUES.search(criterion or "") else "affirm"
    if route_alternatives(answer, nodes):       # >=2 node ids ordered within one branch
        return "route" if ".*" in (answer or "") else "path"
    if [t for t in NODE_RE.findall(cleaned) if t in nodes]:   # bare/alternated node id(s)
        return "identify_node"
    if re.search(r"\d", cleaned):
        return "identify_count"
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
        # two-word negations a `\b`-only template misses (substring guards do not
        # catch "not reachable" / "not able to reach" — only "unreachable").
        f"No, it is not reachable.",
        f"No, the robot is not able to reach {t}.",
    ]
    return correct, wrong


def premise_echo(target):
    """Neutral restatements of the question — NOT a verdict, so they must be
    rejected regardless of task polarity. Exercises the narration-noun leak
    (`a path`/`a route` matching a non-answer). Appended to the WRONG set of
    every task type (never flipped into a 'correct' battery)."""
    t = target or "the area"
    return [
        f"The question is whether there is a path to {t}.",
        f"I must determine if a route to {t} exists.",
    ]


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


def route_reach_battery(waypoints):
    """For endpoint/waypoint answers (``A .* B``): correct answers name the waypoints
    in order; wrong ones deny the route or list them reversed (fails the ordered '.*')."""
    correct = [
        "Yes, a route exists: " + " -> ".join(waypoints) + ".",
        "Go " + " then ".join(waypoints) + ".",
        "Route: " + ", ".join(waypoints) + ".",
    ]
    wrong = [
        "No, there is no route.",
        "The destination is unreachable.",
    ]
    if len(waypoints) >= 2:
        wrong.append("Route: " + ", ".join(reversed(waypoints)) + ".")
    return correct, wrong


def identify_node_battery(target, distractors):
    """For "name the area/object" answers: accept the named node, reject other nodes."""
    t = target or "the area"
    correct = [
        f"The answer is {t}.",
        f"It is {t}.",
        f"{t}.",
        f"The area is {t}.",
    ]
    wrong = [f"The answer is {d}." for d in distractors]
    wrong += ["No, the robot cannot reach it.", "There is no such area."]
    return correct, wrong


def identify_count_battery(num):
    """For "how many" answers: accept the integer, reject neighbouring integers."""
    n = int(num)
    correct = [f"The answer is {num}.", f"{num}", f"There are {num} areas."]
    wrong = [f"The answer is {w}." for w in {str(n + 1), str(n + 2), str(max(0, n - 1))} - {str(n)}]
    wrong += ["No, the robot cannot reach it."]
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
    elif ttype == "route":
        # Endpoint/waypoint answer (A .* B): verify connectivity between consecutive
        # waypoints (a real walk through them exists), not literal adjacency.
        seq = answer_node_ids(answer, nodes)
        wp = [seq[i] for i in range(len(seq)) if i == 0 or seq[i] != seq[i - 1]]
        if len(wp) < 2:
            return verdict(ttype, False, "route answer names fewer than two nodes", None, None, "")
        target = wp[-1]
        broken = next(((wp[i], wp[i + 1]) for i in range(len(wp) - 1)
                       if not reachable(G, wp[i], wp[i + 1])), None)
        if broken:
            return verdict(ttype, False, f"no path {broken[0]} -> {broken[1]} (route infeasible)",
                           None, None, "")
        if init is not None and wp[0] != init and not reachable(G, init, wp[0]):
            graph_ok = False
            graph_note = f"route start {wp[0]} unreachable from init {init}"
        correct, wrong = route_reach_battery(wp)
    elif ttype == "identify_node":
        ids = answer_node_ids(answer, nodes)
        target = ids[0] if ids else None
        if target is None:
            return verdict(ttype, False, "answer names no known node", None, None, "")
        if init is not None and not reachable(G, init, target):
            graph_ok = False
            graph_note = f"identified node '{target}' unreachable from init {init}"
        distractors = [n for n in sorted(nodes) if n != target and n != init][:3]
        correct, wrong = identify_node_battery(target, distractors)
    elif ttype == "identify_count":
        m = re.search(r"\d+", strip_regex(answer))
        if not m:
            return verdict(ttype, False, "count answer has no integer", None, None, "")
        target = m.group(0)
        correct, wrong = identify_count_battery(target)
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
            adj = ADJ_DENY_RE.search(crit)
            cut = CUT_DENY_RE.search(crit)
            if adj and adj.group(1) in G and adj.group(2) in G:
                # criterion makes an explicit adjacency claim ("A is not directly
                # connected to B" / "A is not one move from B") — verify it directly.
                a, b = adj.group(1), adj.group(2)
                graph_ok = not G.has_edge(a, b)
                target = b
                graph_note = "" if graph_ok else f"deny claim FALSE: {a} IS directly connected to {b}"
            elif cut and all(cut.group(i) in G for i in (1, 2, 3)):
                # "all paths A->B must pass through C": true iff removing C disconnects them.
                a, b, c = cut.group(1), cut.group(2), cut.group(3)
                graph_ok = not reachable(G, a, b, frozenset({c}))
                target = b
                graph_note = "" if graph_ok else f"deny claim FALSE: {a}->{b} still reachable without {c}"
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

    # A neutral premise restatement is never a correct answer for ANY polarity --
    # EXCEPT for identify tasks, where `target` IS the answer node: a "premise echo"
    # that names it ("...a route to fuel_depot_1 exists") actually identifies the
    # answer, so feeding it as a WRONG sample manufactures a false leak.
    if ttype not in ("identify_node", "identify_count"):
        wrong = wrong + premise_echo(target)
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
