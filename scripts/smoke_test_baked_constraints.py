"""Deterministic smoke test: do any tasks carry a BAKED (criterion-only) constraint?

The deterministic grader (``prism.eval.path_validator.derive_targets``) reads a
task's required waypoints and avoided regions from the ``acceptance_criterion``
+ ``task`` text combined. The planner, however, only ever sees the ``task`` text.

So if a constraint is named ONLY in the criterion and NOT imposed by the task,
the grader silently demands it while the planner has no way to know — a correct
alternate route is then wrongly failed. This is the "task-8" over-constraint
pattern. This tool flags it, fully deterministically (NetworkX, no LLM).

For every task it compares:
  * the constraints the GRADER enforces  (derive_targets: waypoints, avoid)
  * the constraints the TASK TEXT imposes (cue phrases: "via/through/passing
    through ..." for waypoints, "without/avoid/excluding ..." for avoids)

and checks whether a baked constraint actually changes the answer set:
  * baked waypoint W  -> FAIL if the goal is still reachable WITHOUT W
                         (an alternate valid route exists that omits W; the
                         grader's ``waypoints_ok`` would reject it). WARN if W is
                         forced (every route passes through it) — sloppy but safe.
  * baked avoid A     -> FAIL if the shortest init->goal route PASSES THROUGH A
                         (the natural/optimal answer violates ``avoid_ok``). WARN
                         if the shortest route already misses A.

Verdict per task:
  PASS – no constraint, or every grader constraint is also imposed by the task.
  WARN – a baked constraint exists but is harmless (forced waypoint / unused avoid).
  FAIL – a baked constraint would reject a correct alternate/optimal route.

Usage:
  python scripts/smoke_test_baked_constraints.py <graph.json | dir-of-data_gen_*.json>

Exits nonzero if any task FAILs.
"""
import glob
import json
import os
import re
import sys

import networkx as nx

# Importable from the package root (PYTHONPATH=src), same engine as the grader.
from prism.eval import path_validator as pv

# Constraint CUES as they appear in the natural-language TASK text (no node ids).
_TASK_VIA_CUE = re.compile(
    r"\bvia\b|\bthrough\b|passing\s+through|going\s+through|by\s+way\s+of|\bcrossing\b",
    re.I)
_TASK_AVOID_CUE = re.compile(
    r"\bwithout\b|\bavoid(?:ing|s)?\b|\bexcluding\b|\bnot\s+(?:via|using|through)\b|\bbypass",
    re.I)


def _reachable_without(G, src, dst, blocked):
    if src not in G or dst not in G:
        return False
    H = G.subgraph([n for n in G.nodes if n not in blocked])
    return src in H and dst in H and nx.has_path(H, src, dst)


def check_task(graph_dict, task):
    """Return (verdict, note). Deterministic; never raises."""
    init = task.get("init_node")
    crit = task.get("acceptance_criterion") or ""
    text = task.get("task") or ""
    goal, wp, avoid, _req, kind = pv.derive_targets(
        graph_dict, init_node=init, answer=task.get("answer"),
        criterion=crit, task=text)
    if goal is None or kind != "path" or init is None:
        return "PASS", ""  # nothing route-constrainable to check

    G = pv.build_graph(graph_dict)
    if init not in G or goal not in G:
        return "PASS", ""

    task_has_via = bool(_TASK_VIA_CUE.search(text))
    task_has_avoid = bool(_TASK_AVOID_CUE.search(text))
    notes, verdict = [], "PASS"

    # --- baked waypoints: in the grader's set but no waypoint cue in the task ---
    if wp and not task_has_via:
        for w in wp:
            if _reachable_without(G, init, goal, {w}):
                verdict = "FAIL"
                notes.append(f"baked waypoint '{w}' (task imposes none) — an "
                             f"alternate route omitting it would be rejected")
            else:
                verdict = "WARN" if verdict != "FAIL" else verdict
                notes.append(f"baked waypoint '{w}' but forced (every route uses it)")

    # --- baked avoid: in the grader's set but no avoid cue in the task ---
    if avoid and not task_has_avoid:
        try:
            sp = nx.shortest_path(G, init, goal, weight="distance_m")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            sp = []
        on_sp = [a for a in avoid if a in sp]
        if on_sp:
            verdict = "FAIL"
            notes.append(f"baked avoid {on_sp} lies on the shortest route — the "
                         f"optimal answer would be rejected")
        else:
            verdict = "WARN" if verdict != "FAIL" else verdict
            notes.append(f"baked avoid {sorted(avoid)} (task imposes none) but the "
                         f"shortest route already misses it")

    return verdict, "; ".join(notes)


def audit_file(path):
    with open(path) as f:
        doc = json.load(f)
    tasks = doc.get("tasks", [])
    print(f"\n=== {os.path.basename(path)} ===  ({len(tasks)} tasks)")
    n_fail = n_warn = 0
    for i, task in enumerate(tasks):
        v, note = check_task(doc["graph"], task)
        n_fail += v == "FAIL"
        n_warn += v == "WARN"
        if v != "PASS":
            print(f"[{v:4s}] task {i:2d} | {note} | {task.get('task','')[:70]}")
    if not n_fail and not n_warn:
        print("  all tasks PASS (no baked constraints)")
    return n_fail, n_warn, len(tasks)


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "data/graph.json"
    if os.path.isdir(arg):
        files = sorted(glob.glob(os.path.join(arg, "data_gen_*.json"))) or \
            sorted(glob.glob(os.path.join(arg, "*.json")))
    else:
        files = [arg]
    tot_fail = tot_warn = tot = 0
    for f in files:
        try:
            nf, nw, nt = audit_file(f)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"\n=== {os.path.basename(f)} ===\n  SKIP: not a graph/tasks file ({e})")
            continue
        tot_fail += nf
        tot_warn += nw
        tot += nt
    print(f"\nSUMMARY: {tot} tasks | {tot_fail} FAIL (baked over-constraint) | "
          f"{tot_warn} WARN (baked but harmless) | {tot - tot_fail - tot_warn} PASS")
    sys.exit(1 if tot_fail else 0)


if __name__ == "__main__":
    main()
