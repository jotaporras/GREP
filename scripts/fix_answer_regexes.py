"""Repair leaky/narrow answer-regexes in generated nav data, in place.

The generator emitted bare substring alternations like ``(?i)(yes|can|possible)``.
These (a) FALSE-ACCEPT negatives — ``can`` is a substring of ``cannot`` and
``possible`` of ``impossible`` — and (b) MISS correct paraphrases ("reachable",
"affirmative", "a route"). Both are fixed by emitting word-boundary-anchored,
negation-guarded canonical regexes, chosen per task by the same classifier the
smoke test uses (``task_type``), so battery alignment is guaranteed.

  affirm / already -> AFFIRM_RX   (a "yes/reachable" answer is correct)
  deny             -> DENY_RX     (a "no/unreachable" answer is correct)
  path             -> ordered node-id walk with flexible separators

Usage:  python scripts/fix_answer_regexes.py <dir-or-file> [<dir-or-file> ...]
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from smoke_test_graph_solvable import build_graph, task_type, route_alternatives

# A "yes": needs one positive cue as a whole word. `\b` already blocks the
# cannot/impossible/unreachable substrings; the `(?<!not )` guard kills the only
# remaining leak ("not possible").
AFFIRM_RX = (r"(?i)(?:\byes\b|\baffirmative\b|\bcan reach\b|\bable to reach\b|"
             r"\breachable\b|\ba path\b|\ba route\b|\balready\b|(?<!not )\bpossible\b)")

# A "no": any negation cue as a whole word.
DENY_RX = (r"(?i)(?:\bno\b|\bcannot\b|\bnot\b|\bunreachable\b|"
           r"\bimpossible\b|\bunable\b)")


def path_rx(answer, nodes):
    """Ordered node-id walk, separator-agnostic (matches '->', 'then', ', ')."""
    alts = route_alternatives(answer, nodes)
    if not alts:
        return None
    return "(?i)" + "|".join(".*?".join(seq) for seq in alts)


def fix_task(task, nodes):
    answer = task.get("answer", "") or ""
    crit = task.get("acceptance_criterion", "") or ""
    ttype = task_type(answer, crit, nodes)
    if ttype == "path":
        new = path_rx(answer, nodes)
    elif ttype == "deny":
        new = DENY_RX
    else:  # affirm / already
        new = AFFIRM_RX
    if new and new != answer:
        task["answer"] = new
        return True
    return False


def fix_file(path):
    with open(path) as f:
        doc = json.load(f)
    if "graph" not in doc or "tasks" not in doc:
        return None
    _, nodes, _ = build_graph(doc["graph"])
    changed = sum(fix_task(t, nodes) for t in doc["tasks"])
    if changed:
        with open(path, "w") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
    return changed, len(doc["tasks"])


def main():
    args = sys.argv[1:] or ["."]
    files = []
    for a in args:
        files += sorted(glob.glob(os.path.join(a, "data_gen_*.json"))) if os.path.isdir(a) else [a]
    tot_changed = tot_tasks = tot_files = 0
    for f in files:
        res = fix_file(f)
        if res is None:
            continue
        c, n = res
        tot_changed += c
        tot_tasks += n
        tot_files += 1
        print(f"{os.path.basename(f)}: {c}/{n} answers rewritten")
    print(f"\nDONE: {tot_changed}/{tot_tasks} answers rewritten across {tot_files} files")


if __name__ == "__main__":
    main()
