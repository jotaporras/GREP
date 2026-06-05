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

# A "yes": needs one *unambiguously affirmative* cue as a whole word. `\b`
# already blocks the cannot/impossible/unreachable substrings; the `(?<!not )`
# guards kill the two-word negation leaks ("not reachable", "not able to reach",
# "not possible"). Bare narration nouns `a path`/`a route` are intentionally
# dropped — they match premise echoes ("...whether there is a path to X")
# without stating a verdict; the affirm answers that used them still match via
# `yes`/`affirmative`.
AFFIRM_RX = (r"(?i)(?:\byes\b|\baffirmative\b|\bcan reach\b|(?<!not )\bable to reach\b|"
             r"(?<!not )\breachable\b|(?<!not )\bpossible\b|\balready\b)")

# A "no": a negation bound to a navigation verb/noun. Bare `no`/`not` are
# intentionally dropped — they match discourse incidentals ("No problem",
# "not in one move") and would false-accept an affirmative answer as a denial.
DENY_RX = (r"(?i)(?:\bcannot\b|\bcan't\b|\bunreachable\b|\bimpossible\b|\bunable\b|"
           r"\bnot\s+(?:reachable|possible|able)\b|\bno\s+(?:path|route|way)\b)")


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
