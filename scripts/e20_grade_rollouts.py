"""Grade existing SPINE rollouts with the deterministic eval scorer (e20).

Walks a ``generated_plans/`` directory of committed ``sample_GGG_TTT.json``
rollouts plus its sibling ``populated_graphs/`` (for the ground-truth task
fields, which never appear in the rollout files), extracts each rollout's final
answered route, and grades it with ``path_validator.validate_structured``
(``full_response=None`` — pure RegEx/NetworkX, no judge) — the SAME verdict the
path-only generator uses to keep/discard teacher outputs.

This is the reasoning-arm side of the e20 teacher-accuracy comparison: run it on
the original (think-mode, multi-turn SPINE) corpus and compare its pass rate to
the ``rollout_stats.json`` a ``--path-only`` generation writes. ``*_failed.json``
quarantines (the planner never reached an answer) are counted as failures with
reason ``spine_no_answer``, so both sides charge unusable outputs against the
teacher.

CPU-only, login-node safe. Example:

  python scripts/e20_grade_rollouts.py \\
      --plans-dir  $DATA/gen/nav_n60_gemma_data/generated_plans \\
      --graphs-dir $DATA/gen/nav_n60_gemma_data/populated_graphs \\
      --out        $DATA/gen/nav_n60_gemma_data/generated_plans/reasoning_grade_stats.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prism.data import compact_prompt  # noqa: E402  (torch-free)
from prism.eval import path_validator  # noqa: E402  (torch-free)

_SAMPLE_RE = re.compile(r"sample_(\d+)_(\d+)\.json$")
_FAILED_RE = re.compile(r"sample_(\d+)_(\d+)_failed\.json$")


def _final_route(rollout_path: Path):
    """(route|None, reason) from a committed rollout's last assistant answer."""
    try:
        msgs = json.load(open(rollout_path))
    except Exception:
        return None, "unreadable"
    for m in reversed(msgs):
        if m.get("role") != "assistant":
            continue
        try:
            plan = compact_prompt._unwrap_plan(compact_prompt._as_text(
                json.loads(compact_prompt._strip_code_fence(m["content"]),
                           strict=False)["plan"]))
        except Exception:
            continue  # tool-turn / unparseable — keep walking back
        route = compact_prompt.extract_route(plan)
        return (route, "") if route else (None, "no_route")
    return None, "no_answer_turn"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plans-dir", required=True,
                    help="generated_plans/ dir with sample_GGG_TTT.json rollouts")
    ap.add_argument("--graphs-dir", required=True,
                    help="populated_graphs/ dir with data_gen_GGG.json (GT fields)")
    ap.add_argument("--out", default=None,
                    help="output JSON (default: <plans-dir>/reasoning_grade_stats.json)")
    args = ap.parse_args()

    plans = Path(args.plans_dir)
    graphs_dir = Path(args.graphs_dir)
    out_path = Path(args.out) if args.out else plans / "reasoning_grade_stats.json"

    data_gen = {}
    for f in sorted(graphs_dir.glob("*data_gen*json")):
        m = re.search(r"data_gen_(\d+)", f.name)
        if m:
            try:
                data_gen[m.group(1)] = json.load(open(f))
            except Exception as ex:
                print(f"skipping {f.name}: {ex}")

    passed, failed, reasons = {}, {}, {}
    graded = []
    seen = set()
    for f in sorted(plans.glob("sample_*json")):
        cm = _SAMPLE_RE.search(f.name)
        fm = _FAILED_RE.search(f.name)
        if not cm and not fm:
            continue
        gid, tid = (cm or fm).groups()
        if fm and (plans / f"sample_{gid}_{tid}.json").exists():
            continue  # a later retry committed; count the committed one only
        key = (gid, tid)
        if key in seen:
            continue
        seen.add(key)

        def _fail(reason):
            failed[gid] = failed.get(gid, 0) + 1
            reasons[reason] = reasons.get(reason, 0) + 1
            graded.append({"graph": gid, "task": tid, "ok": False, "reason": reason})

        if fm:
            _fail("spine_no_answer")
            continue
        dg = data_gen.get(gid)
        if dg is None:
            _fail("no_data_gen_file")
            continue
        try:
            entry = dg["tasks"][int(tid)]
        except (KeyError, IndexError, ValueError):
            _fail("no_task_entry")
            continue
        route, why = _final_route(f)
        if route is None:
            _fail(why)
            continue
        verdict = path_validator.validate_structured(
            route, dg["graph"],
            init_node=entry.get("init_node"),
            answer=entry.get("answer"),
            criterion=entry.get("acceptance_criterion"),
            task=entry.get("task"),
            full_response=None,
        )
        if verdict and verdict.get("structured_correct"):
            passed[gid] = passed.get(gid, 0) + 1
            graded.append({"graph": gid, "task": tid, "ok": True, "route": route})
        else:
            _fail("wrong_route" if verdict else "no_goal_resolved")

    n_pass, n_fail = sum(passed.values()), sum(failed.values())
    total = n_pass + n_fail
    stats = {
        "rollout_mode": "spine_reasoning_regrade",
        "plans_dir": str(plans),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_total": total,
        "pass_rate": (n_pass / total) if total else None,
        "fail_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "per_graph": {
            gid: {"pass": passed.get(gid, 0), "fail": failed.get(gid, 0)}
            for gid in sorted(set(passed) | set(failed))
        },
        "samples": graded,
    }
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=1)
    rate = f"{stats['pass_rate']:.3f}" if total else "n/a"
    print(f"[regrade] {n_pass}/{total} passed (rate {rate}); "
          f"reasons {stats['fail_reasons']}\n-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
