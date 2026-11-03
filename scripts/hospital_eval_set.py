"""e22 eval set C: the four hospital graphs as a test_graphs/-shaped dir for cross_eval.

Copies data/hospital_graphs/{hospital_v3, hospital_v4, hospital_v4_length_sweep,
hospital_v4_length_sweep_neutral}.json into --out (the eval loader globs every
*.json there, so the dir holds nothing else), with two changes per task:

- every task gets "hops" (sim_meta hops, else the reference-route length), so
  results can be bucketed by hop count without the sidecar;
- --keep-robot-location marks every task keep_robot_location: true (Arjun's
  prompt keeps the file's robot_location, plaza_1; the task text names the
  start). evaluate.construct_eval_samples_from_dict reads the mark and the
  eval loop then resets the simulator to plaza_1 while grading the route from
  init_node (eval.keep_robot_location=true does the same for every task).

Task text and acceptance criteria are copied verbatim ("never entering X" is
parsed by path_validator._AVOID since e22). Before writing, derive_targets on
every task must reproduce the sim_meta goal, via and avoid with kind "path";
any mismatch raises.

Usage:
  python scripts/hospital_eval_set.py --out data/hospital_graphs/eval_c/test_graphs [--keep-robot-location]
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from prism.eval import path_validator  # noqa: E402

SETS = ("hospital_v3", "hospital_v4", "hospital_v4_length_sweep", "hospital_v4_length_sweep_neutral")


def build_set(src: Path, name: str, keep_robot_location: bool) -> dict:
    data = json.loads((src / f"{name}.json").read_text())
    meta = json.loads((src / f"{name}.sim_meta.json").read_text())["tasks"]
    if len(meta) != len(data["tasks"]):
        raise ValueError(f"{name}: {len(data['tasks'])} tasks vs {len(meta)} sim_meta rows")
    tasks = []
    for i, (t, m) in enumerate(zip(data["tasks"], meta)):
        if t["init_node"] != m["init_node"]:
            raise ValueError(f"{name}[{i}]: init_node {t['init_node']} != sim_meta {m['init_node']}")
        t = dict(t)
        goal, waypoints, avoid, _req, kind = path_validator.derive_targets(
            data["graph"], init_node=t["init_node"], answer=t["answer"],
            criterion=t["acceptance_criterion"], task=t["task"])
        want = (m["goal_node"], [m["via_node"]] if m["via_node"] else [],
                {m["avoid_node"]} if m["avoid_node"] else set(), "path")
        if (goal, waypoints, set(avoid), kind) != want:
            raise ValueError(f"{name}[{i}]: derive_targets {(goal, waypoints, avoid, kind)} != sim_meta {want}")
        route_hops = len(m["reference_route"].split("->")) - 1
        if m["hops"] is not None and m["hops"] != route_hops:
            raise ValueError(f"{name}[{i}]: sim_meta hops {m['hops']} != reference route {route_hops}")
        t["hops"] = route_hops
        if keep_robot_location:
            t["keep_robot_location"] = True
        tasks.append(t)
    return {"graph": data["graph"], "tasks": tasks}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, default=REPO / "data" / "hospital_graphs")
    ap.add_argument("--out", type=Path, required=True, help="output dir (must hold no other *.json)")
    ap.add_argument("--keep-robot-location", action="store_true",
                    help="mark every task keep_robot_location: true (robot stays at the file's plaza_1)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    stray = sorted(p.name for p in args.out.glob("*.json") if p.stem not in SETS)
    if stray:
        raise SystemExit(f"{args.out} holds other *.json the eval loader would pick up: {stray}")
    for name in SETS:
        payload = build_set(args.src, name, args.keep_robot_location)
        (args.out / f"{name}.json").write_text(json.dumps(payload, indent=2))
        hops = [t["hops"] for t in payload["tasks"]]
        print(f"{name}: {len(hops)} tasks, hops {min(hops)}-{max(hops)} -> {args.out / f'{name}.json'}")


if __name__ == "__main__":
    main()
