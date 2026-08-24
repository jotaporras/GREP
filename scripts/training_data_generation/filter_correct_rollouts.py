#!/usr/bin/env python3
"""Rejection-sample SPINE rollouts: quarantine incorrect episodes before split.

Grades every ``generated_plans/sample_GGG_TTT.json`` with the same replay stack
as ``scripts/format_generated_plans.py`` (real SPINE parser + PlanningSim +
path_validator) and renames incorrect ones to ``sample_GGG_TTT_rejected.json``
so ``split_train_val.py`` (glob ``^sample_\\d+_\\d+\\.json$``) never sees them.
Writes ``generated_plans/filter_stats.json`` with per-graph keep/reject counts.

Replay runs with PRISM_GOTO_RATIFY=0: a multi-turn episode's FINAL plan starts
from wherever the agent stood mid-episode, so ratifying the goto prefix from
init would misgrade it. The verdict rests on the structural validation of the
full answer route, which is position-independent.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

os.environ["PRISM_GOTO_RATIFY"] = "0"  # must precede prism imports (see above)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (os.path.join(_REPO, "src"),
           os.path.join(_REPO, "scripts"),
           os.path.join(os.path.dirname(_REPO), "SPINE", "src")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from prism.eval import evaluate                    # noqa: E402
import format_generated_plans as fgp               # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs-dir", required=True,
                    help="Dir with populated data_gen_GGG.json (graph + tasks).")
    ap.add_argument("--plans-dir", required=True,
                    help="Dir with sample_GGG_TTT.json rollouts to filter.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Grade and report only; do not rename anything.")
    args = ap.parse_args()

    stats = {"graphs": {}, "kept": 0, "rejected": 0, "missing": 0, "corrupt": 0}
    for gf in sorted(glob.glob(os.path.join(args.graphs_dir, "data_gen_*.json"))):
        stem = os.path.splitext(os.path.basename(gf))[0]
        ggg = stem.rsplit("_", 1)[-1]
        with open(gf) as f:
            data = json.load(f)
        eval_samples = evaluate.construct_eval_samples_from_dict(
            data["graph"], data["tasks"], stem)

        kept = rejected = 0
        for i, es in enumerate(eval_samples):
            sp = os.path.join(args.plans_dir, f"sample_{ggg}_{i:03d}.json")
            if not os.path.exists(sp):
                stats["missing"] += 1
                continue
            try:
                with open(sp) as f:
                    messages = json.load(f)
            except (json.JSONDecodeError, ValueError) as ex:
                # A handful of rollouts land with trailing garbage (two JSON
                # docs concatenated). split_train_val already skips those; the
                # filter must not abort the whole run over them.
                stats["corrupt"] += 1
                print(f"  corrupt sample_{ggg}_{i:03d}: {ex}")
                if not args.dry_run:
                    os.replace(sp, sp[:-len(".json")] + "_corrupt.json")
                continue
            content = fgp._final_assistant_content(messages)
            _, samples = fgp.grade_graph([es], [content], use_icl=False)
            if samples and samples[0].get("correct"):
                kept += 1
            else:
                rejected += 1
                if not args.dry_run:
                    os.replace(sp, sp[:-len(".json")] + "_rejected.json")
                print(f"  reject sample_{ggg}_{i:03d}")
        stats["graphs"][stem] = {"kept": kept, "rejected": rejected}
        stats["kept"] += kept
        stats["rejected"] += rejected
        print(f"{stem}: kept {kept}, rejected {rejected}")

    total = stats["kept"] + stats["rejected"]
    print(f"TOTAL kept {stats['kept']}/{total} "
          f"({stats['missing']} missing, {stats['corrupt']} corrupt)"
          f"{' [dry-run]' if args.dry_run else ''}")
    if not args.dry_run:
        out = os.path.join(args.plans_dir, "filter_stats.json")
        with open(out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"stats -> {out}")


if __name__ == "__main__":
    main()
