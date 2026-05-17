"""Strip SPINE ICL example turns out of generated_plans/sample_GGG_TTT.json.

Each raw rollout begins with `system` + ~35 messages of canned ICL example
turns prepended by the SPINE prompt builder, followed by the real task at
`msgs[last_task_idx]` and the rollout that answers it. The downstream split
script's "first user / last assistant" heuristic pairs an ICL turn with the
real final answer, producing 393 training examples whose user prompt is
byte-identical and unrelated to the assistant label.

This script keeps `msgs[0]` (system) and `msgs[last_task_idx:]` (the real
task plus every assistant/user turn it generated, including planner retries
and environment updates), writing the pruned rollouts to a sibling directory.

Rule for `last_task_idx`: scan from the end for a `role=user` message whose
content (after lstrip) begins with `task:` (case-insensitive). This is the
real task because every ICL example also begins with `task:` but appears
earlier in the conversation; the rollout's actual task is always the last.

Intermediate assistant/feedback turns from the real task are preserved
(multi-turn rollouts are kept intact, not collapsed to a single final
answer), so planner retries and environment-update interactions remain
trainable.
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path


def find_last_task_user_idx(msgs: list[dict]) -> int | None:
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if m.get("role") == "user" and m.get("content", "").lstrip().lower().startswith("task:"):
            return i
    return None


def prune(msgs: list[dict]) -> tuple[list[dict] | None, str]:
    """Return (pruned_msgs, reason). pruned_msgs is None on failure."""
    if not isinstance(msgs, list) or not msgs:
        return None, "not_a_nonempty_list"
    idx = find_last_task_user_idx(msgs)
    if idx is None:
        return None, "no_task_user_found"
    # Sanity: at least one assistant turn must follow the real task
    if not any(m.get("role") == "assistant" for m in msgs[idx + 1:]):
        return None, "no_assistant_after_task"
    head = [msgs[0]] if msgs and msgs[0].get("role") == "system" else []
    return head + msgs[idx:], "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plans-dir", type=Path,
                    default=Path("data/gen/e4_training_data_v4/generated_plans"))
    ap.add_argument("--out-dir", type=Path,
                    default=Path("data/gen/e4_training_data_v4/generated_plans_pruned"))
    args = ap.parse_args()

    if not args.plans_dir.is_dir():
        sys.exit(f"plans-dir does not exist: {args.plans_dir}")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    counts: Counter[str] = Counter()
    in_len_hist: Counter[int] = Counter()
    out_len_hist: Counter[int] = Counter()
    out_asst_hist: Counter[int] = Counter()

    samples = sorted(p for p in args.plans_dir.iterdir()
                     if p.name.startswith("sample_") and p.suffix == ".json")
    for sp in samples:
        try:
            msgs = json.loads(sp.read_text())
        except json.JSONDecodeError:
            counts["parse_err"] += 1
            continue

        in_len_hist[len(msgs)] += 1
        pruned, reason = prune(msgs)
        if pruned is None:
            counts[f"drop_{reason}"] += 1
            continue

        out_path = args.out_dir / sp.name
        out_path.write_text(json.dumps(pruned, indent=2))
        counts["ok"] += 1
        out_len_hist[len(pruned)] += 1
        out_asst_hist[sum(1 for m in pruned if m.get("role") == "assistant")] += 1

    # Carry over non-sample files (formatted.json etc.) untouched? No — the aggregate
    # formatted.json is built from the unpruned sources and would be stale. Leave it
    # in plans-dir; downstream split should consume out-dir.

    print(f"Input dir : {args.plans_dir}")
    print(f"Output dir: {args.out_dir}")
    print(f"\nSample files scanned: {len(samples)}")
    print(f"Outcome:")
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    print(f"\nInput msg-length histogram (top 5):")
    for n, c in in_len_hist.most_common(5):
        print(f"  {n} msgs: {c} files")
    print(f"\nOutput msg-length histogram (top 10):")
    for n, c in sorted(out_len_hist.items()):
        print(f"  {n} msgs: {c} files")
    print(f"\nOutput assistant-turn-count histogram:")
    for n, c in sorted(out_asst_hist.items()):
        print(f"  {n} asst turns: {c} files")


if __name__ == "__main__":
    main()
