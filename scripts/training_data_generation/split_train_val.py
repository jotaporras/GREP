"""Split generated_plans rollouts into train/val sets by graph ID.

Validates every `sample_GGG_TTT.json` file first. Corrupted files (trailing
garbage past the closing `]`, truncated JSON, etc.) are dropped and listed in
``split_errors.log``. The remaining clean rollouts are grouped by graph id
`GGG`, and ~``val_frac`` of them is held out as validation. Splitting happens
at the graph level (no graph ID appears in both splits); val_frac controls the
target rollout fraction, and graph counts are chosen so the resulting rollout
counts approximate that target.

Outputs:

  <output_dir>/formatted_all_new__train.json
  <output_dir>/formatted_all_new__val.json
  <output_dir>/train_graphs/data_gen_GGG.json   (copied from populated_graphs/)
  <output_dir>/test_graphs/data_gen_GGG.json    (copied from populated_graphs/)
  <output_dir>/split_errors.log
  <output_dir>/split_manifest.json

The two JSON files are HuggingFace-datasets-friendly lists of
``{"conversations": [...]}`` entries, where ``conversations`` is the full
multi-turn rollout (system + user/assistant turns) of one sample.
"""

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from prism.data.utils import strip_icl


SAMPLE_RE = re.compile(r"^sample_(\d+)_(\d+)\.json$")


def validate_samples(plans_dir: Path) -> tuple[dict[str, list[Path]], list[tuple[Path, str]]]:
    """Return (clean_groups_by_gid, errors). A file is clean only if it parses
    fully as a JSON list of message dicts — no trailing garbage tolerated."""
    groups: dict[str, list[Path]] = defaultdict(list)
    errors: list[tuple[Path, str]] = []
    for p in sorted(plans_dir.iterdir()):
        name_match = SAMPLE_RE.match(p.name)
        if not name_match:
            continue
        try:
            obj = json.loads(p.read_text())
        except json.JSONDecodeError as e:
            errors.append((p, f"JSONDecodeError: {e.msg} at line {e.lineno} col {e.colno}"))
            continue
        if not isinstance(obj, list) or not obj:
            errors.append((p, f"not a non-empty list (got {type(obj).__name__})"))
            continue
        if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in obj):
            errors.append((p, "list entries missing role/content fields"))
            continue
        try:
            strip_icl(obj)
        except ValueError as e:
            errors.append((p, f"ICL strip failed: {e}"))
            continue
        groups[name_match.group(1)].append(p)
    return groups, errors


def write_split(out_path: Path, sample_paths: list[Path]) -> int:
    entries = [
        {"conversations": strip_icl(json.loads(sp.read_text()))}
        for sp in sample_paths
    ]
    with out_path.open("w") as f:
        json.dump(entries, f, indent=2)
    return len(entries)


def write_split_2turn(out_path: Path, sample_paths: list[Path]) -> int:
    """Mirror the legacy `formatted_all.json` shape: one entry per rollout
    consisting of the (first user message, last assistant message) pair.
    Drops the system turn and every intermediate observation/action."""
    entries = []
    for sp in sample_paths:
        msgs = strip_icl(json.loads(sp.read_text()))
        first_user = next((m for m in msgs if m["role"] == "user"), None)
        last_assistant = next((m for m in reversed(msgs) if m["role"] == "assistant"), None)
        if first_user is None or last_assistant is None:
            continue
        entries.append({"conversations": [first_user, last_assistant]})
    with out_path.open("w") as f:
        json.dump(entries, f, indent=2)
    return len(entries)


def copy_graphs(graph_ids: list[str], graphs_dir: Path, dst: Path) -> int:
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for gid in graph_ids:
        src = graphs_dir / f"data_gen_{gid}.json"
        if not src.exists():
            print(f"  WARN: missing populated graph for id={gid} ({src})")
            continue
        shutil.copy2(src, dst / src.name)
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plans-dir", type=Path, required=True,
                    help="Directory containing sample_GGG_TTT.json files.")
    ap.add_argument("--graphs-dir", type=Path, required=True,
                    help="Directory containing data_gen_GGG.json populated graphs.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write the split artifacts.")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="Target fraction of clean rollouts to hold out for "
                         "validation. The actual number of val graphs is "
                         "chosen so the val rollout count is closest to this.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups, errors = validate_samples(args.plans_dir)
    log_path = args.output_dir / "split_errors.log"
    with log_path.open("w") as f:
        f.write("# Sample files dropped due to parse/validation errors\n")
        f.write(f"# plans_dir: {args.plans_dir}\n")
        f.write(f"# total errors: {len(errors)}\n\n")
        for p, reason in errors:
            f.write(f"{p.name}\t{reason}\n")
    print(f"Validation: {len(errors)} corrupted file(s) dropped → {log_path}")

    graph_ids = sorted(groups.keys())
    total_rollouts = sum(len(v) for v in groups.values())
    if not graph_ids:
        raise SystemExit("No clean rollouts found; nothing to split.")
    if len(graph_ids) < 2:
        raise SystemExit(f"Only {len(graph_ids)} graph(s) after cleaning; need >= 2 to split.")

    rng = random.Random(args.seed)
    shuffled = graph_ids[:]
    rng.shuffle(shuffled)

    # Choose n_val graphs so the val rollout count is closest to val_frac of total.
    target_val = args.val_frac * total_rollouts
    cum = 0
    best_n, best_diff = 1, float("inf")
    for i, gid in enumerate(shuffled, start=1):
        cum += len(groups[gid])
        if i == len(shuffled):
            break  # keep at least one train graph
        diff = abs(cum - target_val)
        if diff < best_diff:
            best_diff, best_n = diff, i
    n_val = best_n

    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])

    print(f"Clean: {len(graph_ids)} graphs, {total_rollouts} rollouts.")
    print(f"Target val_frac={args.val_frac:.0%} → "
          f"{len(train_ids)} train graphs / {len(val_ids)} val graphs")

    train_samples = [p for gid in train_ids for p in groups[gid]]
    val_samples = [p for gid in val_ids for p in groups[gid]]

    n_train = write_split(
        args.output_dir / "formatted_all_new__train.json", train_samples)
    n_val_rollouts = write_split(
        args.output_dir / "formatted_all_new__val.json", val_samples)
    actual_frac = n_val_rollouts / (n_train + n_val_rollouts)
    print(f"Wrote {n_train} train / {n_val_rollouts} val rollouts "
          f"(val frac = {actual_frac:.1%}).")

    # Legacy-shaped 2-turn variant: (first user, last assistant) per rollout.
    n_train_2t = write_split_2turn(
        args.output_dir / "formatted_all_new_2turn__train.json", train_samples)
    n_val_2t = write_split_2turn(
        args.output_dir / "formatted_all_new_2turn__val.json", val_samples)
    print(f"Wrote 2-turn variant: {n_train_2t} train / {n_val_2t} val.")

    n_tg = copy_graphs(train_ids, args.graphs_dir, args.output_dir / "train_graphs")
    n_eg = copy_graphs(val_ids, args.graphs_dir, args.output_dir / "test_graphs")
    print(f"Copied {n_tg} train graphs, {n_eg} test graphs.")

    manifest = {
        "seed": args.seed,
        "val_frac_target": args.val_frac,
        "val_frac_actual": actual_frac,
        "plans_dir": str(args.plans_dir),
        "graphs_dir": str(args.graphs_dir),
        "n_errors": len(errors),
        "errors_log": log_path.name,
        "train_graph_ids": train_ids,
        "val_graph_ids": val_ids,
        "n_train_conversations": n_train,
        "n_val_conversations": n_val_rollouts,
    }
    with (args.output_dir / "split_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
