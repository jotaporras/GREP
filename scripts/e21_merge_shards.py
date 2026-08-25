"""Merge the per-shard train/val splits of a sharded corpus generation run.

Counterpart of scripts/e21_n60_oracle_v2b_generate.sbatch: each array task
writes <root>/shard<i>/gen/nav_n60_gemma_data/split/formatted_all_new__*.json;
this concatenates them (shard order, then file order within a shard) into
<root>/split/formatted_all_new__{train,val}.json — the layout the training
sbatch DATASET cases expect.

Usage:
  python scripts/e21_merge_shards.py --root .../data/n_60_oracle_v2b
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="corpus root containing shard*/")
    ap.add_argument("--gen-subdir", default="gen/nav_n60_gemma_data/split",
                    help="split dir path inside each shard")
    args = ap.parse_args()

    root = Path(args.root)
    shards = sorted(p for p in root.glob("shard*") if p.is_dir())
    if not shards:
        raise SystemExit(f"no shard*/ dirs under {root}")

    out_dir = root / "split"
    out_dir.mkdir(parents=True, exist_ok=True)

    for part in ("train", "val"):
        merged = []
        for shard in shards:
            f = shard / args.gen_subdir / f"formatted_all_new__{part}.json"
            if not f.exists():
                raise SystemExit(f"missing {f} — shard incomplete, not merging")
            rows = json.loads(f.read_text())
            print(f"{shard.name}/{part}: {len(rows)}")
            merged.extend(rows)
        out = out_dir / f"formatted_all_new__{part}.json"
        out.write_text(json.dumps(merged))
        print(f"-> {out}: {len(merged)}")


if __name__ == "__main__":
    main()
