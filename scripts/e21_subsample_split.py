"""Build the e21 size-control split: oracle v1 train subsampled to N samples.

Separates data-size from target-purity in the e20 oracle result (92.9 vs the
pathonly cell's 84.5 came with 3.05x more data AND cleaner targets AND fresh
graphs). This writes ``split_579`` next to the source ``split`` with the train
file downsampled (seeded, without replacement) and the val file copied
verbatim, so ``DATASET=oracle579`` trains the identical recipe at the pathonly
corpus size.

Usage (betty login node is fine — pure JSON shuffling):
    python scripts/e21_subsample_split.py \
        --split /vast/.../data/n_60_oracle_v1/gen/nav_n60_gemma_data/split \
        --n 579 --seed 0
"""

import argparse
import json
import random
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path,
                    help="source split dir holding formatted_all_new__{train,val}.json")
    ap.add_argument("--n", type=int, default=579)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = args.split.parent / f"{args.split.name}_{args.n}"
    out.mkdir(parents=True, exist_ok=True)

    train = json.loads((args.split / "formatted_all_new__train.json").read_text())
    if len(train) < args.n:
        raise SystemExit(f"source train has {len(train)} < requested {args.n}")
    sub = random.Random(args.seed).sample(train, args.n)
    (out / "formatted_all_new__train.json").write_text(json.dumps(sub, indent=1))
    shutil.copy(args.split / "formatted_all_new__val.json",
                out / "formatted_all_new__val.json")
    print(f"{out}: train {len(sub)} (of {len(train)}), val copied verbatim")


if __name__ == "__main__":
    main()
