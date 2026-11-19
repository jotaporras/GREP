"""Merge geo_v1 (Gemma-named) and geo_v2 templated shards into one geo_v2 split.

Each shard holds <shard>/<gen-subdir>/formatted_all_new__{train,val}.json
(scripts/e22_geo_v2_local.sh). The train split is sized so that conversations on
graphs with >= --large-min-regions regions are --large-frac of --total:

- every large-graph conversation is kept up to the large target;
- small graphs: all templated small shards are kept (so templated naming covers
  the whole size range), then geo_v1 graphs are added, whole graphs at a time in
  a seeded random order, until the small target is reached.

The val split keeps every shard's val rows. Writes <root>/split/ and
<root>/split/merge_manifest.json (counts per source and size).

  python scripts/e22_geo_v2_merge.py --root .../n_geo_v2 --gen-subdir gen/nav_ngeo_gemma_data/split \
      --v1-shards 0 1 2 3 --large-min-regions 200 --large-frac 0.40 --total 9000
"""
import argparse
import collections
import json
import random
import re
from pathlib import Path

REGIONS = re.compile(r'"regions": \[(.*?)\], "object_connections"', re.S)
NAME = re.compile(r'\{"name": "([^"]+)"')


def _graph_of(row):
    """(n regions, sorted region names) of the scene graph in a row's user turn
    (the scene graph is JSON text after "Scene graph:")."""
    conv = row["conversations"]
    text = conv if isinstance(conv, str) else next(m["content"] for m in conv if m["role"] == "user")
    m = REGIONS.search(text)
    names = NAME.findall(m.group(1))
    return len(names), tuple(sorted(names))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gen-subdir", default="gen/nav_ngeo_gemma_data/split")
    ap.add_argument("--v1-shards", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--large-min-regions", type=int, default=200)
    ap.add_argument("--large-frac", type=float, default=0.40)
    ap.add_argument("--total", type=int, default=9000,
                    help="target train conversations (reduced if large graphs run out)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shards = sorted((p for p in args.root.glob("shard*") if p.is_dir()),
                    key=lambda p: int(p.name[5:]))
    large, small_tmpl = [], []
    v1_graphs = collections.OrderedDict()
    val = []
    for sh in shards:
        sid = int(sh.name[5:])
        src = "geo_v1" if sid in args.v1_shards else "templated"
        rows = json.loads((sh / args.gen_subdir / "formatted_all_new__train.json").read_text())
        val += json.loads((sh / args.gen_subdir / "formatted_all_new__val.json").read_text())
        for r in rows:
            n, key = _graph_of(r)
            item = (src, n, r)
            if n >= args.large_min_regions:
                large.append(item)
            elif src == "templated":
                small_tmpl.append(item)
            else:
                v1_graphs.setdefault((sid, key), []).append(item)

    n_large = min(len(large), round(args.large_frac * args.total))
    total = round(n_large / args.large_frac) if n_large < round(args.large_frac * args.total) else args.total
    n_small = total - n_large
    rng = random.Random(args.seed)
    rng.shuffle(large)
    chosen = large[:n_large] + small_tmpl
    keys = list(v1_graphs)
    rng.shuffle(keys)
    for k in keys:
        if len(chosen) - n_large >= n_small:
            break
        chosen += v1_graphs[k]
    rng.shuffle(chosen)

    counts = collections.Counter((src, n) for src, n, _ in chosen)
    out = args.root / "split"
    out.mkdir(parents=True, exist_ok=True)
    (out / "formatted_all_new__train.json").write_text(json.dumps([r for _, _, r in chosen]))
    (out / "formatted_all_new__val.json").write_text(json.dumps(val))
    n_big = sum(c for (s, n), c in counts.items() if n >= args.large_min_regions)
    manifest = {
        "train": len(chosen), "val": len(val), "large_frac": n_big / len(chosen),
        "per_source_size": {f"{s}:{n}": c for (s, n), c in sorted(counts.items())},
    }
    (out / "merge_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"train {len(chosen)} conversations ({n_big} on >= {args.large_min_regions} regions, "
          f"{n_big / len(chosen):.3f}), val {len(val)} -> {out}")
    print("| source | regions | conversations |")
    print("|---|---|---|")
    for (s, n), c in sorted(counts.items(), key=lambda x: (x[0][1], x[0][0])):
        print(f"| {s} | {n} | {c} |")


if __name__ == "__main__":
    main()
