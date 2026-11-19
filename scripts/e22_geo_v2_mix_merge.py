"""Merge the e22 geo_v2_mix corpus: 1/3 direct, 1/3 avoid, 1/3 via per graph size.

Sources (all read-only except --out):
- Gemma-written conversations of geo_v1 graphs: <gemma-root>/shard<i>/<gen-subdir>/
  (the n_geo_v2 rebuild). Their route type comes from path_validator.derive_targets on
  each populated task. Dropped: a task that resolves an avoided region (some geo_v1
  tasks state it only in the criterion, so the text never shows it) and a "via" task
  whose waypoint lies on a shortest route (the route is the direct one).
- Templated mix conversations (scripts/e22_template_populate.py retask) of the same
  graphs: <mix-root>/shard<i>/<gen-subdir>/, roles from their longhop_manifest.json.

Graph set and per-size totals: the geo_v2 train split (--geo-v2-split). Only graphs in
that split contribute train rows; each size gets the same number of conversations as
in geo_v2. Per size each route type gets a third; inside the direct and via thirds the
Gemma rows take up to --gemma-cap of the third, templated rows the rest (and Gemma rows
fill any templated shortfall). Shortfalls are reported, never filled from another
route type.

Writes <out>/formatted_all_new__{train,val}.json and <out>/merge_manifest.json.
"""
import argparse
import collections
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from prism.eval import path_validator  # noqa: E402

REGIONS = re.compile(r'"regions": \[(.*?)\], "object_connections"', re.S)
NAME = re.compile(r'\{"name": "([^"]+)"')
TASK = re.compile(r"^task: (.*?)Scene graph:", re.S)
ROLES = ("direct", "avoid", "via")


def _user(row):
    conv = row["conversations"]
    return conv if isinstance(conv, str) else next(m["content"] for m in conv if m["role"] == "user")


def row_key(row):
    text = _user(row)
    names = NAME.findall(REGIONS.search(text).group(1))
    return len(names), tuple(sorted(names)), TASK.search(text).group(1).strip()


def _gemma_role(g: dict, t: dict, dist: dict):
    """Route type of a Gemma task from the grader's own resolver: a resolved avoid
    -> "gemma_avoid" (dropped); a waypoint off every shortest route -> "via"; a
    waypoint on a shortest route -> "gemma_weak_via" (dropped: the text says via but
    the route is the direct one); else "direct"."""
    from prism.data import graph_gen

    goal, way, avoid, _, kind = path_validator.derive_targets(
        g, init_node=t["init_node"], answer=t["answer"],
        criterion=t["acceptance_criterion"], task=t["task"])
    regions = set(dist)
    if kind != "path" or goal not in regions or t["init_node"] not in regions:
        return None
    d0 = dist[t["init_node"]][goal]
    if avoid:
        return dict(route="gemma_avoid", kind=None, hops=None, direct_hops=d0)
    if way:
        if len(way) != 1 or way[0] not in regions:
            return None
        v = way[0]
        h = dist[t["init_node"]][v] + dist[v][goal]
        on = graph_gen._on_some_shortest_path(dist, t["init_node"], goal)
        return dict(route="gemma_weak_via" if v in on else "via", kind=None, hops=h,
                    direct_hops=d0)
    return dict(route="direct", kind=None, hops=d0, direct_hops=d0)


def role_index(shard: Path, gen_subdir: str, gemma: bool) -> dict:
    """(sorted region names, task text) -> role info. Templated: the retask
    manifest. Gemma: every task of the populated graph, by :func:`_gemma_role`."""
    from prism.data import graph_gen

    pg = shard / Path(gen_subdir).parent / "populated_graphs"
    manifest = json.loads((pg / "longhop_manifest.json").read_text()) if not gemma else None
    out = {}
    for path in sorted(pg.glob("data_gen_*.json")):
        data = json.loads(path.read_text())
        g = data["graph"]
        names = tuple(sorted(r["name"] for r in g["regions"]))
        if gemma:
            dist = graph_gen._region_distances(g)
            for t in data["tasks"]:
                info = _gemma_role(g, t, dist)
                if info:
                    out.setdefault((names, t["task"].strip()), info)
            continue
        cons = manifest.get(path.stem, [])
        for c, t in zip(cons, data["tasks"][-len(cons):] if cons else []):
            out.setdefault((names, t["task"].strip()), dict(
                route=c["route"], kind=c.get("avoid_kind"), hops=c["hops"],
                direct_hops=c["direct_hops"]))
    return out


def load(root: Path, gen_subdir: str, shards, gemma: bool, src: str):
    train, val, missing = [], [], 0
    for i in shards:
        sh = root / f"shard{i}"
        idx = role_index(sh, gen_subdir, gemma)
        for split, dest in (("train", train), ("val", val)):
            for r in json.loads((sh / gen_subdir / f"formatted_all_new__{split}.json").read_text()):
                n, names, task = row_key(r)
                info = idx.get((names, task))
                if info is None:
                    missing += 1
                    continue
                dest.append(dict(info, src=src, n=n, names=names, row=r))
    return train, val, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gemma-root", type=Path, required=True, help="n_geo_v2 (read only)")
    ap.add_argument("--gemma-shards", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--mix-root", type=Path, required=True)
    ap.add_argument("--mix-shards", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--geo-v2-split", type=Path, required=True,
                    help="n_geo_v2/split: graph set and per-size totals")
    ap.add_argument("--gen-subdir", default="gen/nav_ngeo_gemma_data/split")
    ap.add_argument("--gemma-cap", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=9000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ref = json.loads((args.geo_v2_split / "formatted_all_new__train.json").read_text())
    graph_set = set()
    size_total = collections.Counter()
    for r in ref:
        n, names, _ = row_key(r)
        graph_set.add(names)
        size_total[n] += 1
    g_train, g_val, g_miss = load(args.gemma_root, args.gen_subdir, args.gemma_shards, True, "gemma")
    t_train, t_val, t_miss = load(args.mix_root, args.gen_subdir, args.mix_shards, False, "templated")
    print(f"pools: gemma train {len(g_train)} val {len(g_val)} (unmatched {g_miss}); "
          f"templated train {len(t_train)} val {len(t_val)} (unmatched {t_miss})")
    pool = collections.defaultdict(list)
    for it in g_train + t_train:
        if it["names"] in graph_set and it["route"] in ROLES:
            pool[(it["n"], it["route"], it["src"])].append(it)
    rng = random.Random(args.seed)
    for k in sorted(pool):
        rng.shuffle(pool[k])

    chosen, report = [], []
    for n in sorted(size_total):
        T = size_total[n]
        quota = {r: T // 3 for r in ROLES}
        for r in ROLES[: T - 3 * (T // 3)]:
            quota[r] += 1
        for r in ROLES:
            q = quota[r]
            gem, tmp = pool[(n, r, "gemma")], pool[(n, r, "templated")]
            n_g = min(len(gem), int(args.gemma_cap * q)) if r != "avoid" else 0
            n_t = min(len(tmp), q - n_g)
            if n_g + n_t < q and r != "avoid":
                n_g = min(len(gem), q - n_t)
            got = gem[:n_g] + tmp[:n_t]
            chosen += got
            short = q - len(got)
            report.append((n, r, q, n_g, n_t, short))
    chosen = chosen[: args.cap]
    rng.shuffle(chosen)
    val = [it for it in g_val + t_val if it["route"] in ROLES]

    args.out.mkdir(parents=True, exist_ok=True)
    tr_txt = json.dumps([it["row"] for it in chosen])
    (args.out / "formatted_all_new__train.json").write_text(tr_txt)
    (args.out / "formatted_all_new__val.json").write_text(json.dumps([it["row"] for it in val]))
    md5 = hashlib.md5(tr_txt.encode()).hexdigest()
    n_large = sum(1 for it in chosen if it["n"] >= 200)
    hop_hist = collections.defaultdict(collections.Counter)
    for it in chosen:
        hop_hist[it["route"] + (f":{it['kind']}" if it["kind"] else "")][it["hops"]] += 1
    mix = collections.Counter((it["n"], it["route"], it["src"], it["kind"]) for it in chosen)
    manifest = {
        "train": len(chosen), "val": len(val), "large_frac": n_large / max(len(chosen), 1),
        "md5_train": md5, "gemma_cap": args.gemma_cap,
        "quota": [dict(zip(("regions", "route", "quota", "gemma", "templated", "short"), r))
                  for r in report],
        "mix": {f"{n}:{r}:{s}:{k}": c for (n, r, s, k), c in sorted(mix.items(), key=str)},
        "hops_by_route": {k: dict(sorted(v.items())) for k, v in sorted(hop_hist.items())},
    }
    (args.out / "merge_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"train {len(chosen)} ({n_large} on >= 200 regions, {n_large / len(chosen):.3f}), "
          f"val {len(val)}, md5 {md5} -> {args.out}")
    print("| regions | route | quota | Gemma | templated | short |")
    print("|---|---|---|---|---|---|")
    for r in report:
        print("| " + " | ".join(map(str, r)) + " |")


if __name__ == "__main__":
    main()
