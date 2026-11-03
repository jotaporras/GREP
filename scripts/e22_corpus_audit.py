"""e22 corpus gate: audit a sharded geo corpus before any training.

Checks the §6 corpus criteria of docs/2026-11-02 e22_geometric_scale_design.md:

- skeleton topology per size (mean degree, dead ends, bridges, cycles per
  region, diameter, clustering); the 100-region means must sit in the §2 band
  (generate_eval_graphs.TARGET_BAND_100);
- populate coverage per size: populated graphs / skeletons >= --min-coverage
  (0.9), because batched populate drops unparseable responses silently;
- hop histogram of the sampled long-hop constraints (longhop_manifest.json) and
  of the committed routes: mode at 6-7, >= 15% at >= 10 hops;
- named-start rate (keep_robot_location) 50 +- 5% of long-hop tasks, plus the
  starting-area, via, avoid and object-goal rates;
- robot_location != route start on EXACTLY the named-start samples;
- md5 of the merged train split (must differ from v2c when --compare is given);
- prompt length of the merged train split (chars; the 130-region prompts are
  the trainer max_length risk of §8).

Layout read per shard (scripts/e22_geo_v1_generate.sbatch):
  <root>/shard<i>/training/nav_n<label>_skeletons/*.json
  <root>/shard<i>/gen/nav_n<label>_gemma_data/populated_graphs/{data_gen_*.json,longhop_manifest.json}
  <root>/shard<i>/gen/nav_n<label>_gemma_data/generated_plans/sample_GGG_TTT.json
  <root>/split/formatted_all_new__train.json   (scripts/e21_merge_shards.py)

Exit code 1 when any gate fails. Usage:
  python scripts/e22_corpus_audit.py --root $ALELAB_DRIVE/GREP-PRISM/data/n_geo_v1 \
      [--compare $ALELAB_DRIVE/GREP-PRISM/data/n_60_oracle_v2c_r2/split/formatted_all_new__train.json]
"""

import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import generate_eval_graphs  # noqa: E402
from prism.data import graph_gen  # noqa: E402
from prism.eval import path_validator  # noqa: E402

ROUTE = re.compile(r"\[answer\((.*)\)\]", re.S)
SCENE = re.compile(r"Scene graph:(\{.*\})\s*$", re.S)


def region_graph(graph: dict) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(r["name"] for r in graph["regions"])
    G.add_edges_from(map(tuple, graph["region_connections"]))
    return G


def skeleton_stats(shards, label):
    """Per-skeleton layout statistics, grouped by region count."""
    by_size = collections.defaultdict(list)
    for shard in shards:
        for f in sorted((shard / "training" / f"nav_n{label}_skeletons").glob("*.json")):
            st = generate_eval_graphs.layout_statistics(region_graph(json.loads(f.read_text())["graph"]))
            by_size[st["n_regions"]].append(st)
    return by_size


def populate_coverage(shards, label):
    """Skeletons vs populated graphs per region count.

    Batched populate skips a graph whose Gemma response is unparseable (no
    ``data_gen_XXX.json`` is written) and the launch script does not retry, so
    the corpus can silently lose a size bucket; job 9987719 kept 37 of 120
    130-region skeletons. Skeleton index i of a shard is ``data_gen_{i:03d}``.
    """
    cov = collections.defaultdict(lambda: [0, 0])  # size -> [skeletons, populated]
    for shard in shards:
        skel = sorted((shard / "training" / f"nav_n{label}_skeletons").glob("*.json"))
        pop = shard / "gen" / f"nav_n{label}_gemma_data" / "populated_graphs"
        for i, f in enumerate(skel):
            size = len(json.loads(f.read_text())["graph"]["regions"])
            cov[size][0] += 1
            cov[size][1] += (pop / f"data_gen_{i:03d}.json").exists()
    return cov


def task_audit(shards, label):
    """Long-hop role rates, hop histograms and the robot_location invariant."""
    hops_manifest = collections.Counter()
    rates = collections.Counter()
    route_hops = collections.Counter()
    loc = collections.Counter()
    for shard in shards:
        run = shard / "gen" / f"nav_n{label}_gemma_data"
        manifest = json.loads((run / "populated_graphs" / "longhop_manifest.json").read_text())
        for key, constraints in manifest.items():
            data = json.loads((run / "populated_graphs" / f"{key}.json").read_text())
            graph, tasks = data["graph"], data["tasks"]
            lh_tasks = tasks[-len(constraints):]
            for c, t in zip(constraints, lh_tasks):
                hops_manifest[c["hops"]] += 1
                rates["longhop"] += 1
                rates["named_start"] += bool(t.get("keep_robot_location"))
                rates["via"] += c.get("via") is not None
                rates["object_goal"] += bool(c.get("object_goal"))
                rates["starting_area"] += bool(graph_gen.START_AREA_PHRASE.search(t["task"]))
                _goal, _wp, avoid, _req, _kind = path_validator.derive_targets(
                    graph, init_node=t["init_node"], answer=t["answer"],
                    criterion=t["acceptance_criterion"], task=t["task"])
                rates["avoid"] += bool(avoid)
            gid = key.split("_")[-1]
            for ti, t in enumerate(tasks):
                sample = run / "generated_plans" / f"sample_{gid}_{ti:03d}.json"
                if not sample.exists():
                    loc["missing_sample"] += 1
                    continue
                msgs = json.loads(sample.read_text())
                route = [n.strip() for n in ROUTE.search(json.loads(msgs[-1]["content"])["plan"]).group(1).split("->")]
                robot = json.loads(SCENE.search(msgs[1]["content"]).group(1))["robot_location"]
                route_hops[len(route) - 1] += 1
                named = bool(t.get("keep_robot_location"))
                moved = robot != route[0]
                loc["samples"] += 1
                loc["named_samples"] += named
                loc["robot_ne_start"] += moved
                loc["violations"] += named != moved
    return hops_manifest, rates, route_hops, loc


def hist_summary(counter):
    total = sum(counter.values())
    top = max(counter.values())
    return {
        "total": total,
        "mode": sorted(h for h, n in counter.items() if n == top),
        "frac_6_7": (counter[6] + counter[7]) / total,
        "frac_ge10": sum(n for h, n in counter.items() if h >= 10) / total,
        "max": max(counter),
        "hist": {h: counter[h] for h in sorted(counter)},
    }


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, type=Path, help="corpus root containing shard*/")
    ap.add_argument("--label", default="geo", help="GRAPH_LABEL of the run (nav_n<label>_*)")
    ap.add_argument("--compare", type=Path, default=None,
                    help="another corpus's formatted_all_new__train.json (e.g. v2c); md5 must differ")
    ap.add_argument("--min-coverage", type=float, default=0.9,
                    help="gate: populated/skeleton ratio required per region count")
    args = ap.parse_args()

    shards = sorted(p for p in args.root.glob("shard*") if p.is_dir())
    if not shards:
        raise SystemExit(f"no shard*/ dirs under {args.root}")
    gates = []

    print("== skeleton topology (mean per size) ==")
    by_size = skeleton_stats(shards, args.label)
    keys = list(generate_eval_graphs.TARGET_BAND_100)
    print(f"{'regions':>8} {'n':>4} " + " ".join(f"{k:>18}" for k in keys))
    for size in sorted(by_size):
        mean = {k: float(np.mean([s[k] for s in by_size[size]])) for k in keys}
        print(f"{size:>8} {len(by_size[size]):>4} " + " ".join(f"{mean[k]:>18.3f}" for k in keys))
        if size == 100:
            for k, (lo, hi) in generate_eval_graphs.TARGET_BAND_100.items():
                gates.append((f"100-region {k} in [{lo}, {hi}]", lo <= mean[k] <= hi, f"{mean[k]:.3f}"))

    print("\n== populate coverage (skeletons -> populated graphs) ==")
    cov = populate_coverage(shards, args.label)
    for size in sorted(cov):
        n_skel, n_pop = cov[size]
        print(f"{size:>8} {n_pop:>4}/{n_skel:<4} = {n_pop / n_skel:.2f}")
        gates.append((f"{size}-region populate coverage >= {args.min_coverage}",
                      n_pop / n_skel >= args.min_coverage, f"{n_pop}/{n_skel}"))

    print("\n== long-hop tasks ==")
    hops_manifest, rates, route_hops, loc = task_audit(shards, args.label)
    hm = hist_summary(hops_manifest)
    rh = hist_summary(route_hops)
    print(f"sampled long-hop hops : {hm}")
    print(f"committed route hops  : {rh}")
    n_lh = rates["longhop"]
    for k in ("named_start", "starting_area", "via", "avoid", "object_goal"):
        print(f"{k:>14}: {rates[k]}/{n_lh} = {rates[k] / n_lh:.3f}")
    print(f"samples: {dict(loc)}")
    gates += [
        ("long-hop hop mode in 6-7", set(hm["mode"]) <= {6, 7}, str(hm["mode"])),
        ("long-hop >= 15% at >= 10 hops", hm["frac_ge10"] >= 0.15, f"{hm['frac_ge10']:.3f}"),
        ("named-start rate 0.50 +- 0.05", abs(rates["named_start"] / n_lh - 0.5) <= 0.05,
         f"{rates['named_start'] / n_lh:.3f}"),
        ("robot_location != start exactly on named-start samples", loc["violations"] == 0,
         f"{loc['violations']} violations"),
    ]

    train = args.root / "split" / "formatted_all_new__train.json"
    print("\n== merged train split ==")
    rows = json.loads(train.read_text())
    lens = np.array([len(next(m["content"] for m in r["conversations"] if m["role"] == "user"))
                     for r in rows])
    digest = md5(train)
    print(f"{train}: {len(rows)} conversations, md5 {digest}")
    print(f"user prompt chars: median {int(np.median(lens))}, p99 {int(np.percentile(lens, 99))}, "
          f"max {int(lens.max())} (~chars/3 tokens, rough)")
    if args.compare is not None:
        other = md5(args.compare)
        print(f"compare {args.compare}: md5 {other}")
        gates.append(("md5 differs from --compare corpus", digest != other, other))

    print("\n== gates ==")
    for name, ok, value in gates:
        print(f"{'PASS' if ok else 'FAIL'}  {name}  ({value})")
    sys.exit(0 if all(ok for _, ok, _ in gates) else 1)


if __name__ == "__main__":
    main()
