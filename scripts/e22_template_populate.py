"""e22 templated (LLM-free) populate: harvest, populate, compare.

The Gemma populate step does not fit the context window above ~400 regions, so
geo_v2 training shards (all sizes) and the geo_scale evaluation set are named and
tasked by src/prism/data/template_populate.py from a vocabulary harvested from
the Gemma-written geo_v1 corpus. Nothing here runs unless called explicitly; the
LLM populate path is unchanged.

  # 1. vocabulary from the local geo_v1 shards (skeletons/ + populated_graphs/)
  python scripts/e22_template_populate.py harvest \
      --shards outputs/e22_geometric_scale/notebook_data/train/shard* \
      --out scripts/e22_template_vocab.json

  # 2. populate a skeleton dir (writes data_gen_NNN.json, graph_gen_NNN.json and
  #    longhop_manifest.json, the layout generate_data_spine.py --skip-populate reads)
  python scripts/e22_template_populate.py populate --vocab scripts/e22_template_vocab.json \
      --skeleton-dir <skel> --out-dir <run>/populated_graphs --n-tasks 16 \
      --hop-weights "1:0.04,..." --named-start-frac 0.5 --start-area-frac 0.4 \
      --via-frac 0.2 --object-goal-frac 0.5 --seed 7

  # 3. naming/task statistics of templated vs Gemma graphs
  python scripts/e22_template_populate.py compare --vocab scripts/e22_template_vocab.json \
      --a outputs/e22_geometric_scale/notebook_data/train/shard*/populated_graphs \
      --b <run>/populated_graphs
"""
import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from prism.data import graph_gen, template_populate as tp  # noqa: E402


def cmd_harvest(args):
    vocab = tp.harvest_vocab(tp.load_geo_pairs(args.shards))
    args.out.write_text(json.dumps(vocab, indent=1, sort_keys=True))
    st = vocab["stats"]
    print(f"harvested {len(vocab['prefixes'])} zone prefixes, {len(vocab['kinds_all'])} kinds, "
          f"{len(vocab['objects_all'])} object stems, borrow_rate {vocab['borrow_rate']:.3f}")
    print(f"tasks templated {st['tasks_templated']}/{st['tasks_seen']}; templates per signature: "
          + ", ".join(f"{s}={len(t)}" for s, t in sorted(vocab["templates"].items())))
    print(f"-> {args.out}")


def _sample_constraints(skel_graph, args, rng, idx=0):
    if args.hop_strata:
        cons = tp.sample_stratified_constraints(
            skel_graph, args.n_tasks, rng, tp.parse_strata(args.hop_strata),
            args.object_goal_frac, offset=idx)
        max_via = max(hi for _, hi in tp.parse_strata(args.hop_strata))
    else:
        weights = graph_gen.parse_hop_weights(args.hop_weights)
        cons = graph_gen.sample_longhop_constraints(
            skel_graph, args.n_tasks, rng, hop_weights=weights,
            object_goal_frac=args.object_goal_frac)
        max_via = max(weights)
    return graph_gen.assign_longhop_roles(
        skel_graph, cons, rng, named_start_frac=args.named_start_frac,
        via_frac=args.via_frac, max_via_hops=max_via)


def cmd_populate(args):
    vocab = json.loads(args.vocab.read_text())
    skels = sorted(args.skeleton_dir.glob("*graph*json"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    man_path = args.out_dir / "longhop_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    n_ok = n_fail = 0
    for idx, path in enumerate(skels):
        key = f"data_gen_{idx:03d}"
        out = args.out_dir / f"{key}.json"
        if out.exists() and key in manifest:
            n_ok += 1
            continue
        skel = json.loads(path.read_text())
        seed = skel.get("_metadata", {}).get("seed", idx)
        populated = None
        for attempt in range(5):
            rng = np.random.default_rng([args.seed, int(seed), attempt])
            try:
                cons = _sample_constraints(skel["graph"], args, rng, idx)
                populated, final = tp.populate_skeleton(
                    skel, cons, vocab, rng, start_area_frac=args.start_area_frac)
                graph_gen.assert_graph_refs_resolve(populated["graph"])
                graph_gen.validate_task_refs(populated["tasks"], populated["graph"])
                graph_gen.validate_longhop_tasks(populated["tasks"], final, populated["graph"])
                break
            except Exception as ex:  # resample names/roles; report if it never passes
                print(f"{path.name} attempt {attempt}: {ex}")
                populated = None
        if populated is None:
            n_fail += 1
            continue
        out.write_text(json.dumps(populated, indent=2))
        (args.out_dir / f"graph_gen_{idx:03d}.json").write_text(
            json.dumps(populated["graph"], indent=2))
        manifest[key] = cons
        n_ok += 1
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"populated {n_ok}/{len(skels)} skeletons ({n_fail} failed) -> {args.out_dir}")
    if n_fail:
        sys.exit(1)


# --------------------------------------------------------------------------- e22 mix

def _validate(populated, final):
    graph_gen.assert_graph_refs_resolve(populated["graph"])
    graph_gen.validate_task_refs(populated["tasks"], populated["graph"])
    graph_gen.validate_longhop_tasks(populated["tasks"], final, populated["graph"])


def cmd_retask(args):
    """New templated mix tasks (direct / via / avoid) on already-named graphs.

    Reads <populated-dir>/data_gen_NNN.json (Gemma or templated names, kept as
    they are) and writes the same stems to --out-dir with ONLY the new tasks, so
    generate_data_spine.py sees the same graph indices (val ids unchanged)."""
    vocab = json.loads(args.vocab.read_text())
    skels = sorted(args.skeleton_dir.glob("*graph*json")) if args.skeleton_dir else []
    weights = graph_gen.parse_hop_weights(args.hop_weights)
    role_w = tp.parse_role_mix(args.role_mix)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    man_path = args.out_dir / "longhop_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    n_ok = n_fail = 0
    for path in sorted(args.populated_dir.glob("data_gen_*.json")):
        key = path.stem
        out = args.out_dir / path.name
        if out.exists() and key in manifest:
            n_ok += 1
            continue
        idx = int(key.split("_")[-1])
        pop = json.loads(path.read_text())
        skel = json.loads(skels[idx].read_text()) if idx < len(skels) else None
        if skel and [r["coords"] for r in skel["graph"]["regions"]] != \
                [r["coords"] for r in pop["graph"]["regions"]]:
            skel = None  # misaligned pair: no zone-relative mentions
        prefixes = tp.graph_prefixes(skel, pop["graph"])
        new = None
        for attempt in range(5):
            rng = np.random.default_rng([args.seed, idx, attempt, 22])
            try:
                cons = graph_gen.sample_longhop_constraints(
                    pop["graph"], args.n_tasks, rng, hop_weights=weights,
                    object_goal_frac=args.object_goal_frac)
                cons = tp.assign_mix_roles(
                    pop["graph"], cons, rng, args.named_start_frac, role_w,
                    args.detour_frac, args.max_hops)
                new = tp.retask_graph(pop, prefixes, cons, vocab, rng, args.start_area_frac)
                _validate(new, cons)
                break
            except Exception as ex:
                print(f"{path.name} attempt {attempt}: {ex}")
                new = None
        if new is None:
            n_fail += 1
            continue
        out.write_text(json.dumps(new, indent=2))
        (args.out_dir / f"graph_gen_{idx:03d}.json").write_text(json.dumps(new["graph"], indent=2))
        manifest[key] = cons
        n_ok += 1
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"retasked {n_ok} graphs ({n_fail} failed) -> {args.out_dir}")
    if n_fail:
        sys.exit(1)


def cmd_constraints_eval(args):
    """geo_scale_constraints: per graph, --pairs-per-graph (init, goal) pairs, each
    with a direct, a detour-avoid and (where possible) a via task. Writes
    <out-dir>/test_graphs/data_gen_NNN.json and <out-dir>/constraints_manifest.json."""
    vocab = json.loads(args.vocab.read_text())
    strata = tp.parse_strata(args.hop_strata)
    skels = sorted(args.skeleton_dir.glob("*graph*json"))
    tg = args.out_dir / "test_graphs"
    tg.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for idx, path in enumerate(skels):
        skel = json.loads(path.read_text())
        seed = skel.get("_metadata", {}).get("seed", idx)
        done = None
        for attempt in range(5):
            rng = np.random.default_rng([args.seed, int(seed), attempt, 23])
            try:
                g = skel["graph"]
                cons = tp.sample_paired_constraints(g, args.pairs_per_graph, rng, strata,
                                                    offset=idx)
                # Start roles per pair from a counter over the whole size set (3 pairs
                # per graph is too few to round per graph): global pair index q is a
                # named start at rate named_start_frac (when its init is not the
                # robot's location); unnamed pairs say "starting area" at
                # start_area_frac.
                named, area = set(), set()
                fn, fa = args.named_start_frac, args.start_area_frac
                for c in cons:
                    q = idx * args.pairs_per_graph + c["pair"]
                    if c["pair"] in named | area or c["route"] != "direct":
                        continue
                    if int((q + 1) * fn) > int(q * fn) and c["init"] != g["robot_location"]:
                        named.add(c["pair"])
                    elif int((q + 1) * fa) > int(q * fa):
                        area.add(c["pair"])
                rename, descriptions, prefixes = tp.name_graph(skel, vocab, rng)
                graph = graph_gen.apply_rename(skel, rename, descriptions)
                final = []
                for c in cons:
                    f = dict(c, named_start=c["pair"] in named, init=rename[c["init"]],
                             goal=rename[c["goal"]])
                    for k in ("via", "avoid"):
                        if c.get(k):
                            f[k] = rename[c[k]]
                    final.append(f)
                tasks = [tp.render_task(f, graph, vocab, rng, start_area=f["pair"] in area,
                                        prefixes=prefixes) for f in final]
                populated = {"graph": graph, "tasks": tasks,
                             "description": "templated (e22 geo_scale_constraints)"}
                _validate(populated, final)
                done = (populated, final)
                break
            except Exception as ex:
                print(f"{path.name} attempt {attempt}: {ex}")
        if done is None:
            sys.exit(f"{path.name}: no valid constraint set")
        (tg / f"data_gen_{idx:03d}.json").write_text(json.dumps(done[0], indent=2))
        manifest[f"data_gen_{idx:03d}"] = done[1]
    (args.out_dir / "constraints_manifest.json").write_text(json.dumps(manifest, indent=2))
    n = sum(len(v) for v in manifest.values())
    print(f"{len(manifest)} graphs, {n} tasks -> {tg}")


# --------------------------------------------------------------------------- compare

FAMILY = [
    ("from_X_route_to", r"^from .+?, (give the |)route"),
    ("from_X_give_route", r"^from .+?, give the (route|path)"),
    ("from_X_reach", r"^from .+?, reach"),
    ("route_from_X", r"^route from"),
    ("route_the_robot", r"^route the robot"),
    ("give_the_route", r"^give the (route|path)"),
    ("reach", r"^reach"),
    ("navigate", r"^navigate"),
]
ENDING = [
    ("report_edges_route", r"report the connecting edges and the route"),
    ("list_the_path", r"list the path"),
    ("give_edges_route", r"give the connecting edges and the route"),
    ("show_the_path", r"show the (full )?path"),
]


def _family(text, table):
    for name, pat in table:
        if re.search(pat, text, re.IGNORECASE):
            return name
    return "other"


def corpus_stats(dirs, vocab):
    st = collections.Counter()
    tok = collections.Counter()
    ostems = collections.Counter()
    fam = collections.Counter()
    end = collections.Counter()
    n_obj = n_reg = n_graph = 0
    for d in dirs:
        d = Path(d)
        manifest = json.loads((d / "longhop_manifest.json").read_text())
        for key, cons in manifest.items():
            data = json.loads((d / f"{key}.json").read_text())
            g = data["graph"]
            n_graph += 1
            n_reg += len(g["regions"])
            n_obj += len(g["objects"])
            for r in g["regions"]:
                tok.update(tp.split_name(r["name"])[0].split("_"))
            for o in g["objects"]:
                ostems[tp.split_name(o["name"])[0]] += 1
            for c, t in zip(cons, data["tasks"][-len(cons):]):
                st["tasks"] += 1
                st["named_start"] += bool(t.get("keep_robot_location"))
                st["starting_area"] += bool(tp.START_AREA.search(t["task"]))
                st["via"] += bool(c.get("via"))
                st["object_goal"] += bool(c.get("object_goal"))
                fam[_family(t["task"], FAMILY)] += 1
                end[_family(t["task"], ENDING)] += 1
                st["task_words"] += len(t["task"].split())
    return dict(st=st, tok=tok, ostems=ostems, fam=fam, end=end,
                n_graph=n_graph, n_reg=n_reg, n_obj=n_obj)


def _overlap(a: collections.Counter, b: collections.Counter) -> float:
    """Share of b's mass whose key also occurs in a (token-weighted)."""
    tot = sum(b.values())
    return sum(v for k, v in b.items() if k in a) / tot if tot else 0.0


def _tv(a: collections.Counter, b: collections.Counter) -> float:
    ta, tb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / ta - b[k] / tb) for k in keys)


def cmd_compare(args):
    vocab = json.loads(args.vocab.read_text()) if args.vocab else None
    A, B = corpus_stats(args.a, vocab), corpus_stats(args.b, vocab)
    rows = []
    for name, S in (("A", A), ("B", B)):
        st = S["st"]
        n = st["tasks"]
        rows.append((name, S["n_graph"], n, S["n_obj"] / S["n_reg"], st["named_start"] / n,
                     st["starting_area"] / n, st["via"] / n, st["object_goal"] / n,
                     st["task_words"] / n))
    print("| set | graphs | tasks | objects/region | named start | starting area | via | "
          "object goal | words/task |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} | "
              f"{r[6]:.3f} | {r[7]:.3f} | {r[8]:.1f} |")
    print()
    print("| measure | value |")
    print("|---|---|")
    print(f"| B region-name tokens also in A (token-weighted) | {_overlap(A['tok'], B['tok']):.3f} |")
    print(f"| B object stems also in A (count-weighted) | {_overlap(A['ostems'], B['ostems']):.3f} |")
    print(f"| region-name token distribution, total variation A vs B | {_tv(A['tok'], B['tok']):.3f} |")
    print(f"| object-stem distribution, total variation A vs B | {_tv(A['ostems'], B['ostems']):.3f} |")
    print(f"| task opening family, total variation A vs B | {_tv(A['fam'], B['fam']):.3f} |")
    print(f"| task ending family, total variation A vs B | {_tv(A['end'], B['end']):.3f} |")
    print()
    print("| opening family | A | B |")
    print("|---|---|---|")
    ta, tb = sum(A["fam"].values()), sum(B["fam"].values())
    for k in sorted(set(A["fam"]) | set(B["fam"]), key=lambda k: -A["fam"][k]):
        print(f"| {k} | {A['fam'][k] / ta:.3f} | {B['fam'][k] / tb:.3f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--shards", nargs="+", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("populate")
    p.add_argument("--vocab", type=Path, required=True)
    p.add_argument("--skeleton-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-tasks", type=int, default=16)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hop-weights", help="graph_gen.parse_hop_weights spec (training)")
    g.add_argument("--hop-strata", help='eval: round robin over strata, e.g. "1-6,7-15,16-30,31-999"')
    p.add_argument("--named-start-frac", type=float, default=0.5)
    p.add_argument("--start-area-frac", type=float, default=0.4)
    p.add_argument("--via-frac", type=float, default=0.2)
    p.add_argument("--object-goal-frac", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=7)
    r = sub.add_parser("retask", help="e22 mix: new direct/via/avoid tasks on named graphs")
    r.add_argument("--vocab", type=Path, required=True)
    r.add_argument("--populated-dir", type=Path, required=True)
    r.add_argument("--skeleton-dir", type=Path, default=None,
                   help="aligned skeletons (zone prefixes for zone-relative mentions)")
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--n-tasks", type=int, default=12)
    r.add_argument("--hop-weights", required=True)
    r.add_argument("--role-mix", default="direct:1,avoid:1,via:1")
    r.add_argument("--detour-frac", type=float, default=0.8)
    r.add_argument("--max-hops", type=int, default=40, help="cap on the constrained length")
    r.add_argument("--named-start-frac", type=float, default=0.5)
    r.add_argument("--start-area-frac", type=float, default=0.46)
    r.add_argument("--object-goal-frac", type=float, default=0.5)
    r.add_argument("--seed", type=int, default=7)
    e = sub.add_parser("constraints-eval", help="e22 geo_scale_constraints eval set")
    e.add_argument("--vocab", type=Path, required=True)
    e.add_argument("--skeleton-dir", type=Path, required=True)
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--pairs-per-graph", type=int, default=3)
    e.add_argument("--hop-strata", default="1-6,7-15,16-30,31-999")
    e.add_argument("--named-start-frac", type=float, default=0.5)
    e.add_argument("--start-area-frac", type=float, default=0.46)
    e.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("compare")
    c.add_argument("--vocab", type=Path, default=None)
    c.add_argument("--a", nargs="+", required=True, help="populated_graphs dirs (reference)")
    c.add_argument("--b", nargs="+", required=True, help="populated_graphs dirs (templated)")
    args = ap.parse_args()
    {"harvest": cmd_harvest, "populate": cmd_populate, "compare": cmd_compare,
     "retask": cmd_retask, "constraints-eval": cmd_constraints_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
