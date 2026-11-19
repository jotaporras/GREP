"""LLM-free ("templated") populate for e22 floor-plan skeletons.

Why: the Gemma populate prompt + answer no longer fit the vLLM context above
~400 regions (prompt 52k tokens at 550 regions, 90k at 1000), so geo_v2 and the
geo_scale evaluation set name graphs and write tasks deterministically instead.
To keep templated graphs close to the Gemma-written geo_v1 corpus, every choice
is sampled from a vocabulary HARVESTED from geo_v1 (:func:`harvest_vocab`):

- region names ``<zone>_<kind>_<k>``: zone prefixes, per-prefix kinds, and the
  rate at which a zone borrows another zone's prefix;
- object names ``<stem>_<k>`` per host-zone prefix, and the attribute words of
  ``__FILL__`` descriptions;
- task text: whole geo_v1 task sentences with the start / goal / waypoint /
  object mentions replaced by slots, kept per role signature
  (starting area vs named start, region vs object goal, with or without via),
  with their geo_v1 frequencies; the mention styles ("the third command room",
  "the medical hall 4", "admin office 1") with their frequencies;
- acceptance criteria with node ids replaced by slots.

Roles (named start / starting area / via / object goal) and hop lengths come from
the same samplers the LLM path uses (graph_gen.sample_longhop_constraints and
assign_longhop_roles), so a templated task differs from a geo_v1 task only in who
wrote the names and the sentence. Everything is deterministic given the rng.

Entry points: :func:`harvest_vocab` (build the vocabulary JSON once) and
:func:`populate_skeleton` (skeleton dict + constraints -> populated dict with
``graph`` / ``tasks`` / ``description``, the layout of data_gen_*.json).
CLI: scripts/e22_template_populate.py.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ORDINALS = ["zeroth", "first", "second", "third", "fourth", "fifth", "sixth", "seventh",
            "eighth", "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
            "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth"]
_NUM = re.compile(r"_(\d+)$")
NODE_ID = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_\d+\b")
START_AREA = re.compile(
    r"starting (area|location|point|position)|current (location|position)", re.IGNORECASE)
# Region mention styles, most specific first. {w} = name words, {k} = counter.
REGION_STYLES = ("id", "ord_the", "num_the", "num", "bare_the",
                 "zposs_num", "zposs_ord", "ord_in_zone", "ord_of_zone")
OBJECT_STYLES = ("desc_the", "ord_the", "num_the", "bare_the")
SLOTS = ("{S}", "{G}", "{V}", "{O}")


def split_name(name: str) -> Tuple[str, int]:
    """``medical_hall_4`` -> (``medical_hall``, 4)."""
    m = _NUM.search(name)
    if not m:
        return name, 0
    return name[: m.start()], int(m.group(1))


def _ordinal(k: int) -> Optional[str]:
    return ORDINALS[k] if 0 < k < len(ORDINALS) else None


# --------------------------------------------------------------------------- mentions

def _zone_split(stem: str, prefixes: Sequence[str]) -> Optional[Tuple[str, str]]:
    """(zone words, kind words) of a region stem, by the longest graph zone prefix."""
    for pre in sorted(prefixes, key=len, reverse=True):
        if pre and stem.startswith(pre + "_"):
            return pre.replace("_", " "), stem[len(pre) + 1:].replace("_", " ")
    return None


def region_forms(name: str, unique_stem: bool, prefixes: Sequence[str] = ()) -> Dict[str, str]:
    """Surface forms of a region id, keyed by style. ``bare_the`` ("the medical
    hall") only when the stem is unique in the graph, so a rendered mention is
    never ambiguous (harvest passes unique_stem=True to detect it anyway)."""
    stem, k = split_name(name)
    words = stem.replace("_", " ")
    out = {"id": name, "num_the": f"the {words} {k}", "num": f"{words} {k}"}
    if _ordinal(k):
        out["ord_the"] = f"the {_ordinal(k)} {words}"
    if unique_stem:
        out["bare_the"] = f"the {words}"
    split = _zone_split(stem, prefixes)
    if split:
        zone, kind = split
        out["zposs_num"] = f"the {zone}'s {kind} {k}"
        if _ordinal(k):
            o = _ordinal(k)
            out["zposs_ord"] = f"the {zone}'s {o} {kind}"
            out["ord_in_zone"] = f"the {o} {kind} in the {zone}"
            out["ord_of_zone"] = f"the {o} {kind} of the {zone}"
    return out


def object_forms(name: str, desc: str, unique_stem: bool) -> Dict[str, str]:
    stem, k = split_name(name)
    words = stem.replace("_", " ")
    out = {"num_the": f"the {words} {k}"}
    if _ordinal(k):
        out["ord_the"] = f"the {_ordinal(k)} {words}"
    if unique_stem:
        out["bare_the"] = f"the {words}"
        if desc:
            out["desc_the"] = f"the {desc} {words}"
    return out


def _find_mention(text: str, forms: Dict[str, str]) -> Optional[Tuple[int, int, str]]:
    """Longest matching form (case-insensitive, word-bounded): (start, end, style)."""
    best = None
    for style, form in forms.items():
        for m in re.finditer(r"(?<![\w])" + re.escape(form) + r"(?![\w])", text, re.IGNORECASE):
            span = (m.start(), m.end(), style)
            if best is None or (span[1] - span[0]) > (best[1] - best[0]):
                best = span
    return best


# --------------------------------------------------------------------------- harvest

def _zone_prefix(stems: Sequence[str]) -> Optional[str]:
    """Dominant token prefix of a zone's region stems (all but the last token)."""
    c = collections.Counter("_".join(s.split("_")[:-1]) for s in stems if "_" in s)
    if not c:
        return None
    pre, n = c.most_common(1)[0]
    return pre if pre and n >= 0.6 * len(stems) else None


def _signature(named: bool, obj: bool, via: bool) -> str:
    return f"{'named' if named else 'area'}|{'obj' if obj else 'reg'}|{'via' if via else 'direct'}"


def harvest_vocab(pairs: Iterable[Tuple[dict, dict, Optional[list]]]) -> dict:
    """Build the templating vocabulary.

    ``pairs`` yields (skeleton, populated, manifest_constraints_or_None) with the
    populated graph in skeleton order (regions/objects aligned by index, as
    graph_gen.apply_rename produces). Constraints carry skeleton ids.
    """
    prefixes = collections.Counter()
    kinds_by_prefix: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    kinds_all = collections.Counter()
    objs_by_prefix: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    objs_all = collections.Counter()
    descs = collections.Counter()
    n_regions = n_borrowed = 0
    region_style = collections.Counter()
    object_style = collections.Counter()
    templates: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    criteria: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    n_tasks = n_templated = 0
    skipped = collections.Counter()
    skipped_examples: List[str] = []

    def skip(reason, text):
        skipped[reason] += 1
        if len(skipped_examples) < 40:
            skipped_examples.append(f"{reason}: {text}")

    for skel, pop, cons in pairs:
        sg, pg = skel["graph"], pop["graph"]
        rmap = {a["name"]: b["name"] for a, b in zip(sg["regions"], pg["regions"])}
        omap = {a["name"]: b["name"] for a, b in zip(sg["objects"], pg["objects"])}
        zones = skel.get("_metadata", {}).get("zones", {})
        by_zone = collections.defaultdict(list)
        for old, new in rmap.items():
            by_zone[zones.get(old, -1)].append(split_name(new)[0])
        zone_pre = {}
        for z, stems in by_zone.items():
            pre = _zone_prefix(stems)
            zone_pre[z] = pre
            if pre is None:
                continue
            prefixes[pre] += 1
            for s in stems:
                n_regions += 1
                if s.startswith(pre + "_"):
                    kind = s[len(pre) + 1:]
                    kinds_by_prefix[pre][kind] += 1
                    kinds_all[kind] += 1
                else:
                    n_borrowed += 1
        region_zone_pre = {rmap[o]: zone_pre.get(zones.get(o, -1)) for o in rmap}
        graph_prefixes = [p for p in zone_pre.values() if p]
        host = {}
        for a, b in pg["object_connections"]:
            host[a] = b
        for o in pg["objects"]:
            stem = split_name(o["name"])[0]
            objs_all[stem] += 1
            pre = region_zone_pre.get(host.get(o["name"]))
            if pre:
                objs_by_prefix[pre][stem] += 1
            if o.get("description"):
                descs[o["description"]] += 1
        if not cons:
            continue
        stem_count = collections.Counter(split_name(r["name"])[0] for r in pg["regions"])
        ostem_count = collections.Counter(split_name(o["name"])[0] for o in pg["objects"])
        odesc = {o["name"]: o.get("description", "") for o in pg["objects"]}
        hosted = collections.defaultdict(list)
        for o, r in host.items():
            hosted[r].append(o)
        tasks = pop["tasks"][-len(cons):]
        for c, t in zip(cons, tasks):
            n_tasks += 1
            init, goal = rmap[c["init"]], rmap[c["goal"]]
            via = rmap[c["via"]] if c.get("via") else None
            text = t["task"]
            spans = []
            named = not START_AREA.search(text)
            if named:
                m = _find_mention(text, region_forms(init, True, graph_prefixes))
                if not m:
                    skip(f"start {init}", text)
                    continue
                spans.append((m, "S"))
                region_style[m[2]] += 1
            obj = None
            if c.get("object_goal"):
                for o in hosted.get(goal, []):
                    m = _find_mention(text, object_forms(o, odesc.get(o, ""), True))
                    if m:
                        obj = o
                        spans.append((m, "O"))
                        object_style[m[2]] += 1
                        break
            if obj is None:
                m = _find_mention(text, region_forms(goal, True, graph_prefixes))
                if not m:
                    skip(f"goal {goal} obj={c.get('object_goal')} {hosted.get(goal)}", text)
                    continue
                spans.append((m, "G"))
                region_style[m[2]] += 1
            if via:
                m = _find_mention(text, region_forms(via, True, graph_prefixes))
                if not m:
                    skip(f"via {via}", text)
                    continue
                spans.append((m, "V"))
                region_style[m[2]] += 1
            spans.sort(key=lambda s: s[0][0])
            if any(a[0][1] > b[0][0] for a, b in zip(spans, spans[1:])):
                skip("overlap", text)
                continue
            out, pos = [], 0
            for (s, e, style), slot in spans:
                out += [text[pos:s], "{" + slot + "|" + style + "}"]
                pos = e
            out.append(text[pos:])
            tmpl = "".join(out)
            if NODE_ID.search(tmpl):  # an unreplaced id would leak a wrong node
                skip("leftover id", tmpl)
                continue
            # Drop the object's own kind word next to {O} only via slots; the goal
            # kind phrase ("the office containing {O}") is kept generic below.
            tmpl = re.sub(r"\bthe (\w+) (with|containing|holding|housing) (\{O\|\w+\})",
                          lambda m: f"the area {m.group(2)} {m.group(3)}", tmpl)
            sig = _signature(named, obj is not None, via is not None)
            templates[sig][tmpl] += 1
            n_templated += 1
            crit = t["acceptance_criterion"]
            for node, slot in ((init, "{S}"), (goal, "{G}"), (via, "{V}"), (obj, "{O}")):
                if node:
                    crit = re.sub(r"\b" + re.escape(node) + r"\b", slot, crit)
            if not NODE_ID.search(crit) and "<=>" not in crit:
                criteria[sig][crit] += 1

    def top(c: collections.Counter, n: Optional[int] = None) -> dict:
        return dict(c.most_common(n))

    return {
        "prefixes": top(prefixes),
        "kinds_by_prefix": {p: top(c) for p, c in kinds_by_prefix.items()},
        "kinds_all": top(kinds_all),
        "objects_by_prefix": {p: top(c) for p, c in objs_by_prefix.items()},
        "objects_all": top(objs_all),
        "descriptions": top(descs),
        "borrow_rate": n_borrowed / max(n_regions, 1),
        "region_style": top(region_style),
        "object_style": top(object_style),
        "templates": {s: top(c) for s, c in templates.items()},
        "criteria": {s: top(c) for s, c in criteria.items()},
        "stats": {"tasks_seen": n_tasks, "tasks_templated": n_templated,
                  "skipped": dict(skipped), "skipped_examples": skipped_examples},
    }


def load_geo_pairs(shard_dirs: Iterable[Path]):
    """(skeleton, populated, constraints) for a geo_v1 local layout:
    <shard>/skeletons/*graph*json + <shard>/populated_graphs/data_gen_*.json."""
    for sh in shard_dirs:
        sh = Path(sh)
        skel = sorted(sh.glob("skeletons/*graph*json"))
        man_path = sh / "populated_graphs" / "longhop_manifest.json"
        manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
        for p in sorted(sh.glob("populated_graphs/data_gen_*.json")):
            idx = int(p.stem.split("_")[-1])
            s, q = json.loads(skel[idx].read_text()), json.loads(p.read_text())
            if [r["coords"] for r in s["graph"]["regions"]] != [r["coords"] for r in q["graph"]["regions"]]:
                continue  # misaligned pair: never harvest from it
            yield s, q, manifest.get(p.stem)


# --------------------------------------------------------------------------- populate

def _draw(rng, counts: Dict[str, float], exclude=()) -> str:
    keys = [k for k in counts if k not in exclude]
    p = np.array([counts[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=p / p.sum()))]


def name_graph(skel: dict, vocab: dict, rng, kind_alpha: float = 0.3) -> Tuple[dict, dict, list]:
    """Rename map {skeleton id: new id}, descriptions {skeleton id: attr}, and the
    zone prefixes used (for zone-relative mentions such as "the bio lab's third room")."""
    g = skel["graph"]
    zones = skel.get("_metadata", {}).get("zones", {})
    zone_ids = sorted({zones.get(r["name"], 0) for r in g["regions"]})
    used_pre: set = set()
    pre_of = {}
    for z in zone_ids:
        pre = _draw(rng, vocab["prefixes"], exclude=used_pre)
        used_pre.add(pre)
        pre_of[z] = pre
    all_pre = list(pre_of.values())
    kinds_all = vocab["kinds_all"]
    stems = {}
    for r in g["regions"]:
        pre = pre_of[zones.get(r["name"], 0)]
        if len(all_pre) > 1 and rng.random() < vocab["borrow_rate"]:
            pre = all_pre[int(rng.integers(len(all_pre)))]
        own = vocab["kinds_by_prefix"].get(pre, {})
        tot_own = sum(own.values()) or 1.0
        tot_all = sum(kinds_all.values())
        mix = {k: v / tot_own for k, v in own.items()}
        for k, v in kinds_all.items():
            mix[k] = mix.get(k, 0.0) + kind_alpha * v / tot_all
        # No "robotics_bay_bay": the kind must not repeat the zone's last word.
        last = pre.rsplit("_", 1)[-1]
        mix = {k: v for k, v in mix.items() if k.split("_")[0] != last} or mix
        stems[r["name"]] = f"{pre}_{_draw(rng, mix)}"
    rename = _number(stems, rng)
    # Objects: stem by the host region's zone prefix, mixed with the global pool.
    host = {a: b for a, b in g["object_connections"]}
    region_pre = {r["name"]: pre_of[zones.get(r["name"], 0)] for r in g["regions"]}
    ostems = {}
    objs_all = vocab["objects_all"]
    region_stems = set(stems.values())
    for o in g["objects"]:
        own = vocab["objects_by_prefix"].get(region_pre.get(host.get(o["name"])), {})
        tot_own = sum(own.values()) or 1.0
        tot_all = sum(objs_all.values())
        mix = {k: v / tot_own for k, v in own.items()}
        for k, v in objs_all.items():
            mix[k] = mix.get(k, 0.0) + kind_alpha * v / tot_all
        ostems[o["name"]] = _draw(rng, mix, exclude=region_stems)
    rename.update(_number(ostems, rng))
    descriptions = {o["name"]: _draw(rng, vocab["descriptions"])
                    for o in g["objects"] if o.get("description") == "__FILL__"}
    return rename, descriptions, all_pre


def _number(stems: Dict[str, str], rng) -> Dict[str, str]:
    """Number each stem's members 1..m in a random order."""
    groups = collections.defaultdict(list)
    for old, stem in stems.items():
        groups[stem].append(old)
    out = {}
    for stem in sorted(groups):
        olds = sorted(groups[stem])
        order = rng.permutation(len(olds))
        for k, i in enumerate(order, start=1):
            out[olds[int(i)]] = f"{stem}_{k}"
    return out


# A slot keeps the style it was harvested with, because the template text around
# it depends on that style ("From {S|id} area", "From {S|num_the}, ..."). When that
# form does not exist for the rendered node (an ordinal above "twentieth", a bare
# name for a non-unique stem), fall back within the same article group.
_FALLBACK = {
    "ord_the": ("ord_the", "num_the"),
    "bare_the": ("bare_the", "ord_the", "num_the"),
    "desc_the": ("desc_the", "bare_the", "ord_the", "num_the"),
    "num_the": ("num_the",),
    "zposs_ord": ("zposs_ord", "zposs_num", "num_the"),
    "zposs_num": ("zposs_num", "num_the"),
    "ord_in_zone": ("ord_in_zone", "num_the"),
    "ord_of_zone": ("ord_of_zone", "num_the"),
    "num": ("num",),
    "id": ("id",),
}
STYLED_SLOT = re.compile(r"\{([SGVO])\|(\w+)\}")


def _styled(forms: Dict[str, str], style: str) -> str:
    for s in _FALLBACK.get(style, (style,)):
        if s in forms:
            return forms[s]
    return forms["num_the"]


def _capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def render_task(c: dict, graph: dict, vocab: dict, rng, start_area: bool,
                prefixes: Sequence[str] = ()) -> dict:
    """One task dict for a renamed constraint ``c`` (final ids)."""
    stem_count = collections.Counter(split_name(r["name"])[0] for r in graph["regions"])
    ostem_count = collections.Counter(split_name(o["name"])[0] for o in graph["objects"])
    odesc = {o["name"]: o.get("description", "") for o in graph["objects"]}
    init, goal, via = c["init"], c["goal"], c.get("via")
    obj = None
    if c.get("object_goal"):
        hosted = sorted(a for a, b in graph["object_connections"] if b == goal)
        obj = hosted[int(rng.integers(len(hosted)))] if hosted else None
    named = not start_area
    sig = _signature(named, obj is not None, via is not None)
    tmpl = _draw(rng, vocab["templates"][sig])

    node = {"S": init, "G": goal, "V": via, "O": obj}

    def fill(m):
        slot, style = m.group(1), m.group(2)
        n = node[slot]
        if slot == "O":
            forms = object_forms(n, odesc.get(n, ""), ostem_count[split_name(n)[0]] == 1)
        else:
            forms = region_forms(n, stem_count[split_name(n)[0]] == 1, prefixes)
        return _styled(forms, style)

    text = _capitalize_first(STYLED_SLOT.sub(fill, tmpl))
    crit = _draw(rng, vocab["criteria"][sig])
    for slot, node in (("{S}", init), ("{G}", goal), ("{V}", via), ("{O}", obj)):
        if node:
            crit = crit.replace(slot, node)
    chain = [init] + ([via] if via else []) + [goal]
    answer = "(?i)" + r".*".join(rf"\b{n}\b" for n in chain)
    task = {"task": text, "answer": answer, "init_node": init, "acceptance_criterion": crit}
    if c.get("named_start"):
        task["keep_robot_location"] = True
    return task


def populate_skeleton(skel: dict, constraints: List[dict], vocab: dict, rng,
                      start_area_frac: float) -> Tuple[dict, List[dict]]:
    """Populated ``{"graph", "tasks", "description"}`` plus constraints in FINAL ids.

    ``constraints`` come from graph_gen.assign_longhop_roles on the skeleton.
    ``start_area_frac`` of the constraints WITHOUT a named start are worded from
    "the starting area" (geo_v1 --longhop-start-area-frac); the rest name it.
    """
    from prism.data import graph_gen

    rename, descriptions, prefixes = name_graph(skel, vocab, rng)
    graph = graph_gen.apply_rename(skel, rename, descriptions)
    final = []
    for c in constraints:
        f = dict(c, init=rename[c["init"]], goal=rename[c["goal"]])
        if c.get("via"):
            f["via"] = rename[c["via"]]
        final.append(f)
    unnamed = [i for i, c in enumerate(final) if not c.get("named_start")]
    n_area = int(round(start_area_frac * len(unnamed)))
    area = set(rng.choice(unnamed, size=n_area, replace=False).tolist()) if n_area else set()
    tasks = [render_task(c, graph, vocab, rng, start_area=i in area, prefixes=prefixes)
             for i, c in enumerate(final)]
    return {"graph": graph, "tasks": tasks, "description": "templated (e22 geo_v2)"}, final


# --------------------------------------------------------------------------- strata

def parse_strata(spec: str) -> List[Tuple[int, int]]:
    """``"1-6,7-15,16-30,31-999"`` -> [(1, 6), (7, 15), (16, 30), (31, 999)]."""
    out = []
    for part in spec.split(","):
        lo, hi = part.split("-")
        out.append((int(lo), int(hi)))
    return out


def sample_stratified_constraints(graph: dict, n: int, rng, strata, object_goal_frac: float,
                                  offset: int = 0):
    """Eval sampler: task k goes to stratum (k + offset) mod S (round robin over
    the S strata this graph has), hop uniform within the stratum. Pass the graph
    index as ``offset`` so that n % S leftovers rotate across graphs (8 graphs x
    6 tasks over 4 strata -> 12 per stratum). Object-goal slots are spread evenly
    over k, so each stratum gets ~``object_goal_frac`` of them."""
    from prism.data import graph_gen

    dist = graph_gen._region_distances(graph)
    hosts = graph_gen._object_hosts(graph)
    have = {h for row in dist.values() for h in row.values() if h}
    have_obj = {h for s, row in dist.items() for t, h in row.items() if h and t in hosts}
    present = [s for s in strata if any(s[0] <= h <= s[1] for h in have)]
    out = []
    n_obj = int(round(object_goal_frac * n))
    obj_slots = set(np.linspace(0, n - 1, n_obj).round().astype(int).tolist()) if n_obj else set()
    for k in range(n):
        lo, hi = present[(k + offset) % len(present)]
        want_obj = k in obj_slots and any(lo <= h <= hi for h in have_obj)
        weights = {h: 1.0 for h in range(lo, hi + 1) if h in (have_obj if want_obj else have)}
        c = graph_gen._sample_weighted_constraints(graph, 1, rng, weights,
                                                   1.0 if want_obj else 0.0)[0]
        out.append(c)
    return out
