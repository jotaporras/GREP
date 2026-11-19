"""e22 templated (LLM-free) populate: src/prism/data/template_populate.py.

CPU only. Builds floor-plan skeletons with scripts/generate_eval_graphs.py and
names/tasks them with the harvested vocabulary (scripts/e22_template_vocab.json).

- the populate CLI is deterministic given --seed (byte-identical files);
- names follow <zone>_<kind>_<k>, are unique, and the topology is untouched;
- every task passes graph_gen's long-hop validator and the oracle route grader
  accepts the answer regex for the shortest path;
- named-start tasks set keep_robot_location and never say "starting area";
- the eval sampler gives 12 tasks per route-length stratum for 8 graphs x 6 tasks.

Run: PYTHONPATH=<SPINE>/src python -m pytest tests/test_template_populate.py -v
(graph_gen imports spine; the test is skipped when spine is not importable).
"""
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
nx = pytest.importorskip("networkx")
pytest.importorskip("spine")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from prism.data import graph_gen, template_populate as tp  # noqa: E402

VOCAB_PATH = ROOT / "scripts" / "e22_template_vocab.json"
REGION_NAME = re.compile(r"^[a-z][a-z0-9]*(_[a-z][a-z0-9]*)+_\d+$")  # <zone>_<kind>_<k>
OBJECT_NAME = re.compile(r"^[a-z][a-z0-9]*(_[a-z][a-z0-9]*)*_\d+$")  # <stem>_<k>
V1_HOP_WEIGHTS = ("1:0.04,2:0.04,3:0.06,4:0.08,5:0.12,6:0.17,7:0.17,8:0.08,9:0.07,"
                  "10:0.05,11:0.04,12:0.03,13:0.02,14:0.02,15:0.01")


def _skeleton(path: Path, n_regions: int, seed: int) -> Path:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_eval_graphs.py"),
         "--layout", "floorplan", "--n-regions", str(n_regions),
         "--door-extra-min", "0.05", "--door-extra-max", "0.20", "--corridor-bias", "0.3",
         "--zone-depth", "3", "--region-area", "25.0", "--robot-location", "random",
         "--object-rate", "0.4", "--description-prob", "0.05", "--n-tasks", "16",
         "--seed", str(seed), "--output", str(path)],
        check=True, capture_output=True)
    return path


@pytest.fixture(scope="module")
def vocab():
    return json.loads(VOCAB_PATH.read_text())


@pytest.fixture(scope="module")
def skel_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("skel")
    for i, (n, seed) in enumerate([(40, 9101), (100, 9102), (200, 9103)]):
        _skeleton(d / f"eval_graph_unique_geo_seed{seed}.json", n, seed)
    return d


def _populate(skel_dir: Path, out: Path, seed: int, extra=()):
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "e22_template_populate.py"), "populate",
         "--vocab", str(VOCAB_PATH), "--skeleton-dir", str(skel_dir), "--out-dir", str(out),
         "--n-tasks", "16", "--named-start-frac", "0.5", "--start-area-frac", "0.4",
         "--via-frac", "0.2", "--object-goal-frac", "0.5", "--seed", str(seed), *extra],
        check=True, capture_output=True)
    return {p.name: p.read_bytes() for p in sorted(out.glob("*.json"))}


def _populate_one(skel: dict, vocab: dict, seed: int, n: int = 16):
    rng = np.random.default_rng([seed, 0])
    weights = graph_gen.parse_hop_weights(V1_HOP_WEIGHTS)
    cons = graph_gen.sample_longhop_constraints(skel["graph"], n, rng, hop_weights=weights,
                                                object_goal_frac=0.5)
    cons = graph_gen.assign_longhop_roles(skel["graph"], cons, rng, named_start_frac=0.5,
                                          via_frac=0.2, max_via_hops=max(weights))
    return tp.populate_skeleton(skel, cons, vocab, rng, start_area_frac=0.4)


class TestCLI:
    def test_deterministic_given_seed(self, skel_dir, tmp_path):
        args = ("--hop-weights", V1_HOP_WEIGHTS)
        a = _populate(skel_dir, tmp_path / "a", 7, args)
        b = _populate(skel_dir, tmp_path / "b", 7, args)
        c = _populate(skel_dir, tmp_path / "c", 8, args)
        assert len(a) == 3 * 2 + 1  # data_gen + graph_gen per skeleton + manifest
        assert a == b
        assert a["data_gen_000.json"] != c["data_gen_000.json"]

    def test_eval_strata_balanced(self, tmp_path):
        """geo_scale layout: 8 graphs x 6 tasks over 4 strata -> 12 per stratum."""
        d = tmp_path / "skel"
        d.mkdir()
        for s in range(8):
            _skeleton(d / f"eval_graph_unique_geo_seed{9200 + s}.json", 200, 9200 + s)
        _populate(d, tmp_path / "out", 7, ("--n-tasks", "6", "--hop-strata", "1-6,7-15,16-30,31-999"))
        man = json.loads((tmp_path / "out" / "longhop_manifest.json").read_text())
        strata = tp.parse_strata("1-6,7-15,16-30,31-999")
        per = collections.Counter()
        n_obj = 0
        for cons in man.values():
            assert len(cons) == 6
            for c in cons:
                per[next(i for i, (lo, hi) in enumerate(strata) if lo <= c["hops"] <= hi)] += 1
                n_obj += bool(c.get("object_goal"))
        diam_ok = all(per[i] == 12 for i in range(4))
        # 200-region floor plans have diameter ~25-35; if a graph lacks the 31+
        # stratum its tasks go round robin over the strata it has.
        assert diam_ok or (per[3] < 12 and sum(per.values()) == 48)
        # an object slot falls back to a region goal when no object sits in that stratum
        assert 8 * 3 - 2 <= n_obj <= 8 * 3


class TestPopulateSkeleton:
    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_names_tasks_and_roles(self, skel_dir, vocab, idx):
        path = sorted(skel_dir.glob("*.json"))[idx]
        skel = json.loads(path.read_text())
        populated, final = _populate_one(skel, vocab, seed=3)
        g = populated["graph"]
        names = [r["name"] for r in g["regions"]] + [o["name"] for o in g["objects"]]
        assert len(set(names)) == len(names)
        bad = [r["name"] for r in g["regions"] if not REGION_NAME.match(r["name"])]
        bad += [o["name"] for o in g["objects"] if not OBJECT_NAME.match(o["name"])]
        assert not bad, bad[:5]
        stems = {tp.split_name(r["name"])[0] for r in g["regions"]}
        assert not {tp.split_name(o["name"])[0] for o in g["objects"]} & stems
        # topology untouched: same edge structure up to the rename
        sg = skel["graph"]
        assert len(g["region_connections"]) == len(sg["region_connections"])
        assert len(g["object_connections"]) == len(sg["object_connections"])
        graph_gen.assert_graph_refs_resolve(g)
        graph_gen.validate_task_refs(populated["tasks"], g)
        graph_gen.validate_longhop_tasks(populated["tasks"], final, g)
        G = nx.Graph([tuple(e) for e in g["region_connections"]])
        n_area = 0
        for task, c in zip(populated["tasks"], final):
            text = task["task"]
            assert "{" not in text and "}" not in text
            if c.get("named_start"):
                assert task.get("keep_robot_location") is True
                assert not tp.START_AREA.search(text)
                assert c["init"] != g["robot_location"]
            else:
                n_area += bool(tp.START_AREA.search(text))
            # the shortest path satisfies the answer regex (via in order)
            if c.get("via"):
                path_nodes = (nx.shortest_path(G, c["init"], c["via"])
                              + nx.shortest_path(G, c["via"], c["goal"])[1:])
            else:
                path_nodes = nx.shortest_path(G, c["init"], c["goal"])
            assert re.search(task["answer"], " -> ".join(path_nodes)), (task["answer"], path_nodes)
        unnamed = sum(1 for c in final if not c.get("named_start"))
        assert n_area == round(0.4 * unnamed)

    def test_same_rng_same_output(self, skel_dir, vocab):
        skel = json.loads(sorted(skel_dir.glob("*.json"))[1].read_text())
        assert _populate_one(skel, vocab, 5) == _populate_one(skel, vocab, 5)


class TestMentions:
    def test_region_forms(self):
        f = tp.region_forms("bio_lab_room_3", True, ["bio_lab"])
        assert f["id"] == "bio_lab_room_3"
        assert f["num_the"] == "the bio lab room 3"
        assert f["ord_the"] == "the third bio lab room"
        assert f["zposs_ord"] == "the bio lab's third room"
        assert f["ord_in_zone"] == "the third room in the bio lab"
        assert "bare_the" not in tp.region_forms("bio_lab_room_3", False, ["bio_lab"])

    def test_styled_fallback_keeps_article(self):
        # A style the name lacks falls back inside the same article group, so a
        # template "From {S|ord_the}" never renders "From the the ..." or drops "the".
        forms = tp.region_forms("bio_lab_room_3", False)
        assert tp._styled(forms, "bare_the").startswith("the ")
        assert tp._styled(forms, "zposs_ord").startswith("the ")


# --------------------------------------------------------------------------- e22 mix roles

from prism.data.data_gen import DataGenerator  # noqa: E402
from prism.eval import path_validator  # noqa: E402


def _nxg(graph):
    g = nx.Graph()
    g.add_nodes_from(r["name"] for r in graph["regions"])
    g.add_edges_from(map(tuple, graph["region_connections"]))
    return g


def _mix(skel, vocab, seed, mix="direct:1,avoid:1,via:1", n=18, detour=0.8):
    """Name a skeleton (geo_v2 populate), then retask it with mix route types."""
    populated, _ = _populate_one(skel, vocab, seed)
    rng = np.random.default_rng([seed, 1])
    weights = graph_gen.parse_hop_weights(V1_HOP_WEIGHTS)
    cons = graph_gen.sample_longhop_constraints(populated["graph"], n, rng, hop_weights=weights,
                                                object_goal_frac=0.5)
    cons = tp.assign_mix_roles(populated["graph"], cons, rng, 0.5, tp.parse_role_mix(mix),
                               detour, 40)
    prefixes = tp.graph_prefixes(skel, populated["graph"])
    new = tp.retask_graph(populated, prefixes, cons, vocab, rng, 0.46)
    return new, cons


@pytest.fixture(scope="module")
def mixed(skel_dir, vocab):
    out = []
    for p in sorted(skel_dir.glob("*.json")):
        for seed in (1, 2, 3):
            out.append(_mix(json.loads(p.read_text()), vocab, seed))
    return out


class TestAvoidRole:
    def test_validators_accept_mix(self, mixed):
        for new, cons in mixed:
            graph_gen.validate_task_refs(new["tasks"], new["graph"])
            graph_gen.validate_longhop_tasks(new["tasks"], cons, new["graph"])

    def test_oracle_misses_avoid_and_has_recorded_length(self, mixed):
        kinds = collections.Counter()
        for new, cons in mixed:
            g = _nxg(new["graph"])
            for c, t in zip(cons, new["tasks"]):
                goal, way, avoid, _, kind = path_validator.derive_targets(
                    new["graph"], init_node=t["init_node"], answer=t["answer"],
                    criterion=t["acceptance_criterion"], task=t["task"])
                assert goal == c["goal"] and kind == "path"
                assert set(avoid) == ({c["avoid"]} if c["avoid"] else set())
                assert way == ([c["via"]] if c["via"] else [])
                route, why = DataGenerator._oracle_route(new["graph"], t)
                assert route, why
                nodes = route.split(" -> ")
                assert len(nodes) - 1 == c["hops"]
                d0 = nx.shortest_path_length(g, c["init"], c["goal"])
                assert c["direct_hops"] == d0
                if c["avoid"]:
                    assert c["avoid"] not in nodes
                    kinds[c["avoid_kind"]] += 1
                    on_path = graph_gen._on_some_shortest_path(
                        graph_gen._region_distances(new["graph"]), c["init"], c["goal"])
                    assert c["avoid"] in on_path  # never a vacuous avoid
                    if c["avoid_kind"] == "detour":
                        assert c["hops"] > d0
                    else:
                        assert c["hops"] == d0
                m = path_validator.evaluate_sample(
                    t["task"], route, new["graph"], init_node=t["init_node"],
                    answer=t["answer"], acceptance_criterion=t["acceptance_criterion"])
                assert m["structured_correct"], (t["task"], route)
        # floor plans rarely have equal-length alternatives, so nodetour slots
        # mostly fall back to detour (counted in the corpus report)
        assert kinds["detour"] > 0

    def test_avoid_candidates_kinds(self):
        """Square a-b-d / a-c-d: avoiding b is a nodetour (2 hops). Line a-b-d plus
        a loop a-x-y-z-d... with c removed: avoiding b is a 4-hop detour."""
        def graph(edges, robot="a"):
            names = sorted({n for e in edges for n in e})
            return {"regions": [{"name": n} for n in names], "region_connections": edges,
                    "robot_location": robot, "objects": [], "object_connections": []}
        sq = graph([["a", "b"], ["b", "d"], ["a", "c"], ["c", "d"]], robot="z0")
        sq["regions"].append({"name": "z0"})
        sq["region_connections"].append(["z0", "a"])
        cand = tp.avoid_candidates(sq, graph_gen._region_distances(sq), "a", "d", 40)
        assert cand == {"detour": [], "nodetour": [("b", 2), ("c", 2)]}
        loop = graph([["a", "b"], ["b", "d"], ["a", "x"], ["x", "y"], ["y", "z"], ["z", "d"]],
                     robot="y")
        cand = tp.avoid_candidates(loop, graph_gen._region_distances(loop), "a", "d", 40)
        assert cand == {"detour": [("b", 4)], "nodetour": []}
        assert tp.avoid_candidates(loop, graph_gen._region_distances(loop), "a", "d", 3) == \
            {"detour": [], "nodetour": []}  # constrained-length cap

    def test_grader_ties_pass_and_violations_fail(self, mixed):
        """Any equal-length valid route meeting the constraint passes; a route
        through A, or one that skips V, fails the headline metric."""
        n_tie = n_through = n_skip = 0
        for new, cons in mixed:
            g = _nxg(new["graph"])
            for c, t in zip(cons, new["tasks"]):
                kw = dict(init_node=t["init_node"], answer=t["answer"],
                          acceptance_criterion=t["acceptance_criterion"])
                if c["avoid"]:
                    h = g.copy()
                    h.remove_node(c["avoid"])
                    for k, p in enumerate(nx.all_shortest_paths(h, c["init"], c["goal"])):
                        if k == 4:
                            break
                        m = path_validator.evaluate_sample(t["task"], " -> ".join(p), new["graph"], **kw)
                        assert m["structured_correct"]
                        n_tie += 1
                    through = (nx.shortest_path(g, c["init"], c["avoid"])
                               + nx.shortest_path(g, c["avoid"], c["goal"])[1:])
                    m = path_validator.evaluate_sample(t["task"], " -> ".join(through), new["graph"], **kw)
                    assert m["avoid_ok"] is False and not m["structured_correct"]
                    n_through += 1
                elif c["via"]:
                    p = nx.shortest_path(g, c["init"], c["goal"])  # V is off every shortest path
                    m = path_validator.evaluate_sample(t["task"], " -> ".join(p), new["graph"], **kw)
                    assert m["waypoints_ok"] is False and not m["structured_correct"]
                    n_skip += 1
        assert n_tie and n_through and n_skip

    def test_every_avoid_phrasing_resolves_from_task_text(self, vocab):
        """The task sentence alone (no criterion) names A for the grader's _AVOID."""
        graph = {"regions": [{"name": n} for n in ("hall_a_1", "lab_b_2", "office_c_3")],
                 "objects": [], "object_connections": [],
                 "region_connections": [["hall_a_1", "lab_b_2"], ["lab_b_2", "office_c_3"]]}
        for clause in tp.AVOID_CLAUSES:
            for rest in ("", " and list the path.", "; list the path."):
                text = "From hall_a_1, give the route to office_c_3" + clause.replace(
                    "{A}", "lab_b_2") + ("," if clause.startswith(",") and rest.startswith(" and") else "") + rest
                _, _, avoid, _, _ = path_validator.derive_targets(
                    graph, init_node="hall_a_1", answer=r"(?i)\bhall_a_1\b.*\boffice_c_3\b",
                    criterion=None, task=text)
                assert set(avoid) == {"lab_b_2"}, text

    def test_flag_off_keeps_old_validator(self, skel_dir, vocab):
        """Constraints without the ``avoid`` key (geo_v2) keep the old rule: an
        on-route "without using" avoid is rejected."""
        skel = json.loads(sorted(skel_dir.glob("*.json"))[1].read_text())
        populated, final = _populate_one(skel, vocab, 5)
        assert all("avoid" not in c for c in final)
        dist = graph_gen._region_distances(populated["graph"])
        i, c = next((i, c) for i, c in enumerate(final) if not c["via"] and c["hops"] >= 3)
        a = sorted(graph_gen._on_some_shortest_path(dist, c["init"], c["goal"]) - {c["init"], c["goal"]})[0]
        tasks = [dict(t) for t in populated["tasks"]]
        tasks[i]["acceptance_criterion"] = tasks[i]["acceptance_criterion"].rstrip(".") + f" without using {a}."
        with pytest.raises(ValueError, match="hop length would drift"):
            graph_gen.validate_longhop_tasks(tasks, final, populated["graph"])

    def test_detour_fraction_and_counts(self, mixed):
        for _, cons in mixed:
            routes = collections.Counter(c["route"] for c in cons)
            fell = sum(1 for c in cons if c.get("fallback", "").endswith("->direct"))
            assert routes["avoid"] + routes["via"] + routes["direct"] == 18
            assert routes["direct"] == 6 + fell

    def test_constraints_eval_pairs(self, skel_dir):
        skel = json.loads(sorted(skel_dir.glob("*.json"))[2].read_text())
        rng = np.random.default_rng(3)
        cons = tp.sample_paired_constraints(skel["graph"], 3, rng,
                                            tp.parse_strata("1-6,7-15,16-30,31-999"))
        by_pair = collections.defaultdict(list)
        for c in cons:
            by_pair[c["pair"]].append(c)
        assert by_pair
        for p, cs in by_pair.items():
            assert {(c["init"], c["goal"]) for c in cs} == {(cs[0]["init"], cs[0]["goal"])}
            routes = [c["route"] for c in cs]
            assert routes[:2] == ["direct", "avoid"] and cs[1]["avoid_kind"] == "detour"
            assert cs[1]["hops"] > cs[0]["hops"]
