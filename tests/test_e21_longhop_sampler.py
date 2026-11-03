"""e21 hop-stratified endpoint sampling + longhop avoid/grounding directives.

Covers docs/2026-08-24 e21_oracle_scale_design.md:
- sample_longhop_constraints max_boost: diameter bucket oversampled, other
  buckets uniform; max_boost=1.0 reproduces the legacy sampler bit-identically.
- validate_longhop_tasks graph check: an avoided region on a shortest
  init->goal path is rejected (hop drift), an off-path one passes.
- build_prompt flags: grounding_directives / longhop_allow_avoid are
  byte-identical no-ops when off.
- longhop_start_area_frac (v2c): marks that fraction of the constrained tasks
  to be worded "from the starting area" while init_node / grading fields keep
  naming the region, so hop stratification is untouched.
"""

import collections
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

np = pytest.importorskip("numpy")

from prism.data import graph_gen  # noqa: E402


def _line_graph(n=7):
    """room_0 - room_1 - ... : diameter n-1, unique shortest paths."""
    return {
        "regions": [{"name": f"room_{i}", "coords": [i, 0], "description": ""}
                    for i in range(n)],
        "region_connections": [[f"room_{i}", f"room_{i+1}"] for i in range(n - 1)],
        "objects": [],
        "object_connections": [],
        "robot_location": "room_0",
    }


def _cycle_graph(n=6):
    """Even cycle: two equal-length arcs between antipodes."""
    return {
        "regions": [{"name": f"cell_{i}", "coords": [i, 0], "description": ""}
                    for i in range(n)],
        "region_connections": [[f"cell_{i}", f"cell_{(i+1) % n}"] for i in range(n)],
        "objects": [],
        "object_connections": [],
        "robot_location": "cell_0",
    }


class TestMaxBoost:
    def test_boost_one_is_bit_identical_to_legacy(self):
        # Same seed, same graph: the max_boost=1.0 path must consume the rng
        # stream exactly as the pre-e21 sampler did (rng.integers per draw),
        # so old corpora resample identically.
        g = _line_graph()
        a = graph_gen.sample_longhop_constraints(
            g, 50, np.random.default_rng(7))
        b = graph_gen.sample_longhop_constraints(
            g, 50, np.random.default_rng(7), max_boost=1.0)
        assert a == b

    def test_boost_oversamples_diameter_bucket(self):
        # Line graph n=7: diameter 6, buckets {3,4,5,6}. With max_boost=3 the
        # diameter bucket carries 3/6 of the mass, the rest 1/6 each.
        g = _line_graph(7)
        rng = np.random.default_rng(0)
        hops = [c["hops"] for c in
                graph_gen.sample_longhop_constraints(g, 6000, rng, max_boost=3.0)]
        freq = collections.Counter(hops)
        assert set(freq) == {3, 4, 5, 6}
        assert abs(freq[6] / 6000 - 0.5) < 0.03
        for h in (3, 4, 5):
            assert abs(freq[h] / 6000 - 1 / 6) < 0.03

    def test_boost_valid_endpoints_and_recorded_hops(self):
        g = _line_graph(7)
        for c in graph_gen.sample_longhop_constraints(
                g, 200, np.random.default_rng(1), max_boost=5.0):
            # On the line graph the true distance IS |i - j|.
            assert abs(int(c["init"].rsplit("_", 1)[1]) - int(c["goal"].rsplit("_", 1)[1])) == c["hops"]

    def test_single_bucket_graph_unaffected_by_boost(self):
        # Diameter 3 == min_hops: only one bucket, boost must be a no-op.
        g = _line_graph(4)
        out = graph_gen.sample_longhop_constraints(
            g, 20, np.random.default_rng(2), max_boost=10.0)
        assert {c["hops"] for c in out} == {3}

    def test_nonpositive_boost_rejected(self):
        with pytest.raises(ValueError):
            graph_gen.sample_longhop_constraints(
                _line_graph(), 1, np.random.default_rng(0), max_boost=0.0)


def _longhop_task(init="room_0", goal="room_4", avoid=None):
    crit = f"A correct answer gives a valid route from {init} to {goal}"
    if avoid:
        crit += f" without using {avoid}"
    crit += "."
    return {
        "task": "Route to the far room; give the path and edges.",
        "answer": rf"(?i)\b{init}\b.*\b{goal}\b",
        "init_node": init,
        "acceptance_criterion": crit,
    }


class TestAvoidValidation:
    def test_on_path_avoid_rejected(self):
        # Line graph: room_2 lies on the unique shortest room_0->room_4 path.
        g = _line_graph()
        with pytest.raises(ValueError, match="hop length would drift"):
            graph_gen.validate_longhop_tasks(
                [_longhop_task(avoid="room_2")],
                [{"init": "room_0", "goal": "room_4", "hops": 4}], graph=g)

    def test_off_path_avoid_accepted(self):
        # room_6 is beyond the goal — off every shortest room_0->room_4 path.
        g = _line_graph()
        graph_gen.validate_longhop_tasks(
            [_longhop_task(avoid="room_6")],
            [{"init": "room_0", "goal": "room_4", "hops": 4}], graph=g)

    def test_avoid_on_any_of_multiple_shortest_paths_rejected(self):
        # Even cycle cell_0..cell_5: both cell_1 and cell_4 sit on SOME
        # shortest cell_0->cell_3 path — either must be rejected.
        g = _cycle_graph(6)
        for avoid in ("cell_1", "cell_4"):
            with pytest.raises(ValueError, match="hop length would drift"):
                graph_gen.validate_longhop_tasks(
                    [_longhop_task(init="cell_0", goal="cell_3", avoid=avoid)],
                    [{"init": "cell_0", "goal": "cell_3", "hops": 3}], graph=g)

    def test_avoiding_start_or_goal_rejected(self):
        g = _line_graph()
        with pytest.raises(ValueError, match="start/goal"):
            graph_gen.validate_longhop_tasks(
                [_longhop_task(avoid="room_4")],
                [{"init": "room_0", "goal": "room_4", "hops": 4}], graph=g)

    def test_no_graph_keeps_legacy_behavior(self):
        # Without graph= the avoid clause is not inspected (pre-e21 contract).
        graph_gen.validate_longhop_tasks(
            [_longhop_task(avoid="room_2")],
            [{"init": "room_0", "goal": "room_4", "hops": 4}])


class TestPromptFlags:
    def _prompt(self, **kw):
        gen = graph_gen.TaskGraphGen.__new__(graph_gen.TaskGraphGen)
        return gen.build_prompt(
            base_graph="{}", n_tasks=4,
            longhop_constraints=[{"init": "room_0", "goal": "room_4", "hops": 4}], **kw)

    def test_flags_off_is_byte_identical_noop(self):
        assert self._prompt() == self._prompt(
            grounding_directives=False, longhop_allow_avoid=False)
        assert "Start-reference diversity" not in self._prompt()
        assert "Do not add waypoint or avoid constraints" in self._prompt()

    def test_grounding_directives_block_present_when_on(self):
        p = self._prompt(grounding_directives=True)
        assert "Start-reference diversity" in p
        assert "robot_location" in p

    def test_longhop_allow_avoid_swaps_the_ban(self):
        p = self._prompt(longhop_allow_avoid=True)
        assert "Do not add waypoint or avoid constraints" not in p
        assert "optimal route already bypasses" in p
        assert "Waypoint constraints are still FORBIDDEN" in p


MARKER = "Word this task's START as the robot's current position"


class TestStartAreaPhrasing:
    def _prompt(self, n_lh=10, **kw):
        gen = graph_gen.TaskGraphGen.__new__(graph_gen.TaskGraphGen)
        return gen.build_prompt(
            base_graph="{}", n_tasks=12,
            longhop_constraints=[
                {"init": f"room_{i}", "goal": "room_6", "hops": 4}
                for i in range(n_lh)
            ], **kw)

    def test_default_is_byte_identical_noop(self):
        assert self._prompt() == self._prompt(longhop_start_area_frac=0.0)
        assert MARKER not in self._prompt()

    def test_fraction_marks_that_many_tasks(self):
        # 0.4 x 10 constrained tasks -> 4 marked, the rest keep named starts.
        assert self._prompt(longhop_start_area_frac=0.4).count(MARKER) == 4
        assert self._prompt(longhop_start_area_frac=1.0).count(MARKER) == 10

    def test_marked_tasks_still_ground_the_grading_fields(self):
        # The whole point: the wording changes, the labels do not.
        p = self._prompt(longhop_start_area_frac=0.4)
        assert "init_node" in p
        assert "must still name" in p and "NEW node id" in p

    def test_endpoints_are_unchanged_by_the_flag(self):
        # Hop stratification must survive: every task still states its fixed
        # base-graph endpoints and hop count.
        p = self._prompt(longhop_start_area_frac=0.4)
        assert p.count("optimal route: 4 hops") == 10

    def test_out_of_range_rejected(self):
        for bad in (-0.1, 1.5):
            with pytest.raises(ValueError, match="longhop_start_area_frac"):
                self._prompt(longhop_start_area_frac=bad)


class TestBatchedPopulateThreadsTheFlag:
    """The vLLM path builds its prompts in populate_graphs_and_tasks_batch, not
    in get_tasks. v2c's corpus came out byte-identical to v2b's because that
    path dropped the kwarg on the floor."""

    def _batched_prompt(self, frac, grounding=False, allow_avoid=False):
        from prism.data import data_gen as dg

        longhop = [{"init": f"room_{i}", "goal": "room_6", "hops": 4}
                   for i in range(10)]
        seen = {}

        class _Client:
            @staticmethod
            def batch_query_gpt_5(prompts, **_kw):
                seen["prompt"] = prompts[0]
                return [""]

        class _Ctx(graph_gen.TaskGraphGen):
            def __init__(self):
                self.client = _Client()

            def parse_response(self, *_a, **_kw):
                raise RuntimeError("stop after prompt construction")

        gen = dg.DataGenerator.__new__(dg.DataGenerator)
        gen.longhop_start_area_frac = frac
        gen.grounding_directives = grounding
        gen.longhop_allow_avoid = allow_avoid
        gen.context_gen = _Ctx()
        gen._load_valid_populated = lambda _path: None
        gen._sample_task_labels = lambda _g, _n: (None, None, longhop)

        gen.populate_graphs_and_tasks_batch(["{}"], log_dir="/dev/null", n_tasks=12)
        return seen["prompt"]

    def test_flag_reaches_the_batched_prompt(self):
        assert self._batched_prompt(0.4).count(MARKER) == 4

    def test_default_batched_prompt_is_unmarked(self):
        assert MARKER not in self._batched_prompt(0.0)

    def test_grounding_and_avoid_reach_the_batched_prompt(self):
        # e22: the batched path also dropped grounding_directives and
        # longhop_allow_avoid (e21 v2/v2b/v2c never saw them).
        p = self._batched_prompt(0.0, grounding=True, allow_avoid=True)
        assert "Start-reference diversity" in p
        assert "optimal route already bypasses" in p
        off = self._batched_prompt(0.0)
        assert "Start-reference diversity" not in off
        assert "Do not add waypoint or avoid constraints" in off


# --------------------------------------------------------------------------
# e22 (docs/2026-11-02 e22_geometric_scale_design.md §3): hop_weights sampler,
# named-start / via / object-goal roles, via validator, prompt blocks.
# --------------------------------------------------------------------------

def _grid_graph(rows=4, cols=4, robot="g_0_0", objects=()):
    """rows x cols grid g_r_c; ``objects`` = host regions of one object each."""
    regions = [{"name": f"g_{r}_{c}", "coords": [c, r], "description": ""}
               for r in range(rows) for c in range(cols)]
    edges = ([[f"g_{r}_{c}", f"g_{r}_{c+1}"] for r in range(rows) for c in range(cols - 1)]
             + [[f"g_{r}_{c}", f"g_{r+1}_{c}"] for r in range(rows - 1) for c in range(cols)])
    return {
        "regions": regions,
        "region_connections": edges,
        "objects": [{"name": f"crate_{k + 1}", "coords": [0, 0], "description": ""}
                    for k in range(len(objects))],
        "object_connections": [[f"crate_{k + 1}", host] for k, host in enumerate(objects)],
        "robot_location": robot,
    }


def _nx(graph):
    nx = pytest.importorskip("networkx")
    G = nx.Graph()
    G.add_edges_from(map(tuple, graph["region_connections"]))
    return nx, G


class TestHopWeights:
    def test_parse_hop_weights(self):
        assert graph_gen.parse_hop_weights("1:0.5,6:2") == {1: 0.5, 6: 2.0}
        for bad in ("0:1", "3:-1", "3:0,4:0"):
            with pytest.raises(ValueError, match="bad hop weights"):
                graph_gen.parse_hop_weights(bad)

    def test_none_is_the_legacy_stream(self):
        g = _line_graph()
        a = graph_gen.sample_longhop_constraints(g, 50, np.random.default_rng(7))
        b = graph_gen.sample_longhop_constraints(
            g, 50, np.random.default_rng(7), hop_weights=None)
        assert a == b
        assert all(set(c) == {"init", "goal", "hops"} for c in a)

    def test_frequencies_follow_the_weights(self):
        # Line graph n=8: buckets 1..7 exist; mass only on 2 and 5 (1:3).
        g = _line_graph(8)
        out = graph_gen.sample_longhop_constraints(
            g, 6000, np.random.default_rng(0), hop_weights={2: 1.0, 5: 3.0})
        freq = collections.Counter(c["hops"] for c in out)
        assert set(freq) == {2, 5}
        assert abs(freq[5] / 6000 - 0.75) < 0.03
        for c in out:
            assert abs(int(c["init"].rsplit("_", 1)[1]) - int(c["goal"].rsplit("_", 1)[1])) == c["hops"]

    def test_missing_buckets_are_skipped(self):
        # Diameter 3: buckets 9 and 15 do not exist, so all mass goes to 2.
        g = _line_graph(4)
        out = graph_gen.sample_longhop_constraints(
            g, 100, np.random.default_rng(1), hop_weights={2: 1.0, 9: 5.0, 15: 5.0})
        assert {c["hops"] for c in out} == {2}
        with pytest.raises(ValueError, match="no hop bucket"):
            graph_gen.sample_longhop_constraints(
                g, 1, np.random.default_rng(1), hop_weights={9: 1.0})

    def test_flag_conflicts_rejected(self):
        g = _line_graph()
        with pytest.raises(ValueError, match="replaces max_boost"):
            graph_gen.sample_longhop_constraints(
                g, 1, np.random.default_rng(0), max_boost=2.0, hop_weights={3: 1.0})
        with pytest.raises(ValueError, match="needs hop_weights"):
            graph_gen.sample_longhop_constraints(
                g, 1, np.random.default_rng(0), object_goal_frac=0.5)

    def test_object_goals_land_on_object_hosts(self):
        hosts = ("g_3_3", "g_0_3", "g_2_1")
        g = _grid_graph(objects=hosts)
        out = graph_gen.sample_longhop_constraints(
            g, 40, np.random.default_rng(3), hop_weights={h: 1.0 for h in range(1, 7)},
            object_goal_frac=0.5)
        flagged = [c for c in out if c["object_goal"]]
        assert len(flagged) == 20
        assert all(c["goal"] in hosts for c in flagged)


class TestLonghopRoles:
    def test_named_starts_avoid_the_robot_region(self):
        g = _line_graph(7)  # robot_location room_0
        cons = [{"init": f"room_{i % 4}", "goal": "room_6", "hops": 6 - i % 4}
                for i in range(20)]
        out = graph_gen.assign_longhop_roles(
            g, cons, np.random.default_rng(0), named_start_frac=0.5, via_frac=0.0,
            max_via_hops=15)
        named = [c for c in out if c["named_start"]]
        assert len(named) == 10
        assert all(c["init"] != "room_0" for c in named)
        assert all(c["via"] is None for c in out)

    def test_line_graph_has_no_waypoint(self):
        # Every off-path region forces a backtrack: the route would not be simple.
        dist = graph_gen._region_distances(_line_graph(7))
        assert graph_gen.via_candidates(dist, "room_0", "room_4") == []

    def test_grid_has_no_waypoint(self):
        # Strict rule: on a grid the two segments' shortest-path rectangles
        # always share a column, so no tie-break-proof waypoint exists.
        dist = graph_gen._region_distances(_grid_graph(4, 4))
        assert graph_gen.via_candidates(dist, "g_0_0", "g_0_3") == []

    def test_via_candidates_off_path_and_simple(self):
        # 8-cycle, cell_0 -> cell_2 (direct 2 hops via cell_1): only cell_5
        # has unique shortest segments meeting only at itself (3 + 3 hops).
        g = _cycle_graph(8)
        nx, G = _nx(g)
        dist = graph_gen._region_distances(g)
        cands = graph_gen.via_candidates(dist, "cell_0", "cell_2")
        assert cands == [("cell_5", 6)]
        for p1 in nx.all_shortest_paths(G, "cell_0", "cell_5"):
            for p2 in nx.all_shortest_paths(G, "cell_5", "cell_2"):
                route = p1 + p2[1:]
                assert len(route) == len(set(route)) == 7

    def test_via_role_rewrites_hops(self):
        g = _cycle_graph(8)
        cons = [{"init": "cell_0", "goal": "cell_2", "hops": 2}] * 6
        out = graph_gen.assign_longhop_roles(
            g, cons, np.random.default_rng(5), named_start_frac=0.0, via_frac=0.5,
            max_via_hops=15)
        vias = [c for c in out if c["via"]]
        assert len(vias) == 3
        assert all(c["via"] == "cell_5" and c["hops"] == 6 and c["direct_hops"] == 2
                   for c in vias)
        # max_via_hops below the constrained length: no waypoint is assigned.
        out = graph_gen.assign_longhop_roles(
            g, cons, np.random.default_rng(5), named_start_frac=0.0, via_frac=0.5,
            max_via_hops=5)
        assert all(c["via"] is None and c["hops"] == 2 for c in out)


def _via_task(init="cell_0", goal="cell_2", via=None, avoid=None,
              text="Give the route to the far cell, with its path and edges."):
    crit = f"A correct answer gives a valid route from {init} to {goal}"
    if via:
        crit += f" via {via}"
    if avoid:
        crit += f" without using {avoid}"
    crit += "."
    mid = rf".*\b{via}\b" if via else ""
    return {"task": text, "answer": rf"(?i)\b{init}\b{mid}.*\b{goal}\b",
            "init_node": init, "acceptance_criterion": crit}


def _role(via=None, named=False, hops=2):
    return {"init": "cell_0", "goal": "cell_2", "hops": hops, "named_start": named, "via": via}


class TestViaValidator:
    g = _cycle_graph(8)

    def test_matching_waypoint_accepted(self):
        graph_gen.validate_longhop_tasks(
            [_via_task(via="cell_5")], [_role(via="cell_5", hops=6)], graph=self.g)

    def test_missing_waypoint_rejected(self):
        with pytest.raises(ValueError, match="grader resolves waypoints"):
            graph_gen.validate_longhop_tasks(
                [_via_task()], [_role(via="cell_5", hops=6)], graph=self.g)

    def test_unrequested_waypoint_rejected(self):
        with pytest.raises(ValueError, match="grader resolves waypoints"):
            graph_gen.validate_longhop_tasks(
                [_via_task(via="cell_5")], [_role()], graph=self.g)

    def test_avoid_on_a_segment_rejected(self):
        # cell_7 is on the cell_0 -> cell_5 segment.
        with pytest.raises(ValueError, match="shortest segment"):
            graph_gen.validate_longhop_tasks(
                [_via_task(via="cell_5", avoid="cell_7")], [_role(via="cell_5", hops=6)],
                graph=self.g)

    def test_avoid_on_the_direct_path_accepted_for_a_waypoint_task(self):
        # cell_1 is on the direct cell_0 -> cell_2 path (the legacy rule would
        # reject it) but off both segments of cell_0 -> cell_5 -> cell_2.
        graph_gen.validate_longhop_tasks(
            [_via_task(via="cell_5", avoid="cell_1")], [_role(via="cell_5", hops=6)],
            graph=self.g)

    def test_named_start_must_not_say_starting_area(self):
        bad = _via_task(text="From the starting area, give the route to the far cell.")
        with pytest.raises(ValueError, match="robot's position"):
            graph_gen.validate_longhop_tasks([bad], [_role(named=True)], graph=self.g)
        good = _via_task(text="From the first cell, give the route to the far cell.")
        graph_gen.validate_longhop_tasks([good], [_role(named=True)], graph=self.g)

    def test_role_constraints_need_the_graph(self):
        with pytest.raises(ValueError, match="need graph="):
            graph_gen.validate_longhop_tasks([_via_task()], [_role()])


class TestE22Prompt:
    def _prompt(self, constraints, base_graph="{}", **kw):
        gen = graph_gen.TaskGraphGen.__new__(graph_gen.TaskGraphGen)
        return gen.build_prompt(base_graph=base_graph, n_tasks=len(constraints) + 2,
                                longhop_constraints=constraints, **kw)

    def test_role_lines(self):
        cons = [dict(_role(via="cell_5", hops=6)),
                dict(_role(named=True), object_goal=True),
                dict(_role(), object_goal=False)]
        p = self._prompt(cons)
        assert p.count("REQUIRED WAYPOINT (optimal route through it: 6 hops)") == 1
        assert p.count("NAMED START: the robot is NOT in this region") == 1
        assert p.count("Phrase the DESTINATION as the area containing") == 1
        assert p.count("(optimal route: 2 hops)") == 2
        assert "never word it as the robot's current position" in p
        assert "or waypoints other than the REQUIRED WAYPOINTS" in p
        p_avoid = self._prompt(cons, longhop_allow_avoid=True)
        assert "except the REQUIRED WAYPOINTS listed above" in p_avoid

    def test_start_area_fraction_counts_unnamed_tasks_only(self):
        # 10 tasks, 4 named starts: 0.5 x 6 unnamed -> 3 "starting area" marks,
        # never on a named-start task.
        cons = [dict(_role(named=k < 4)) for k in range(10)]
        p = self._prompt(cons, longhop_start_area_frac=0.5)
        assert p.count(MARKER) == 3
        for chunk in p.split("- Task ")[1:5]:
            assert MARKER not in chunk

    def test_zone_block_only_with_zones(self):
        cons = [_role()]
        assert "Zone naming" not in self._prompt(cons)
        skel = json.dumps({"graph": {}, "_metadata": {"zones": {"region_1": 0}}})
        p = self._prompt(cons, base_graph=skel)
        assert "Zone naming (OVERRIDES the region-naming rules of Step 2)" in p
        assert "<zone>_<kind>_<k>" in p


class TestParseResponseMarksNamedStarts:
    BASE = {
        "graph": {
            "objects": [],
            "regions": [{"name": f"region_{i}", "coords": [i, 0], "description": ""}
                        for i in range(1, 6)],
            "object_connections": [],
            "region_connections": [[f"region_{i}", f"region_{i + 1}"] for i in range(1, 5)],
            "robot_location": "region_1",
        },
        "tasks": [],
        "_metadata": {"n_communities": 1, "n_tasks": 2},
    }
    RENAME = {f"region_{i}": f"hall_{i}" for i in range(1, 6)}

    def _task(self, init, text):
        return {"task": text, "answer": rf"(?i)\b{init}\b.*\bhall_5\b", "init_node": init,
                "acceptance_criterion": f"A correct answer gives a route from {init} to hall_5."}

    def test_keep_robot_location_set_mechanically(self):
        gen = graph_gen.TaskGraphGen(client=object())
        tasks = [self._task("hall_3", "From the third hall, give the route to the last hall."),
                 self._task("hall_1", "From the starting area, give the route to the last hall.")]
        cons = [{"init": "region_3", "goal": "region_5", "hops": 2, "named_start": True,
                 "via": None, "object_goal": False},
                {"init": "region_1", "goal": "region_5", "hops": 4, "named_start": False,
                 "via": None, "object_goal": False}]
        out = gen.parse_response(
            json.dumps({"rename": self.RENAME, "descriptions": {}, "tasks": tasks}),
            json.dumps(self.BASE), n_tasks=2, longhop_constraints=cons)
        assert out["tasks"][0]["keep_robot_location"] is True
        assert "keep_robot_location" not in out["tasks"][1]
        assert out["graph"]["robot_location"] == "hall_1"
