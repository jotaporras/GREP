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
