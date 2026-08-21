"""Scoring contract of scripts/neighbour_probe.py (pure Python; no model)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import neighbour_probe as probe

GRAPH = {
    "regions": [{"name": n} for n in ("sub_dock_1", "sub_dock_2", "bridge_1", "lab_1")],
    "region_connections": [["sub_dock_1", "bridge_1"], ["bridge_1", "lab_1"],
                           ["sub_dock_2", "lab_1"]],
}


def test_neighbours_symmetric():
    adj = probe.neighbours(GRAPH)
    assert adj["bridge_1"] == {"sub_dock_1", "lab_1"}
    assert adj["sub_dock_2"] == {"lab_1"}


def test_named_regions_ordered_by_first_occurrence_and_excludes_self():
    regions = [r["name"] for r in GRAPH["regions"]]
    text = "From bridge_1 go to lab_1, then sub_dock_1; bridge_1 again."
    assert probe.named_regions(text, regions, exclude="bridge_1") == ["lab_1", "sub_dock_1"]
    # word boundary: sub_dock_1 must not match inside sub_dock_12
    assert probe.named_regions("sub_dock_12", regions, exclude="x") == []


def test_score_sibling_vs_hallucination():
    regions = [r["name"] for r in GRAPH["regions"]]
    truth = {"sub_dock_1", "lab_1"}                      # neighbours of bridge_1
    s = probe.score(["sub_dock_2", "lab_1"], truth, regions)
    assert s["first_ok"] is False and s["exact"] is False
    assert s["sibling_err"] == 1 and s["hallucinated"] == 0
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    assert s["missed"] == ["sub_dock_1"] and s["wrong"] == ["sub_dock_2"]
    s2 = probe.score(["lab_1", "sub_dock_1"], truth, regions)
    assert s2["first_ok"] and s2["exact"] and s2["precision"] == 1.0
    s3 = probe.score([], truth, regions)
    assert s3["first_ok"] is False and s3["precision"] == 0.0 and s3["recall"] == 0.0


def test_plan_text_prefers_plan_field():
    assert probe.plan_text('{"plan": "lab_1, sub_dock_1"}') == "lab_1, sub_dock_1"
    assert probe.plan_text("not json at all") == "not json at all"


def test_aggregate_means():
    recs = [probe.score(["lab_1"], {"lab_1", "bridge_1"}, []),
            probe.score(["bridge_1", "lab_1"], {"lab_1", "bridge_1"}, [])]
    a = probe.aggregate(recs)
    assert a["n_queries"] == 2 and a["exact"] == 0.5 and a["first_ok"] == 1.0
    assert abs(a["recall"] - 0.75) < 1e-9
