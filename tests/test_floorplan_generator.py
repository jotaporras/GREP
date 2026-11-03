"""e22 floor-plan skeleton generator (scripts/generate_eval_graphs.py --layout).

Covers docs/2026-11-02 e22_geometric_scale_design.md §2 and §7 step 8:
- BSP floor plans are connected, rooms tile the plan without overlap, and
  every door sits on a shared wall >= MIN_WALL (planar adjacency);
- the mean topology over 20 seeds at 100 regions is inside the §2 band
  (TARGET_BAND_100), with door_extra / corridor_bias drawn as the CLI does;
- the CLI writes layout / zones metadata and community_assignments == zones;
- --layout sbm (the default) is byte-identical to the pre-e22 script for
  seeds 201-203 (md5s of files generated from the untouched script).
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
nx = pytest.importorskip("networkx")
pytest.importorskip("scipy")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_eval_graphs.py"
sys.path.insert(0, str(SCRIPT.parent))

import generate_eval_graphs as geg  # noqa: E402


def _cli_draw(seed, n_regions):
    """main()'s floorplan draw order: door_extra first, then the plan."""
    rng = np.random.default_rng(seed)
    door_extra = float(rng.uniform(0.05, 0.20))
    return geg.generate_floorplan_graph(n_regions, door_extra, 0.3, 3, 25.0, rng)


class TestFloorplanGeometry:
    @pytest.mark.parametrize("n_regions", [40, 70, 100, 130])
    def test_connected_and_sized(self, n_regions):
        for seed in range(5):
            G, coords, zones = _cli_draw(seed, n_regions)
            assert G.number_of_nodes() == n_regions == len(coords) == len(zones)
            assert nx.is_connected(G)

    def test_rooms_tile_the_plan(self):
        rooms = geg.bsp_rooms(100, 0.3, 3, np.random.default_rng(0))
        assert len(rooms) == 100
        assert all(r["w"] > 0 and r["h"] > 0 for r in rooms)
        assert sum(r["w"] * r["h"] for r in rooms) == pytest.approx(1.0 / geg.PLAN_ASPECT)
        for i, a in enumerate(rooms):
            for b in rooms[i + 1:]:
                ox = min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"])
                oy = min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"])
                assert ox <= 1e-9 or oy <= 1e-9

    def test_doors_are_shared_walls(self):
        # Same seed: the generator's first plan IS bsp_rooms' output when the
        # candidate graph was connected on the first try (centroids confirm).
        for seed in range(10):
            rooms = geg.bsp_rooms(100, 0.3, 3, np.random.default_rng(seed))
            G, coords, _ = geg.generate_floorplan_graph(
                100, 0.1, 0.3, 3, 25.0, np.random.default_rng(seed))
            scale = float(np.sqrt(25.0 * 100 * geg.PLAN_ASPECT))
            assert coords[0] == [round((rooms[0]["x"] + rooms[0]["w"] / 2) * scale, 1),
                                 round((rooms[0]["y"] + rooms[0]["h"] / 2) * scale, 1)]
            for u, v in G.edges:
                assert geg._shared_wall(rooms[u], rooms[v]) >= geg.MIN_WALL

    def test_zones_follow_the_split_tree(self):
        _, _, zones = _cli_draw(0, 100)
        # 2**zone_depth = 8 subtrees, plus one zone per corridor carved above
        # that depth.
        assert 8 <= len(set(zones.values())) <= 16
        assert sorted(set(zones.values())) == list(range(len(set(zones.values()))))


class TestTargetBand:
    def test_mean_statistics_in_band_at_100_regions(self):
        stats = [geg.layout_statistics(_cli_draw(seed, 100)[0]) for seed in range(20)]
        for key, (lo, hi) in geg.TARGET_BAND_100.items():
            mean = float(np.mean([s[key] for s in stats]))
            assert lo <= mean <= hi, f"{key} mean {mean:.3f} outside [{lo}, {hi}]"

    def test_emst_fallback_connected(self):
        for seed in range(5):
            G, coords = geg.generate_emst_graph(100, 0.1, 25.0, np.random.default_rng(seed))
            assert G.number_of_nodes() == 100 == len(coords)
            assert nx.is_connected(G)


def _run(tmp_path, name, *args):
    out = tmp_path / f"{name}.json"
    subprocess.run([sys.executable, str(SCRIPT), *args, "--output", str(out)],
                   check=True, capture_output=True)
    return out


class TestCli:
    def test_floorplan_metadata(self, tmp_path):
        out = _run(tmp_path, "fp", "--layout", "floorplan", "--n-regions", "40",
                   "--robot-location", "random", "--seed", "7")
        skel = json.loads(out.read_text())
        meta, graph = skel["_metadata"], skel["graph"]
        assert meta["layout"] == "floorplan" and meta["n_regions"] == 40
        assert 0.05 <= meta["door_extra"] <= 0.20
        names = {r["name"] for r in graph["regions"]}
        assert set(meta["zones"]) == names
        assert meta["community_assignments"] == meta["zones"]
        assert graph["robot_location"] in names
        G = nx.Graph()
        G.add_nodes_from(names)
        G.add_edges_from(map(tuple, graph["region_connections"]))
        assert nx.is_connected(G)

    # md5 of the pre-e22 script's output (generated before any edit).
    SBM_BASELINE = {
        ("default", 201): "167d8fe69b96f86f8b9a2c2b77a389d5",
        ("default", 202): "7d1715b1667f4fe32d68abf491102750",
        ("default", 203): "72afa1608b97ced49041900f5406d752",
        ("n60", 201): "5d1030e25320e77c45e18debd64956e9",
        ("n60", 202): "f289cdd3a14125a937e0f0a1bbf0c395",
        ("n60", 203): "63ca0a192cac0152257c8a9f625c31dd",
    }
    N60 = ("--n-communities", "6", "--nodes-per-community", "8",
           "--object-rate", "0.3", "--n-tasks", "18")

    @pytest.mark.parametrize("config,seed", sorted(SBM_BASELINE))
    def test_sbm_byte_identical(self, tmp_path, config, seed):
        extra = self.N60 if config == "n60" else ()
        out = _run(tmp_path, f"{config}_{seed}", "--seed", str(seed), *extra)
        assert hashlib.md5(out.read_bytes()).hexdigest() == self.SBM_BASELINE[(config, seed)]
