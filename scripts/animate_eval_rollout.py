"""Animate eval rollouts over their scene graphs as a neon, video-game-style path player.

This is the animated sibling of ``render_scene_graph.py``.  It keeps that script's
vis.js background scaffold (nodes pre-positioned from real spatial coordinates,
draggable, self-contained single-file HTML) and layers a canvas overlay that plays
back a model's navigation rollout one edge at a time:

  • Green neon edge   — the model's hop exists in the graph (a real connection).
  • Red dashed edge   — a hallucinated hop; a dashed red line animatedly *extends*
                        from the current node out to the node the model chose.
  • Green node        — the start / robot location.
  • Cyan nodes        — nodes the walk has visited so far.
  • Yellow node       — the intended goal.
  • Blue node         — the last node of the model's path (where it ended up).

A video-style transport bar (restart · play/pause · scrubber · speed) drives the
animation; edges are drawn progressively as the scrubber advances.

It consumes:
  • Rollouts in the ``results/e13f_transferability`` format — a JSON file with a
    ``samples`` list, each carrying ``path_metrics.parsed_nodes`` and
    ``path_metrics.goal``.  Pass a single file or a whole directory of them.
  • Graphs in the ``data/n_100/gen/nav_n100_gemma_data/test_graphs`` format,
    matched to each rollout by filename stem (``data_gen_001`` -> ``data_gen_001.json``).

Usage:
    python scripts/animate_eval_rollout.py \
        --rollouts results/e13f_transferability \
        --graphs-dir data/n_100/gen/nav_n100_gemma_data/test_graphs --open

    python scripts/animate_eval_rollout.py \
        --rollouts results/e13f_transferability/data_gen_001.json --out rollout.html --open
"""

import argparse
import heapq
import json
import math
import random
import webbrowser
from pathlib import Path

# REUSE the coordinate / IO primitives from the static renderer rather than
# re-implementing them (PRIME DIRECTIVE: extend + compose existing code).
try:
    from render_scene_graph import load_graph, _scale_coords, _cartesian
except ImportError:  # allow ``python -m scripts.animate_eval_rollout`` too
    from scripts.render_scene_graph import load_graph, _scale_coords, _cartesian


# --------------------------------------------------------------------------- #
# Graph -> neon background scaffold (mirrors build_vis_data, restyled dark)     #
# --------------------------------------------------------------------------- #
def _declump(scaled: dict, min_sep: float, iterations: int = 220,
             anchor: float = 0.015) -> dict:
    """Spread nodes that sit closer than ``min_sep`` while pinning each near its
    original position, so tightly-clustered rooms/objects become visible without
    collapsing the separation between distant regions.

    Only pairs closer than ``min_sep`` repel (distant wings never interact), and a
    weak spring to each node's anchor keeps every cluster centred where it was.
    """
    names = list(scaled.keys())
    pos = {n: [scaled[n][0], scaled[n][1]] for n in names}
    # de-coincide exactly-overlapping nodes deterministically so r > 0
    for i, n in enumerate(names):
        pos[n][0] += 0.01 * math.cos(i * 2.399)
        pos[n][1] += 0.01 * math.sin(i * 2.399)

    cutoff2 = (3 * min_sep) ** 2
    for _ in range(iterations):
        disp = {n: [0.0, 0.0] for n in names}
        for a in range(len(names)):
            xa, ya = pos[names[a]]
            for b in range(a + 1, len(names)):
                xb, yb = pos[names[b]]
                dx, dy = xa - xb, ya - yb
                d2 = dx * dx + dy * dy
                if d2 > cutoff2 or d2 == 0.0:
                    continue
                d = math.sqrt(d2)
                if d < min_sep:
                    push = 0.5 * (min_sep - d)
                    ux, uy = dx / d, dy / d
                    disp[names[a]][0] += ux * push
                    disp[names[a]][1] += uy * push
                    disp[names[b]][0] -= ux * push
                    disp[names[b]][1] -= uy * push
        for n in names:
            ax, ay = scaled[n]
            pos[n][0] += disp[n][0] + anchor * (ax - pos[n][0])
            pos[n][1] += disp[n][1] + anchor * (ay - pos[n][1])
    return {n: (pos[n][0], pos[n][1]) for n in names}


def build_neon_base(graph_dict: dict, min_sep: float = 52.0) -> tuple:
    """Return (vis_nodes, vis_edges) lists styled as a dim neon scaffold.

    Nodes are placed at the same scaled spatial coordinates the static renderer
    uses (then locally de-clumped so co-located rooms/objects are legible); the
    overlay draws all the bright role colouring on top, so the base is kept faint.
    """
    raw_coords = [
        (node["name"], node["coords"])
        for node in graph_dict.get("objects", []) + graph_dict.get("regions", [])
    ]
    scaled = _scale_coords(raw_coords)
    if min_sep > 0 and len(scaled) > 1:
        scaled = _declump(scaled, min_sep)

    nodes = []
    for node in graph_dict.get("regions", []):
        name = node["name"]
        x, y = scaled[name]
        nodes.append({
            "id": name, "label": name, "x": round(x, 1), "y": round(y, 1),
            "shape": "dot", "size": 7,
            "color": {"background": "#242a44", "border": "#39406b"},
            "font": {"size": 9, "color": "#5b6488"},
        })
    for node in graph_dict.get("objects", []):
        name = node["name"]
        x, y = scaled[name]
        nodes.append({
            "id": name, "label": name, "x": round(x, 1), "y": round(y, 1),
            "shape": "diamond", "size": 6,
            "color": {"background": "#3a2440", "border": "#5b3960"},
            "font": {"size": 8, "color": "#6b5b7a"},
        })

    edges = []
    for i, (src, dst) in enumerate(graph_dict.get("region_connections", [])):
        edges.append({"id": f"r{i}", "from": src, "to": dst,
                      "color": {"color": "rgba(120,130,180,0.10)"}, "width": 0.6})
    for i, (src, dst) in enumerate(graph_dict.get("object_connections", [])):
        edges.append({"id": f"o{i}", "from": src, "to": dst,
                      "color": {"color": "rgba(150,110,170,0.10)"}, "width": 0.6,
                      "dashes": True})
    return nodes, edges


# Per-task metrics not worth showing: hallucination_rate ≡ 1−edge_validity_rate;
# parsed_nodes is the animated path; goal is shown in the panel header.
_METRIC_BLOCKLIST = {"hallucination_rate", "parsed_nodes", "goal"}


def build_adjacency(graph_dict: dict) -> set:
    """Undirected edge set for classifying a hop as real vs. hallucinated."""
    adj = set()
    for src, dst in (graph_dict.get("region_connections", []) +
                     graph_dict.get("object_connections", [])):
        adj.add((src, dst))
        adj.add((dst, src))
    return adj


def _weighted_adjacency(graph_dict: dict, coord: dict) -> dict:
    """Undirected adjacency ``node -> [(neighbour, cartesian_distance), ...]``.

    Edge weights are the real spatial (node) distances between endpoints, so the
    Dijkstra search below optimises travelled distance rather than hop count.
    """
    adjw: dict = {}
    for u, v in (graph_dict.get("region_connections", []) +
                 graph_dict.get("object_connections", [])):
        if u in coord and v in coord:
            w = _cartesian(coord[u], coord[v])
            adjw.setdefault(u, []).append((v, w))
            adjw.setdefault(v, []).append((u, w))
    return adjw


def _dijkstra(adjw: dict, start: str, goal: str, rng: random.Random) -> tuple:
    """Shortest start→goal path by node distance; returns ``(cost, [nodes])``.

    When several equal-cost shortest paths exist, one is sampled uniformly at
    random by choosing randomly among tied predecessors during reconstruction.
    Returns ``(inf, [])`` if the goal is unreachable.
    """
    INF = float("inf")
    if not start or not goal or start not in adjw or goal not in adjw:
        return (0.0, [start]) if start and start == goal else (INF, [])
    if start == goal:
        return (0.0, [start])

    dist = {start: 0.0}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF):
            continue
        for v, w in adjw.get(u, []):
            nd = d + w
            if nd < dist.get(v, INF) - 1e-12:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    if goal not in dist:
        return (INF, [])

    # Walk back from goal, randomly breaking ties among optimal predecessors.
    path = [goal]
    cur = goal
    for _ in range(len(dist) + 1):
        if cur == start:
            break
        preds = [u for u, w in adjw.get(cur, [])
                 if u in dist and abs(dist[u] + w - dist[cur]) < 1e-9]
        if not preds:
            return (INF, [])
        cur = rng.choice(preds)
        path.append(cur)
    path.reverse()
    return (dist[goal], path) if path and path[0] == start else (INF, [])


def build_episode(eval_data: dict, graph_dict: dict, name: str,
                  min_sep: float = 52.0) -> dict:
    """Bundle one rollout file + its graph into an animatable episode."""
    nodes, edges = build_neon_base(graph_dict, min_sep=min_sep)
    adj = build_adjacency(graph_dict)
    node_names = {n["id"] for n in nodes}
    coord = {n["name"]: n["coords"]
             for n in graph_dict.get("objects", []) + graph_dict.get("regions", [])}
    adjw = _weighted_adjacency(graph_dict, coord)
    robot = graph_dict.get("robot_location", "")

    def path_cost(p: list) -> float:
        return sum(_cartesian(coord[a], coord[b])
                   for a, b in zip(p, p[1:]) if a in coord and b in coord)

    samples = []
    for s in eval_data.get("samples", []):
        pm = s.get("path_metrics") or {}
        path = pm.get("parsed_nodes") or []
        segments = [
            {"from": a, "to": b, "valid": (a, b) in adj}
            for a, b in zip(path, path[1:])
        ]
        start = robot or (path[0] if path else "")
        goal = pm.get("goal", "")

        # Optimal start→goal route over node distances (random tie-break, stable
        # per graph+idx so the same HTML regenerates identically).
        rng = random.Random(f"{name}:{s.get('idx')}")
        opt_cost, dpath = _dijkstra(adjw, start, goal, rng)

        # Cost optimality replaces the (redundant) hallucination rate: the model
        # path is at best as short as Dijkstra's, so optimal/actual ∈ (0, 1].
        model_cost = path_cost(path)
        cost_opt = (min(1.0, opt_cost / model_cost)
                    if opt_cost != float("inf") and model_cost > 1e-9 else None)

        # Full per-task diagnostics, minus keys that are redundant or shown
        # elsewhere (hallucination_rate ≡ 1−edge_validity; parsed_nodes ≡ path;
        # goal shown in the header). Fill the (usually null) cost_optimality with
        # our geometric value.
        metrics = {k: v for k, v in pm.items()
                   if k not in _METRIC_BLOCKLIST}
        if metrics.get("cost_optimality") is None and cost_opt is not None:
            metrics["cost_optimality"] = cost_opt

        # Reasoning chain + final answer for the right-hand drop-down.
        resp = s.get("response") or {}
        answer = ""
        for act in resp.get("plan") or []:
            if isinstance(act, (list, tuple)) and len(act) >= 2 and act[0] == "answer":
                answer = act[1]

        samples.append({
            "idx": s.get("idx"),
            "task": s.get("task", ""),
            "goal": goal,
            "start": path[0] if path else start,
            "path": path,
            "segments": segments,
            "dijkstra": dpath,
            "correct": bool(s.get("correct")),
            "metrics": metrics,
            "reasoning": resp.get("reasoning", ""),
            "answer": answer,
            "answer_key": s.get("answer_key", ""),
            # any parsed node absent from the graph is a ghost the overlay places
            "missing": sorted(n for n in path if n not in node_names),
        })

    return {
        "name": name,
        "nodes": nodes,
        "edges": edges,
        "samples": samples,
        "robot_location": graph_dict.get("robot_location", ""),
        "accuracy": eval_data.get("accuracy"),
        "architecture": eval_data.get("architecture", ""),
    }


def _is_multigraph_log(eval_data: dict) -> bool:
    """A training-run eval log (e.g. results/eval_logs/step_*.json) bundles many
    graphs in one file: samples carry differing ``graph_name`` and a ``per_graph``
    summary is present.  A per-graph rollout file has one graph for the whole file.
    """
    if "per_graph" in eval_data:
        return True
    names = {s.get("graph_name") for s in eval_data.get("samples", [])}
    names.discard(None)
    return len(names) > 1


def _episodes_from_multigraph(eval_data: dict, gdir: Path, source: str,
                              min_sep: float) -> list:
    """Split a multi-graph eval log into one episode per ``graph_name``, matching
    each group to ``<graphs_dir>/<graph_name>.json`` and reusing build_episode."""
    per_graph = eval_data.get("per_graph", {})
    grouped: dict = {}
    for s in eval_data.get("samples", []):
        grouped.setdefault(s.get("graph_name"), []).append(s)

    episodes = []
    for name in sorted(k for k in grouped if k):
        graph_path = gdir / f"{name}.json"
        if not graph_path.exists():
            print(f"  ! no graph for {name} at {graph_path} — skipping")
            continue
        graph_dict = load_graph(str(graph_path))
        acc = (per_graph.get(name) or {}).get("accuracy", eval_data.get("accuracy"))
        # Reuse build_episode via a per-graph view of the log (only its samples).
        sub = {"samples": grouped[name], "accuracy": acc,
               "architecture": eval_data.get("architecture", "")}
        episodes.append(build_episode(sub, graph_dict, name, min_sep=min_sep))
        print(f"  · {source} → {name}: {len(grouped[name])} tasks")
    return episodes


def collect_episodes(rollouts: str, graphs_dir: str, min_sep: float = 52.0) -> list:
    """Build animatable episodes from either per-graph rollout files (matched by
    filename stem) or multi-graph training eval logs (split by ``graph_name``)."""
    rp = Path(rollouts)
    files = sorted(rp.glob("*.json")) if rp.is_dir() else [rp]
    gdir = Path(graphs_dir)

    episodes = []
    for f in files:
        eval_data = json.loads(f.read_text())

        # Skip non-rollout files (e.g. run summaries with no per-task samples).
        if not eval_data.get("samples"):
            continue

        # Multi-graph eval log: one file → one episode per graph_name.
        if _is_multigraph_log(eval_data):
            episodes.extend(
                _episodes_from_multigraph(eval_data, gdir, f.name, min_sep))
            continue

        # Per-graph rollout file: one file → one episode. Prefer the samples'
        # own graph_name (filenames may carry run-id prefixes like
        # "e13f_alpha00_binary_..._data_gen_001"); fall back to the file stem.
        samples = eval_data.get("samples", [])
        gnames = {s.get("graph_name") for s in samples if s.get("graph_name")}
        gname = gnames.pop() if len(gnames) == 1 else f.stem
        graph_path = gdir / f"{gname}.json"
        if not graph_path.exists():
            print(f"  ! no graph for {f.name} ({gname}) at {graph_path} — skipping")
            continue
        graph_dict = load_graph(str(graph_path))
        episodes.append(build_episode(eval_data, graph_dict, gname, min_sep=min_sep))
        print(f"  · {gname}: {len(samples)} tasks")
    return episodes


# --------------------------------------------------------------------------- #
# HTML template.  Injection points are unique tokens (not str.format) so the   #
# large JS/CSS body can keep its literal braces.                               #
# --------------------------------------------------------------------------- #
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>__TITLE__</title>
  <script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; }
    body {
      font-family: "Segoe UI", system-ui, sans-serif;
      background: #06060f; color: #cdd3f0; overflow: hidden;
    }
    header {
      display: flex; align-items: center; gap: 18px;
      padding: 10px 18px; background: linear-gradient(90deg,#0b0b1e,#141433);
      border-bottom: 1px solid #23234a;
    }
    header h1 {
      font-size: 0.95rem; font-weight: 700; letter-spacing: .5px;
      color: #7cf3ff; text-shadow: 0 0 10px rgba(57,255,190,.4);
    }
    header select {
      background: #12122b; color: #cdd3f0; border: 1px solid #2c2c58;
      border-radius: 5px; padding: 4px 8px; font-size: .8rem;
    }
    header .spacer { margin-left: auto; }
    .badge {
      font-size: .72rem; font-weight: 700; padding: 3px 9px; border-radius: 999px;
      text-transform: uppercase; letter-spacing: .5px;
    }
    .badge.ok  { background: rgba(57,255,20,.15); color: #7dff6b; border: 1px solid #2f7d24; }
    .badge.bad { background: rgba(255,45,85,.15); color: #ff6b86; border: 1px solid #7d2436; }
    #stage { position: relative; width: 100%; height: calc(100vh - 52px); }
    #graph { width: 100%; height: 100%; }
    #panel {
      position: absolute; top: 14px; left: 14px; width: 300px; max-width: 30vw;
      max-height: calc(100vh - 90px); overflow-y: auto;
      background: rgba(10,10,26,.82); border: 1px solid #26264d; border-radius: 10px;
      padding: 14px 16px; backdrop-filter: blur(6px);
      box-shadow: 0 8px 30px rgba(0,0,0,.5);
    }
    #panel .task { font-size: 1rem; line-height: 1.45; color: #e6e9ff; margin-bottom: 10px; }
    #panel .row { display: flex; gap: 10px; font-size: .74rem; margin: 4px 0; color: #9aa2cf; }
    #panel .row b { color: #cdd3f0; font-weight: 600; }
    .section-title { font-size: .78rem; text-transform: uppercase; letter-spacing: .6px;
      color: #7cf3ff; margin: 12px 0 6px; opacity: .8; }
    #p-metrics .mrow { display: flex; justify-content: space-between; gap: 12px;
      font-size: .88rem; padding: 5px 0; border-bottom: 1px solid rgba(255,255,255,.05); }
    #p-metrics .mrow span { color: #8a92c0; }
    #p-metrics .mrow b { color: #cdd3f0; font-weight: 600; font-variant-numeric: tabular-nums;
      text-align: right; word-break: break-word; }
    /* right-hand reasoning-chain drop-down */
    #reasoning {
      position: absolute; top: 14px; right: 14px; width: 350px; max-width: 32vw;
      max-height: calc(100vh - 90px); overflow-y: auto;
      background: rgba(10,10,26,.86); border: 1px solid #26264d; border-radius: 10px;
      backdrop-filter: blur(6px); box-shadow: 0 8px 30px rgba(0,0,0,.5);
    }
    #reasoning details > summary {
      cursor: pointer; list-style: none; user-select: none;
      padding: 12px 16px; font-weight: 700; font-size: .96rem; letter-spacing: .3px; color: #9cf6ff;
    }
    #reasoning details > summary::-webkit-details-marker { display: none; }
    #reasoning details > summary::before { content: "▸ "; color: #39ffbe; }
    #reasoning details[open] > summary::before { content: "▾ "; }
    #r-reasoning { padding: 0 16px 14px; font-size: .92rem; line-height: 1.55;
      color: #d4d9f7; white-space: pre-wrap; }
    #answerbox { margin: 0 12px 14px; border: 1px solid #2c2c58; border-radius: 8px;
      background: rgba(20,20,45,.55); }
    #answerbox > summary { color: #ffe14a; padding: 9px 12px; font-size: .9rem; }
    #answerbox > summary::before { color: #ffe14a; }
    #r-answer { padding: 0 12px 12px; font-size: .9rem; line-height: 1.5; color: #eef0ff; white-space: pre-wrap; }
    .answer-key { display: block; margin-top: 8px; font-size: .78rem; color: #8a92c0; word-break: break-all; }
    .legend { margin-top: 12px; display: grid; grid-template-columns: 1fr 1fr; gap: 6px 12px; font-size: .72rem; }
    .legend .li { display: flex; align-items: center; gap: 7px; color: #aab1de; }
    .swatch { width: 14px; height: 14px; border-radius: 50%; flex: none; box-shadow: 0 0 8px currentColor; }
    .swatch.line { width: 18px; height: 3px; border-radius: 2px; }
    .swatch.dash { background: repeating-linear-gradient(90deg,#ff2d55 0 5px,transparent 5px 9px); box-shadow: none; }
    #transport {
      position: absolute; left: 50%; bottom: 18px; transform: translateX(-50%);
      display: flex; align-items: center; gap: 12px; width: min(760px, 92vw);
      background: rgba(10,10,26,.9); border: 1px solid #26264d; border-radius: 999px;
      padding: 8px 16px; box-shadow: 0 8px 30px rgba(0,0,0,.55);
    }
    #transport button {
      background: #16163a; color: #9cf6ff; border: 1px solid #2c2c58; cursor: pointer;
      width: 34px; height: 34px; border-radius: 50%; font-size: 1rem; line-height: 1;
      display: flex; align-items: center; justify-content: center; transition: .15s;
    }
    #transport button:hover { background: #22224e; border-color: #39ffbe; color: #39ffbe; }
    #transport #play { width: 42px; height: 42px; font-size: 1.15rem; }
    #scrub { flex: 1; accent-color: #39ffbe; height: 4px; cursor: pointer; }
    #time { font-size: .74rem; color: #9aa2cf; min-width: 92px; text-align: right; font-variant-numeric: tabular-nums; }
    #transport select {
      background: #12122b; color: #cdd3f0; border: 1px solid #2c2c58;
      border-radius: 5px; padding: 3px 6px; font-size: .74rem;
    }
    .hint { font-size: .66rem; color: #5b6488; margin-top: 8px; }
  </style>
</head>
<body>
  <header>
    <h1>▶ __TITLE__</h1>
    <label style="font-size:.75rem;color:#8a92c0">Graph
      <select id="sel-ep"></select>
    </label>
    <label style="font-size:.75rem;color:#8a92c0">Task
      <select id="sel-sample"></select>
    </label>
    <span class="spacer"></span>
    <span id="verdict" class="badge">—</span>
    <span class="stats" style="font-size:.75rem;color:#8a92c0">__SUBTITLE__</span>
  </header>

  <div id="stage">
    <div id="graph"></div>
    <div id="panel">
      <div class="task" id="p-task">—</div>
      <div class="section-title">How the model did</div>
      <div id="p-metrics"></div>
      <div class="legend">
        <div class="li" style="color:#39ff14"><span class="swatch line" style="background:#39ff14"></span>Valid hop</div>
        <div class="li" style="color:#ff2d55"><span class="swatch dash line"></span>Hallucinated</div>
        <div class="li" style="color:#b26bff"><span class="swatch line" style="background:#b26bff"></span>Optimal (Dijkstra)</div>
        <div class="li" style="color:#39ff14"><span class="swatch" style="background:#39ff14"></span>Start</div>
        <div class="li" style="color:#00e5ff"><span class="swatch" style="background:#00e5ff"></span>Visited</div>
        <div class="li" style="color:#ffe14a"><span class="swatch" style="background:#ffe14a"></span>Goal</div>
        <div class="li" style="color:#3aa0ff"><span class="swatch" style="background:#3aa0ff"></span>Last node</div>
      </div>
      <div class="hint">Drag nodes to rearrange · scroll to zoom · scrub or press play to draw the route.</div>
    </div>
    <div id="reasoning">
      <details id="chain" open>
        <summary>Reasoning chain</summary>
        <div id="r-reasoning">—</div>
        <details id="answerbox">
          <summary>Answer</summary>
          <div id="r-answer">—</div>
        </details>
      </details>
    </div>
  </div>

  <div id="transport">
    <button id="prev" title="Previous task">⏮</button>
    <button id="restart" title="Restart">⟲</button>
    <button id="play" title="Play / pause">▶</button>
    <button id="next" title="Next task">⏭</button>
    <input id="scrub" type="range" min="0" max="0" step="0.001" value="0">
    <span id="time">0 / 0 hops</span>
    <select id="speed" title="Speed">
      <option value="0.5">0.5×</option>
      <option value="1" selected>1×</option>
      <option value="2">2×</option>
      <option value="4">4×</option>
    </select>
  </div>

  <script>
    const EPISODES = /*__EPISODES__*/[];

    // ---- palette -----------------------------------------------------------
    const C = {
      valid: "#39ff14", validCore: "#d6ffcb",
      hall: "#ff2d55",  hallCore: "#ffd0da",
      optimal: "#b26bff", optimalCore: "#e7d4ff",
      start: "#39ff14", visited: "#00e5ff", goal: "#ffe14a", last: "#3aa0ff",
      traveller: "#ffffff",
    };

    // ---- state -------------------------------------------------------------
    let epIdx = 0, sampleIdx = 0;
    let network = null, ghost = {}, flowPhase = 0;
    let segProgress = 0, playing = false, speedMul = 1, lastTs = null;
    const BASE_SPEED = 1.3;  // hops per second at 1×

    const $ = id => document.getElementById(id);
    const container = $("graph");
    const options = {
      physics: { enabled: false },
      interaction: { hover: true, tooltipDelay: 120, keyboard: false, dragNodes: true },
      nodes: { borderWidth: 1 },
      edges: { smooth: false },   // straight lines so the overlay tracks base geometry
    };

    const ep = () => EPISODES[epIdx];
    const smp = () => ep().samples[sampleIdx];
    const numSegs = () => smp().segments.length;

    // ---- overlay drawing ---------------------------------------------------
    function neonLine(ctx, a, b, color, core, dashed, width) {
      ctx.save();
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      if (dashed) { ctx.setLineDash([12, 10]); ctx.lineDashOffset = -flowPhase * 45; }
      ctx.shadowColor = color; ctx.shadowBlur = 18;
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.globalAlpha = 0.4;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      ctx.shadowBlur = 8; ctx.lineWidth = width * 0.42; ctx.globalAlpha = 1;
      ctx.strokeStyle = core;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      ctx.restore();
    }

    function halo(ctx, p, color, radius, intensity) {
      const pulse = 1 + 0.12 * Math.sin(flowPhase * 3);
      ctx.save();
      ctx.shadowColor = color; ctx.shadowBlur = 26 * intensity;
      ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.globalAlpha = 0.9 * intensity;
      ctx.beginPath(); ctx.arc(p.x, p.y, radius * pulse, 0, 2 * Math.PI); ctx.stroke();
      ctx.globalAlpha = 0.22 * intensity; ctx.fillStyle = color; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(p.x, p.y, radius * 0.62, 0, 2 * Math.PI); ctx.fill();
      ctx.restore();
    }

    function tag(ctx, p, text, color) {
      ctx.save();
      ctx.font = "700 13px 'Segoe UI', sans-serif";
      ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.shadowColor = "#000"; ctx.shadowBlur = 5;
      ctx.fillStyle = color; ctx.fillText(text, p.x, p.y - 12);
      ctx.restore();
    }

    function drawOverlay(ctx) {
      const s = smp(), pos = network.getPositions();
      const gp = name => pos[name] || ghost[name];
      const reached = Math.floor(segProgress + 1e-6);

      // 0) optimal (Dijkstra) route in purple — a persistent reference beneath
      //    the animated model path, so the model can be compared against it.
      const dj = s.dijkstra || [];
      for (let i = 0; i < dj.length - 1; i++) {
        const a = gp(dj[i]), b = gp(dj[i + 1]);
        if (a && b) neonLine(ctx, a, b, C.optimal, C.optimalCore, false, 4);
      }

      // 1) path edges up to the scrubber
      let tip = null;
      for (let i = 0; i < s.segments.length; i++) {
        if (segProgress <= i) break;
        const seg = s.segments[i], a = gp(seg.from), b = gp(seg.to);
        if (!a || !b) continue;
        const t = Math.min(1, segProgress - i);
        const cur = { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
        neonLine(ctx, a, cur, seg.valid ? C.valid : C.hall,
                 seg.valid ? C.validCore : C.hallCore, !seg.valid, 6);
        tip = cur;
      }

      // 2) visited node halos (start green, else cyan; hallucinated target red)
      for (let j = 0; j <= reached && j < s.path.length; j++) {
        const p = gp(s.path[j]); if (!p) continue;
        let col = C.visited;
        if (j === 0) col = C.start;
        else if (!s.segments[j - 1].valid) col = C.hall;
        halo(ctx, p, col, 11, 1);
      }

      // 3) goal (yellow) and last node (blue) always shown; concentric if same
      const goalP = gp(s.goal);
      if (goalP) halo(ctx, goalP, C.goal, 16, 1);
      const lastName = s.path[s.path.length - 1];
      const lastP = lastName ? gp(lastName) : null;
      if (lastP) halo(ctx, lastP, C.last, 10, segProgress >= numSegs() ? 1 : 0.5);

      // 4) labels for the relevant nodes
      for (let j = 0; j <= reached && j < s.path.length; j++) {
        const p = gp(s.path[j]); if (p) tag(ctx, p, s.path[j], "#dfe4ff");
      }
      if (goalP) tag(ctx, goalP, s.goal + " (goal)", C.goal);
      if (lastP && segProgress >= numSegs()) tag(ctx, lastP, lastName, C.last);

      // 5) traveller pulse at the leading tip
      const tipNode = tip || (s.path.length ? gp(s.path[0]) : null);
      if (tipNode) {
        const r = 6 + 2 * Math.sin(flowPhase * 6);
        ctx.save();
        ctx.shadowColor = C.traveller; ctx.shadowBlur = 30;
        ctx.fillStyle = C.traveller; ctx.globalAlpha = 0.95;
        ctx.beginPath(); ctx.arc(tipNode.x, tipNode.y, r, 0, 2 * Math.PI); ctx.fill();
        ctx.restore();
      }
    }

    // ---- ghost placement for nodes absent from the graph -------------------
    function computeGhosts() {
      ghost = {};
      const pos = network.getPositions();
      const s = smp();
      let ang = 0.6;
      s.segments.forEach(seg => {
        if (!pos[seg.to] && !ghost[seg.to]) {
          const f = pos[seg.from] || ghost[seg.from] || { x: 0, y: 0 };
          ghost[seg.to] = { x: f.x + 130 * Math.cos(ang), y: f.y + 130 * Math.sin(ang) };
          ang += 1.3;
        }
      });
    }

    // ---- building / selection ---------------------------------------------
    function buildNetwork() {
      if (network) network.destroy();
      const e = ep();
      network = new vis.Network(container,
        { nodes: new vis.DataSet(e.nodes), edges: new vis.DataSet(e.edges) }, options);
      network.on("afterDrawing", drawOverlay);
      network.fit();
    }

    function selectSample(i) {
      sampleIdx = i;
      const s = smp();
      segProgress = 0; playing = false; $("play").textContent = "▶";
      computeGhosts();
      $("scrub").max = Math.max(numSegs(), 0.001);
      $("scrub").value = 0;
      $("p-task").textContent = s.task || "(no task text)";
      renderMetrics(s.metrics || {});
      // reasoning chain + answer (answer stays collapsed by default each task)
      $("r-reasoning").textContent = s.reasoning || "(no reasoning recorded)";
      $("r-answer").innerHTML = "";
      $("r-answer").append(document.createTextNode(s.answer || "(no answer)"));
      if (s.answer_key) {
        const k = document.createElement("span");
        k.className = "answer-key"; k.textContent = "answer_key: " + s.answer_key;
        $("r-answer").append(k);
      }
      $("answerbox").open = false;
      const v = $("verdict");
      v.textContent = s.correct ? "correct" : "incorrect";
      v.className = "badge " + (s.correct ? "ok" : "bad");
      updateTime();
      network.redraw();
    }

    // Plain-language names for each diagnostic, in the order they should read.
    const METRIC_LABELS = {
      num_parsed: "Route steps",
      nodes_exist_rate: "Locations exist",
      edge_validity_rate: "Valid connections",
      full_path_valid: "Route valid",
      start_goal_ok: "Start & goal",
      cost_optimality: "Cost optimality",
      hop_optimality: "Length optimality",
      path_from_reasoning: "From reasoning",
      path_rescued: "Rescued",
      kind: "Task type",
      waypoints_ok: "Stops visited",
      avoid_ok: "Avoided no-go areas",
      required_edges: "Required links",
      required_edges_present: "Required links used",
      structured_correct: "Correct (cleaned)",
      structured: "Well-formed",
      judge_used: "AI-judged",
      llm_judge_pass: "Judge verdict",
      path_expected: "Route expected",
      valid_path_ab: "Reaches goal",
    };
    const METRIC_ORDER = Object.keys(METRIC_LABELS);
    const prettyKey = k => METRIC_LABELS[k] ||
      (k.charAt(0).toUpperCase() + k.slice(1).replace(/_/g, " "));

    // Presentation-friendly value: percentages, Yes/No, node pairs as "a ↔ b".
    function fmtMetric(k, v) {
      if (v === null || v === undefined) return ["—", "#7c84ad"];
      if (typeof v === "boolean") return [v ? "Yes" : "No", v ? "#7dff6b" : "#ff6b86"];
      if (typeof v === "number") {
        if (/optimality$/.test(k)) return [Number(v).toPrecision(3), "#cdd3f0"];
        if (/_rate$/.test(k)) return [(100 * v).toFixed(0) + "%", "#cdd3f0"];
        return [Number.isInteger(v) ? String(v) : v.toFixed(2), "#cdd3f0"];
      }
      if (Array.isArray(v)) {
        if (!v.length) return ["none", "#7c84ad"];
        return [v.map(e => Array.isArray(e) ? e.join(" ↔ ") : String(e)).join(", "), "#aab1de"];
      }
      const s = String(v);
      return [s.charAt(0).toUpperCase() + s.slice(1), "#cdd3f0"];
    }

    function renderMetrics(m) {
      const el = $("p-metrics"); el.innerHTML = "";
      const keys = [...METRIC_ORDER.filter(k => k in m),
                    ...Object.keys(m).filter(k => !(k in METRIC_LABELS))];
      keys.forEach(k => {
        const [txt, col] = fmtMetric(k, m[k]);
        const row = document.createElement("div"); row.className = "mrow";
        const a = document.createElement("span"); a.textContent = prettyKey(k);
        const b = document.createElement("b"); b.textContent = txt; b.style.color = col;
        row.append(a, b); el.append(row);
      });
    }

    function selectEpisode(i) {
      epIdx = i; sampleIdx = 0;
      buildNetwork();
      const sel = $("sel-sample");
      sel.innerHTML = "";
      ep().samples.forEach((s, k) => {
        const o = document.createElement("option");
        o.value = k;
        o.textContent = `#${k} ${s.correct ? "✓" : "✗"} → ${s.goal || "?"}`;
        sel.appendChild(o);
      });
      sel.value = 0;
      selectSample(0);
    }

    function updateTime() {
      const done = Math.min(numSegs(), segProgress);
      $("time").textContent = `${done.toFixed(1)} / ${numSegs()} hops`;
    }

    // ---- transport ---------------------------------------------------------
    function setProgress(v) {
      segProgress = Math.max(0, Math.min(numSegs(), v));
      $("scrub").value = segProgress;
      updateTime();
    }

    $("play").addEventListener("click", () => {
      if (segProgress >= numSegs()) segProgress = 0;
      playing = !playing;
      $("play").textContent = playing ? "⏸" : "▶";
      lastTs = null;
    });
    $("restart").addEventListener("click", () => { setProgress(0); playing = false; $("play").textContent = "▶"; });
    $("scrub").addEventListener("input", e => { playing = false; $("play").textContent = "▶"; setProgress(parseFloat(e.target.value)); });
    $("speed").addEventListener("change", e => { speedMul = parseFloat(e.target.value); });
    $("sel-ep").addEventListener("change", e => selectEpisode(parseInt(e.target.value)));
    $("sel-sample").addEventListener("change", e => selectSample(parseInt(e.target.value)));

    // Step ±1 task, wrapping across graph (episode) boundaries.
    function stepTask(delta) {
      let e2 = epIdx, i2 = sampleIdx + delta;
      while (i2 < 0) { e2 = (e2 - 1 + EPISODES.length) % EPISODES.length; i2 += EPISODES[e2].samples.length; }
      while (i2 >= EPISODES[e2].samples.length) { i2 -= EPISODES[e2].samples.length; e2 = (e2 + 1) % EPISODES.length; }
      if (e2 !== epIdx) { $("sel-ep").value = e2; selectEpisode(e2); }
      $("sel-sample").value = i2;
      selectSample(i2);
    }
    $("prev").addEventListener("click", () => stepTask(-1));
    $("next").addEventListener("click", () => stepTask(1));

    // ---- main loop (always running: drives breathing + playback) -----------
    function loop(ts) {
      flowPhase = ts / 1000;
      if (playing) {
        if (lastTs == null) lastTs = ts;
        const dt = (ts - lastTs) / 1000; lastTs = ts;
        setProgress(segProgress + dt * BASE_SPEED * speedMul);
        if (segProgress >= numSegs()) { playing = false; $("play").textContent = "▶"; }
      } else {
        lastTs = null;
      }
      if (network) network.redraw();
      requestAnimationFrame(loop);
    }

    // ---- init --------------------------------------------------------------
    (function init() {
      const selEp = $("sel-ep");
      EPISODES.forEach((e, k) => {
        const o = document.createElement("option");
        o.value = k; o.textContent = e.name + (e.accuracy != null ? `  (${(100 * e.accuracy).toFixed(0)}%)` : "");
        selEp.appendChild(o);
      });
      if (!EPISODES.length) { document.body.innerHTML = "<p style='padding:2rem'>No episodes loaded.</p>"; return; }
      selectEpisode(0);
      requestAnimationFrame(loop);
    })();
  </script>
</body>
</html>
"""


def render_html(episodes: list, out_path: str, title: str) -> None:
    n_tasks = sum(len(e["samples"]) for e in episodes)
    subtitle = f"{len(episodes)} graphs · {n_tasks} tasks"
    html = (HTML_TEMPLATE
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle)
            .replace("/*__EPISODES__*/[]", json.dumps(episodes)))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Saved → {out_path}  ({len(episodes)} graphs, {n_tasks} tasks)")


def main():
    parser = argparse.ArgumentParser(
        description="Animate eval rollouts over their scene graphs (neon path player)")
    parser.add_argument("--rollouts", required=True,
                        help="Rollout JSON file or directory (results/e13f_transferability format)")
    parser.add_argument("--graphs-dir",
                        default="data/n_100/gen/nav_n100_gemma_data/test_graphs",
                        help="Directory of matching scene-graph JSONs (matched by filename stem)")
    parser.add_argument("--out", default="rollout_animation.html",
                        help="Output HTML path (default: rollout_animation.html)")
    parser.add_argument("--title", default="Rollout Navigator",
                        help="Title shown in the header")
    parser.add_argument("--min-sep", type=float, default=52.0,
                        help="Minimum on-canvas separation (px) enforced between "
                             "co-located nodes; 0 disables de-clumping (default: 52)")
    parser.add_argument("--open", action="store_true",
                        help="Open the output file in the default browser after saving")
    args = parser.parse_args()

    print(f"Loading rollouts from {args.rollouts}")
    episodes = collect_episodes(args.rollouts, args.graphs_dir, min_sep=args.min_sep)
    if not episodes:
        raise SystemExit("No episodes could be built — check --rollouts / --graphs-dir.")

    render_html(episodes, args.out, args.title)
    if args.open:
        webbrowser.open(Path(args.out).resolve().as_uri())


if __name__ == "__main__":
    main()
