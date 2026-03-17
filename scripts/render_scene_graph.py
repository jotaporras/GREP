"""Render a scene graph JSON as an interactive HTML with movable nodes.

Uses vis.js Network (loaded from CDN) so the output is a single self-contained
HTML file.  Nodes are pre-positioned from the scene graph's real spatial
coordinates and can be dragged freely.

Node colours:
  • Blue circles  — regions
  • Red diamonds  — objects
  • Gold circle   — robot start location

Click a node to highlight its immediate neighbours; click the background to
reset.  Use the toolbar buttons to zoom-to-fit or toggle physics.

Usage:
    python scripts/render_scene_graph.py --graph data/scene_graph_150.json
    python scripts/render_scene_graph.py --graph data/eval/eval_1_multi_step.json --out my_graph.html
"""

import argparse
import json
import math
import webbrowser
from pathlib import Path


def load_graph(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)
    # Support both bare graph dicts and eval JSON that wraps graph under "graph"
    return data.get("graph", data)


def _scale_coords(nodes_with_coords: list) -> dict:
    """Map raw scene-graph coords to vis.js canvas pixels (y-axis flipped)."""
    xs = [c[0] for _, c in nodes_with_coords]
    ys = [c[1] for _, c in nodes_with_coords]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min, 1)
    CANVAS = 1400
    PADDING = 100
    scale = (CANVAS - 2 * PADDING) / span

    result = {}
    for name, (rx, ry) in nodes_with_coords:
        result[name] = (
            (rx - x_min) * scale + PADDING,
            -(ry - y_min) * scale - PADDING,   # flip Y so north = up
        )
    return result


def _cartesian(c1: list, c2: list) -> float:
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def build_vis_data(graph_dict: dict) -> tuple:
    """Return (vis_nodes_json, vis_edges_json) strings ready to embed in HTML."""
    robot_location = graph_dict.get("robot_location", "")

    # Build a name → raw-coord lookup before scaling
    raw_coord: dict = {}
    raw_coords: list = []
    for node in graph_dict.get("objects", []) + graph_dict.get("regions", []):
        raw_coord[node["name"]] = node["coords"]
        raw_coords.append((node["name"], node["coords"]))

    scaled = _scale_coords(raw_coords)

    vis_nodes = []
    for node in graph_dict.get("regions", []):
        name = node["name"]
        x, y = scaled[name]
        desc = node.get("description", "") or ""
        title = f"{name}" + (f"<br><i>{desc}</i>" if desc else "")
        color = "#FFD700" if name == robot_location else "#4a90d9"
        border = "#b8860b" if name == robot_location else "#2c6fad"
        vis_nodes.append({
            "id": name,
            "label": name,
            "title": title,
            "x": round(x, 1),
            "y": round(y, 1),
            "color": {"background": color, "border": border,
                      "highlight": {"background": "#ffe066", "border": "#b8860b"}},
            "shape": "dot",
            "size": 16 if name != robot_location else 22,
            "font": {"size": 12, "color": "#1a1a2e"},
        })

    for node in graph_dict.get("objects", []):
        name = node["name"]
        x, y = scaled[name]
        desc = node.get("description", "") or ""
        title = f"{name}" + (f"<br><i>{desc}</i>" if desc else "")
        vis_nodes.append({
            "id": name,
            "label": name,
            "title": title,
            "x": round(x, 1),
            "y": round(y, 1),
            "color": {"background": "#e05c5c", "border": "#a32929",
                      "highlight": {"background": "#ff9999", "border": "#a32929"}},
            "shape": "diamond",
            "size": 14,
            "font": {"size": 12, "color": "#1a1a2e"},
        })

    # Pixel distance between two already-scaled nodes (used as physics spring length)
    def pixel_dist(a: str, b: str) -> float:
        ax, ay = scaled[a]
        bx, by = scaled[b]
        return round(math.sqrt((ax - bx) ** 2 + (ay - by) ** 2), 1)

    edge_font = {"size": 10, "color": "#555", "align": "middle"}

    vis_edges = []
    for i, (src, dst) in enumerate(graph_dict.get("region_connections", [])):
        dist = _cartesian(raw_coord[src], raw_coord[dst])
        vis_edges.append({
            "id": f"r{i}", "from": src, "to": dst,
            "label": f"{dist:.1f}",
            "font": edge_font,
            "length": pixel_dist(src, dst),
            "color": {"color": "#4a90d9", "highlight": "#1a5fa8"},
            "width": 1.5,
        })
    for i, (src, dst) in enumerate(graph_dict.get("object_connections", [])):
        dist = _cartesian(raw_coord[src], raw_coord[dst])
        vis_edges.append({
            "id": f"o{i}", "from": src, "to": dst,
            "label": f"{dist:.1f}",
            "font": edge_font,
            "length": pixel_dist(src, dst),
            "color": {"color": "#e05c5c", "highlight": "#a32929"},
            "width": 1.5,
            "dashes": True,
        })

    return json.dumps(vis_nodes), json.dumps(vis_edges)


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/vis-network.min.js"></script>
  <link  href="https://cdn.jsdelivr.net/npm/vis-network@9.1.9/dist/dist/vis-network.min.css" rel="stylesheet">
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: "Segoe UI", system-ui, sans-serif; background: #f0f2f5; }}
    header {{
      display: flex; align-items: center; gap: 16px;
      padding: 12px 20px; background: #1a1a2e; color: #fff;
    }}
    header h1 {{ font-size: 1rem; font-weight: 600; }}
    header .stats {{ font-size: 0.8rem; opacity: 0.7; }}
    .legend {{
      display: flex; gap: 18px; align-items: center;
      margin-left: auto; font-size: 0.8rem;
    }}
    .legend-item {{ display: flex; align-items: center; gap: 6px; }}
    .dot {{ width: 12px; height: 12px; border-radius: 50%; }}
    .diamond {{
      width: 12px; height: 12px;
      transform: rotate(45deg); border-radius: 1px;
    }}
    #toolbar {{
      padding: 8px 20px; background: #fff; border-bottom: 1px solid #dde;
      display: flex; gap: 10px;
    }}
    button {{
      padding: 5px 14px; border: 1px solid #bbb; border-radius: 4px;
      background: #fff; cursor: pointer; font-size: 0.82rem;
    }}
    button:hover {{ background: #eef; border-color: #4a90d9; }}
    #graph-container {{
      width: 100%; height: calc(100vh - 90px);
    }}
  </style>
</head>
<body>
  <header>
    <h1>Scene Graph — {source_file}</h1>
    <span class="stats">{n_nodes} nodes &nbsp;·&nbsp; {n_edges} edges</span>
    <div class="legend">
      <div class="legend-item">
        <div class="dot" style="background:#4a90d9"></div> Region
      </div>
      <div class="legend-item">
        <div class="diamond" style="background:#e05c5c"></div> Object
      </div>
      <div class="legend-item">
        <div class="dot" style="background:#FFD700;border:2px solid #b8860b"></div>
        Robot start ({robot_location})
      </div>
    </div>
  </header>
  <div id="toolbar">
    <button onclick="network.fit()">Fit to view</button>
    <button id="btn-physics">Enable physics</button>
    <button onclick="network.setOptions({{physics:{{enabled:true}}}});physicsOn=true;document.getElementById('btn-physics').textContent='Disable physics';network.stabilize()">Re-stabilise</button>
  </div>
  <div id="graph-container"></div>

  <script>
    const nodesData = {vis_nodes};
    const edgesData = {vis_edges};

    const nodes = new vis.DataSet(nodesData);
    const edges = new vis.DataSet(edgesData);

    const container = document.getElementById("graph-container");
    const options = {{
      physics: {{
        enabled: false,
        stabilization: {{ iterations: 300, fit: true }},
        // spring length per edge overrides this default; kept as fallback
        forceAtlas2Based: {{ springLength: 80, damping: 0.2 }},
        solver: "forceAtlas2Based",
      }},
      interaction: {{
        hover: true,
        tooltipDelay: 100,
        navigationButtons: false,
        keyboard: true,
      }},
      nodes: {{ borderWidth: 1.5 }},
      edges: {{
        smooth: {{ type: "dynamic" }},
        font: {{ size: 10, color: "#555", align: "middle", strokeWidth: 2, strokeColor: "#fff" }},
      }},
    }};

    const network = new vis.Network(container, {{ nodes, edges }}, options);

    // Toggle physics
    let physicsOn = false;
    document.getElementById("btn-physics").addEventListener("click", function() {{
      physicsOn = !physicsOn;
      network.setOptions({{ physics: {{ enabled: physicsOn }} }});
      this.textContent = physicsOn ? "Disable physics" : "Enable physics";
    }});

    // Click-to-highlight neighbours
    let highlighted = false;
    const defaultColors = {{}};
    nodesData.forEach(n => {{ defaultColors[n.id] = n.color; }});

    network.on("click", function(params) {{
      if (params.nodes.length === 0) {{
        // Reset
        const updates = nodesData.map(n => ({{ id: n.id, color: defaultColors[n.id], opacity: 1 }}));
        nodes.update(updates);
        highlighted = false;
        return;
      }}
      const selected = params.nodes[0];
      const neighbours = new Set(network.getConnectedNodes(selected));
      const updates = nodesData.map(n => {{
        if (n.id === selected) return {{ id: n.id, color: defaultColors[n.id], opacity: 1 }};
        if (neighbours.has(n.id)) return {{ id: n.id, color: defaultColors[n.id], opacity: 0.9 }};
        return {{ id: n.id, color: "rgba(180,180,180,0.3)", opacity: 0.3 }};
      }});
      nodes.update(updates);
      highlighted = true;
    }});
  </script>
</body>
</html>
"""


def render_html(graph_dict: dict, source_file: str, out_path: str) -> None:
    robot_location = graph_dict.get("robot_location", "")
    vis_nodes, vis_edges = build_vis_data(graph_dict)
    n_nodes = len(graph_dict.get("objects", [])) + len(graph_dict.get("regions", []))
    n_edges = (len(graph_dict.get("object_connections", [])) +
               len(graph_dict.get("region_connections", [])))

    html = HTML_TEMPLATE.format(
        title=f"Scene Graph — {source_file}",
        source_file=source_file,
        n_nodes=n_nodes,
        n_edges=n_edges,
        robot_location=robot_location or "—",
        vis_nodes=vis_nodes,
        vis_edges=vis_edges,
    )

    Path(out_path).write_text(html, encoding="utf-8")
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Render a scene graph JSON as an interactive HTML file"
    )
    parser.add_argument("--graph", required=True, help="Path to scene graph JSON file")
    parser.add_argument("--out", default="scene_graph.html",
                        help="Output HTML path (default: scene_graph.html)")
    parser.add_argument("--open", action="store_true",
                        help="Open the output file in the default browser after saving")
    args = parser.parse_args()

    graph_dict = load_graph(args.graph)
    source_file = Path(args.graph).name

    n_regions = len(graph_dict.get("regions", []))
    n_objects = len(graph_dict.get("objects", []))
    n_edges   = (len(graph_dict.get("object_connections", [])) +
                 len(graph_dict.get("region_connections", [])))
    print(f"Graph: {n_regions + n_objects} nodes "
          f"({n_regions} regions, {n_objects} objects), {n_edges} edges")
    print(f"Robot start: {graph_dict.get('robot_location', '—')}")

    render_html(graph_dict, source_file, args.out)

    if args.open:
        webbrowser.open(Path(args.out).resolve().as_uri())


if __name__ == "__main__":
    main()
