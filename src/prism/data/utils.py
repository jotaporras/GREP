import io
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as pyg_utils
from openai import OpenAI
from scipy.spatial.transform import Rotation
from spine.mapping.graph_util import parse_graph_coord
from torch_geometric.data import Data


def safe_parse_graph(
    data: Dict[str, Dict[str, str]],
    custom_data: Optional[Dict[str, Dict[str, str]]] = {},
    rotation: Optional[Rotation] = None,
    utm_origin: Optional[np.ndarray] = None,
    flip_coords=False,
) -> Tuple[nx.Graph, str]:
    """Parse scene graph in `data` into a networkx object.

    Parameters
    ----------
    data : Dict[str, Dict[str, str]]
        graph where keys-values are nodes-attributes
    rotation : Optional[Rotation]
        current rotation of robot

    Returns
    -------
    Tuple[nx.Graph, str]
        Networkx and string of json
    """
    origin = np.array([0, 0])
    data = deepcopy(data)  # don't modify input data
    as_str = str(data)

    if utm_origin is not None:
        origin = utm_origin

    if len(custom_data):
        add_keys = ["regions", "region_connections", "objects", "object_connections"]
        for key in add_keys:
            if key in data and key in custom_data:
                data[key].extend(custom_data[key])

    G = nx.Graph()
    for node in data["objects"]:
        coords = parse_graph_coord(node["coords"], origin=origin, rotation=rotation)
        if flip_coords:
            raise ValueError()
            # print("flipping coords")
            coords = [coords[0], -coords[1]]

        node.pop("coords")
        name = node.pop("name")
        G.add_node(name, coords=coords, type="object", **node)

    for node in data["regions"]:
        assert "coords" in node, node
        c = node["coords"]
        # print(f"node: {node}, coords: {c}")
        coords = parse_graph_coord(node["coords"], origin=origin, rotation=rotation)

        if flip_coords:
            raise ValueError
            # print("flipping coords")
            coords = [coords[0], -coords[1]]
        node.pop("coords")
        name = node.pop("name")
        G.add_node(name, coords=coords, type="object", **node)

    for edge in data["object_connections"]:
        if edge[0] in G.nodes and edge[1] in G.nodes:
            c1 = G.nodes[edge[0]]["coords"]
            c2 = G.nodes[edge[1]]["coords"]
            # print(f"edge: {edge}, c1, c2: {c1}, {c2}")
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            G.add_edge(edge[0], edge[1], type="object", weight=dist)

    for edge in data["region_connections"]:
        if edge[0] in G.nodes and edge[1] in G.nodes:
            c1 = G.nodes[edge[0]]["coords"]
            c2 = G.nodes[edge[1]]["coords"]
            # print(f"edge: {edge}, c1, c2: {c1}, {c2}")
            dist = np.linalg.norm(np.array(c1) - np.array(c2))
            G.add_edge(edge[0], edge[1], type="region", weight=dist)

    return G, as_str


def scene_graph_dict_to_pyg(scene_graph_dict: dict) -> Data:
    """Convert a scene graph dict to a PyG Data object.

    Parameters
    ----------
    scene_graph_dict : dict
        Scene graph with the following expected keys:

        - ``"objects"`` : list of dicts, each with at minimum:
            - ``"name"`` (str) — unique node identifier
            - ``"coords"`` (list[float]) — 2-D or 3-D spatial coordinates
            - any additional attributes are preserved as node features
        - ``"regions"`` : list of dicts, same schema as ``"objects"``
        - ``"object_connections"`` : list of 2-element lists ``[name_a, name_b]``
          representing undirected edges between object nodes
        - ``"region_connections"`` : list of 2-element lists ``[name_a, name_b]``
          representing undirected edges between region nodes
        - ``"robot_location"`` (optional) : any value indicating where the robot
          is in the scene; stored verbatim on the returned graph

    Returns
    -------
    Data
        PyG Data object with attributes:
        ``coords``, ``x``, ``edge_index``, ``node_names``, ``node_types``,
        ``robot_location``, and ``raw_scene_graph``.
    """
    nx_graph, _ = safe_parse_graph(scene_graph_dict)
    node_names = list(nx_graph.nodes)
    coords = torch.tensor(
        [nx_graph.nodes[n]["coords"] for n in node_names], dtype=torch.float32
    )
    pyg_graph = pyg_utils.from_networkx(nx_graph)
    # safe_parse_graph stores the raw Euclidean edge distance (meters) under the
    # networkx "weight" attribute, which from_networkx copies to pyg_graph.weight
    # (one entry per directed edge). Grab it before the cleanup below.
    raw_distance = getattr(pyg_graph, "weight", None)
    # Drop all networkx node/edge attributes that are not explicitly set below.
    # from_networkx copies every attr (type, weight, …); graphs with no edges/nodes
    # won't have the edge attrs, causing Batch.from_data_list to fail when mixing
    # empty and non-empty graphs.
    for attr in ("edge_type", "edge_weight", "weight", "type"):
        if hasattr(pyg_graph, attr):
            delattr(pyg_graph, attr)
    # Convert raw distance to the E1 Gaussian heat-kernel affinity
    # exp(-d^2 / 2σ^2), σ = per-graph median distance, and keep it as
    # `edge_weight` so TAGConv couples closer nodes more strongly. Raw meters are
    # retained under `distance_m` for the path validator (M10). Edgeless graphs
    # get no edge_weight, so downstream forwards stay unweighted (and Batch
    # mixing keeps working).
    if raw_distance is not None and raw_distance.numel() > 0:
        d = raw_distance.float()
        sigma = d.median()
        pyg_graph.edge_weight = (
            torch.exp(-(d ** 2) / (2.0 * sigma ** 2)) if sigma > 0 else torch.ones_like(d)
        )
        pyg_graph.distance_m = d
    pyg_graph.coords = coords
    pyg_graph.x = torch.zeros((coords.size(0), 1), dtype=torch.float32)
    pyg_graph.node_names = node_names
    pyg_graph.node_types = [nx_graph.nodes[n]["type"] for n in node_names]
    pyg_graph.robot_location = scene_graph_dict.get("robot_location")
    pyg_graph.raw_scene_graph = scene_graph_dict
    return pyg_graph


def try_load_json(file):
    with open(file) as f:
        content = f.read()

    # Not sure why but some files have trailing strings
    # after the json entry
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return json.loads("".join(content.split("\n")[:-4]))


def strip_icl(msgs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Strip the few-shot ICL prefix from a logged SPINE rollout.

    A logged rollout is ``[system] + ICL example turns + real task turns``.
    Every ICL example *and* the real task start with a ``user`` message whose
    content begins with ``task:``; the real task is the *last* such message.
    Returns ``[system] + msgs[real_task_idx:]``, keeping the system turn.

    Replaces the old hardcoded ``cutbefore=36`` slice, which silently corrupted
    training data whenever the ICL example set changed length.
    """
    if not (isinstance(msgs, list) and msgs):
        raise ValueError("rollout must be a non-empty list of messages")
    task_idx = next(
        (
            i
            for i in range(len(msgs) - 1, -1, -1)
            if msgs[i].get("role") == "user"
            and msgs[i].get("content", "").lstrip().lower().startswith("task:")
        ),
        None,
    )
    if task_idx is None:
        raise ValueError("no `task:` user message found in rollout")
    if not any(m.get("role") == "assistant" for m in msgs[task_idx + 1 :]):
        raise ValueError("no assistant turn after the real task")
    head = [msgs[0]] if msgs[0].get("role") == "system" else []
    return head + msgs[task_idx:]


def aggregate(
    root_dir: str, glob_str: str, out_file: str, strip_icl_prefix: bool = True
) -> None:
    """Concatenate rollout JSON files under ``root_dir`` into one training file.

    With ``strip_icl_prefix`` (default), each rollout has its few-shot ICL
    prefix removed via :func:`strip_icl`. Pass ``strip_icl_prefix=False`` for
    non-SPINE rollouts that have no ``task:``-prefixed ICL turns.
    """
    json_files = Path(root_dir).glob(glob_str)

    all_data = []
    for json_file in json_files:
        try:
            data = try_load_json(json_file)
            conversations = strip_icl(data) if strip_icl_prefix else data
            all_data.append({"conversations": conversations})
        except Exception as ex:
            print(f"{json_file}: {ex}")

    with open(out_file, "w") as f:
        json.dump(all_data, f)


class GPTQueryClient:
    def __init__(self):
        self.client = OpenAI()

    def query_gpt(
        self,
        query: str,
        temperature: Optional[float] = 0.31,
        max_tokens: Optional[int] = 10240,
        reasoning_effort: str = "low",
    ):
        return self.query_gpt_5(query, temperature, max_tokens, reasoning_effort)

    def query_gpt_5(
        self,
        query: str,
        temperature: Optional[float] = 0.31,
        max_tokens: Optional[int] = 10240,
        reasoning_effort: str = "low",
    ) -> str:

        response = self.client.responses.create(
            model="gpt-5.1",
            input=[
                {"role": "user", "content": [{"type": "input_text", "text": query}]}
            ],
            text={"format": {"type": "text"}, "verbosity": "low"},
            reasoning={"effort": reasoning_effort, "summary": "auto"},
        )

        return response.output_text

    def batch_query_gpt_5(
        self,
        queries: List[str],
        model: str = "gpt-5.1",
        reasoning_effort: str = "low",
        poll_interval: int = 60,
    ) -> List[str]:
        """Submit queries via the Batch API and return responses in order (~50% cheaper)."""
        requests_jsonl = "\n".join(
            json.dumps({
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": model,
                    "input": [{"role": "user", "content": [{"type": "input_text", "text": q}]}],
                    "text": {"format": {"type": "text"}, "verbosity": "low"},
                    "reasoning": {"effort": reasoning_effort, "summary": "auto"},
                },
            })
            for i, q in enumerate(queries)
        )

        file_obj = self.client.files.create(
            file=("batch.jsonl", io.BytesIO(requests_jsonl.encode()), "application/jsonl"),
            purpose="batch",
        )
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/responses",
            completion_window="24h",
        )
        print(f"Batch submitted: {batch.id}")

        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(poll_interval)
            batch = self.client.batches.retrieve(batch.id)
            c = batch.request_counts
            print(f"Batch {batch.id}: {batch.status} ({c.completed}/{c.total})")

        if batch.status != "completed":
            raise RuntimeError(f"Batch {batch.id} ended with status: {batch.status}")

        result_lines = self.client.files.content(batch.output_file_id).text.splitlines()
        responses = {}
        for line in filter(None, result_lines):
            record = json.loads(line)
            responses[int(record["custom_id"])] = record["response"]["body"]["output_text"]
        return [responses[i] for i in range(len(queries))]

    def query_gpt_4(
        self,
        query: str,
        temperature: Optional[float] = 0.31,
        max_tokens: Optional[int] = 10240,
    ) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": query}],
                }
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            max_completion_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        return response.choices[0].message.content
