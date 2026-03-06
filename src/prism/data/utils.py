import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
from openai import OpenAI
from openai.types.chat import ChatCompletion
from scipy.spatial.transform import Rotation
from spine.mapping.graph_util import parse_graph_coord


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
    # Drop all networkx node/edge attributes that are not explicitly set below.
    # from_networkx copies every attr (type, weight, …); graphs with no edges/nodes
    # won't have the edge attrs, causing Batch.from_data_list to fail when mixing
    # empty and non-empty graphs.
    for attr in ("edge_type", "edge_weight", "weight", "type"):
        if hasattr(pyg_graph, attr):
            delattr(pyg_graph, attr)
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


def aggregate(
    root_dir: str, glob_str: str, out_file: str, cutbefore: Optional[int] = 36
) -> None:
    json_files = Path(root_dir).glob(glob_str)

    all_data = []
    for json_file in json_files:
        try:
            data = try_load_json(json_file)

            train_data = data[cutbefore:]

            all_data.append({"conversations": train_data})
        except Exception as ex:
            print(json_file)

    with open(out_file, "w") as f:
        json.dump(all_data, f)


class GPTQueryClient:
    def __init__(self):
        self.client = OpenAI()

    def query_gpt(
        self,
        query: str,
        temperature: Optional[float] = 0.31,
        max_tokens: Optional[int] = 2048,
    ) -> ChatCompletion:
        response = self.client.chat.completions.create(
            model="gpt-4o",
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

        return response
