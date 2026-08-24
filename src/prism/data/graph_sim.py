import os
from copy import deepcopy
from typing import Optional

import numpy as np
from spine.mapping.graph_util import GraphHandler
from spine.spine_util import UpdatePromptFormer


def _goto_ratify_disabled() -> bool:
    """Kill-switch for goto path ratification (e19 SPINE closed loop).

    Ratification is ON by default: a goto over an edge that does not exist in
    the ground-truth graph is rejected with corrective feedback instead of
    silently teleporting the agent. Set PRISM_GOTO_RATIFY=0 to restore the
    legacy (non-validating) behavior for A/B comparisons.
    """
    return os.environ.get("PRISM_GOTO_RATIFY", "1").strip().lower() in (
        "0", "false", "no", "off")


class GraphSim:
    def __init__(self, graph: GraphHandler):
        """Initialize simulation from a ground-truth graph.

        Deep copies the ground-truth graph to create a partial (observed) graph,
        then strips all descriptions to simulate an agent that hasn't explored yet.
        """
        self.graph = graph

        for node in self.graph.graph.nodes:
            if "description" not in self.graph.graph.nodes[node]:
                self.graph.graph.nodes[node]["description"] = "no description"

        self.partial_graph = deepcopy(graph)
        self.init_partial_graph()

    def init_partial_graph(self):
        """Strip descriptions from the partial graph and reset tracking state.

        Removes all node descriptions from the partial graph (agent starts with no
        knowledge of contents) and initializes the UpdatePromptFormer that tracks
        what the agent has discovered since the last query.
        """
        for node in self.partial_graph.graph.nodes:
            self.partial_graph.graph.nodes[node].pop("description", 0)

        self.have_updates = False
        self.removed_nodes = []
        self.action_history = []
        self.updator = UpdatePromptFormer()
        self.have_updates = False
        self.partial_graph.as_json_str = self.partial_graph.to_json_str()

    def reset(self, **args):
        self.graph.reset(**args)
        self.partial_graph = deepcopy(self.graph)
        self.init_partial_graph()

    def randomly_remove_nodes(
        self, *, pct: float = 0, n_nodes: float = 0, to_remove=[]
    ):
        """Remove nodes from the partial graph to simulate unexplored areas.

        Difficulty can be set by percentage of nodes (pct), exact count (n_nodes),
        or an explicit list (to_remove). The agent's current location is never removed.
        """
        all_nodes = list(self.partial_graph.graph.nodes)

        # first check if we should randomly remove
        if pct > 0:
            assert n_nodes == 0 and len(to_remove) == 0
            n_to_remove = (len(all_nodes) * pct) // 100
            n_nodes = n_to_remove
        elif n_nodes > 0:
            assert len(to_remove) == 0

        # then check to remove specific nodes
        if len(to_remove) > 0:
            assert n_nodes == 0 and pct == 0
        else:
            to_remove = list(np.random.choice(all_nodes, n_nodes, replace=False))

        # make sure we don't remove current location
        if self.partial_graph.current_location in to_remove:
            to_remove.remove(self.partial_graph.current_location)

        for node in to_remove:
            self.partial_graph.remove_node(node)

        self.removed_nodes.extend(list(to_remove))

        self.partial_graph.as_json_str = self.partial_graph.to_json_str()

    def corrupt_with_fake_edges(self, n_edges: int, rng,
                                preferred_pairs=None) -> list:
        """Add plausible-but-fake shortcut edges to the OBSERVED graph only.

        Samples region pairs at ground-truth distance exactly 2 that have no
        direct edge and adds them to the partial graph, so the agent's map
        contains shortcuts that do not exist while goto ratification still
        checks the ground truth. This mirrors the dominant navigator failure
        mode (hallucinated 2-hop shortcut edges), making the planner walk into
        rejections and produce recovery turns. Returns the fake edges added.

        ``preferred_pairs`` (optional list of (u, v)) are used FIRST after
        validation (both nodes exist, no true edge): pass shortcut pairs that
        lie ON the task's ground-truth route so the planner is actually
        tempted to take the bait — random 2-hop shortcuts rarely intersect
        the planned route (smoke 7826590: 1 rejection in 11 corrupted
        rollouts). Any remaining budget is filled from the random pool.
        """
        regions = [n for n, d in self.graph.graph.nodes(data=True)
                   if d.get("type") == "region"]
        picks = []
        for u, v in preferred_pairs or []:
            pair = tuple(sorted((u, v)))
            if (pair not in picks and u in self.graph.graph.nodes
                    and v in self.graph.graph.nodes
                    and not self.graph.graph.has_edge(u, v)):
                picks.append(pair)
            if len(picks) >= n_edges:
                break
        if len(picks) < n_edges:
            candidates = set()
            for u in regions:
                for mid in self.graph.get_neighbors(u):
                    for v in self.graph.get_neighbors(mid):
                        if (v != u and v in regions
                                and not self.graph.graph.has_edge(u, v)):
                            candidates.add(tuple(sorted((u, v))))
            candidates = sorted(candidates - set(picks))
            if candidates:
                picks += [candidates[i] for i in rng.choice(
                    len(candidates),
                    size=min(n_edges - len(picks), len(candidates)),
                    replace=False)]
        if not picks:
            return []
        for u, v in picks:
            cu = np.array(self.graph.graph.nodes[u]["coords"], dtype=float)
            cv = np.array(self.graph.graph.nodes[v]["coords"], dtype=float)
            self.partial_graph.graph.add_edge(
                u, v, type="region", weight=float(np.linalg.norm(cu - cv)))
        self.partial_graph.as_json_str = self.partial_graph.to_json_str()
        return picks

    def get_updator(self) -> UpdatePromptFormer:
        return self.updator

    def add_new_node(self, source_node, target, debug: Optional[bool] = True):
        if debug:
            print(f"discovered missing node: {target}")
        node_info = self.graph.graph.nodes[target]

        node_info["name"] = target
        self.updator.update(new_nodes=[{target: node_info}])
        edge_info = self.graph.graph.get_edge_data(target, source_node)

        # only add type and coorindates
        self.partial_graph.graph.add_node(
            target, type=node_info["type"], coords=node_info["coords"]
        )
        self.partial_graph.graph.add_edge(target, source_node, **edge_info)

        self.removed_nodes.remove(target)
        self.have_updates = True

    def add_edges(self, source: str, target: str):
        self.updator.update(new_connections=[[source, target]])
        edge_info = self.graph.graph.get_edge_data(source, target)
        self.partial_graph.graph.add_edge(source, target, **edge_info)
        self.have_updates = True

    def _reveal_region(self, current_location: str) -> None:
        """Reveal neighbors and description of `current_location` in the partial graph."""
        neighbors = self.graph.get_neighbors(current_location)

        for n in neighbors:
            if n not in self.partial_graph.graph.nodes:
                self.add_new_node(current_location, n)

            gt_edges = [
                sorted(e)
                for e in list(self.partial_graph.get_edges(current_location).keys())
            ]

            query_edge = sorted((current_location, n))
            if query_edge not in gt_edges:
                self.add_edges(query_edge[0], query_edge[1])

        assert (
            "description" in self.graph.graph.nodes[current_location]
        ), f"{current_location} has no description"
        description = self.graph.graph.nodes[current_location]["description"]
        self.partial_graph.graph.nodes[current_location]["description"] = description
        self.updator.update(
            attribute_updates=[{"name": current_location, "description": description}]
        )

    def take_action(self, action, argument) -> bool:
        """Execute a planner action against the ground-truth graph.

        Handles explore/map (reveals neighbors and descriptions), inspect (reveals
        object description), goto (updates agent location), and extend_map. Reveals
        discovered information in the partial graph and records diffs in the updator.
        Returns True if new information was added to the partial graph.
        """
        if action == "map_region":
            current_location = argument
            self._reveal_region(current_location)

        elif action == "explore_region":
            current_location, radius = argument
            self._reveal_region(current_location)

        # TODO incomplete
        elif action == "extend_map":
            self.updator.update(
                freeform_updates=["Do not call extend_map. Try explore_region instead"]
            )

            # current_location = self.partial_graph.current_location
            # line =  np.array(argument) - self.partial_graph.graph.nodes[current_location]["coords"]
            # debug = 0

            # # see what nodes are close
            # # TODO try and implement exploration
            # for n in self.removed_nodes:
            #     distance_thresh = 10
            #     print(np.linalg.norm(self.graph.get_node_coord(current_location) - self.graph.get_node_coord(n)))
            #     if np.linalg.norm(self.graph.get_node_coord(current_location) - self.graph.get_node_coord(n)) < 25:
            #         self.add_new_node(source_node=current_location, target=n)

        elif action == "inspect":
            target = argument[0]
            description = self.graph.graph.nodes[target]["description"]
            self.partial_graph.graph.nodes[target]["description"] = description
            self.updator.update(
                attribute_updates=[{"name": target, "description": description}]
            )
            self.have_updates = True

        elif action == "goto":
            # SPINE semantics (per the ICL prompt): goto(target) runs a graph
            # search over the agent's OBSERVED map and the robot follows that
            # route until an edge turns out not to exist in the true graph.
            # Success reports the traversed route (grounded pathfinding is the
            # tool's contribution); failure stops the robot at the last node
            # actually reached, retracts the bogus edge from the observed map,
            # and interrupts the plan so the planner replans from there.
            location = argument
            if _goto_ratify_disabled():
                self.updator.update(location_updates=[location])
                self.partial_graph.update_location(location)
            else:
                import networkx as nx

                here = self.partial_graph.current_location
                if location == here:
                    pass  # already there — silent no-op, plan continues
                elif location not in self.graph.graph.nodes:
                    self.updator.update(freeform_updates=[
                        f"goto({location}) rejected: no region named "
                        f"{location} exists. You are still at {here}. Replan "
                        f"using only regions and edges that exist."])
                    self.have_updates = True
                else:
                    try:
                        bpath = nx.shortest_path(
                            self.partial_graph.graph, here, location)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        bpath = None
                        self.updator.update(freeform_updates=[
                            f"goto({location}) rejected: your map has no "
                            f"route from {here} to {location}. You are still "
                            f"at {here}."])
                        self.have_updates = True
                    if bpath is not None:
                        reached, failed = bpath[0], None
                        for a, b in zip(bpath, bpath[1:]):
                            if self.graph.graph.has_edge(a, b):
                                reached = b
                            else:
                                failed = (a, b)
                                break
                        if failed is None:
                            self.partial_graph.update_location(location)
                            self.updator.update(location_updates=[location])
                            if len(bpath) > 2:
                                self.updator.update(freeform_updates=[
                                    f"goto({location}): traversed "
                                    + " -> ".join(bpath) + "."])
                        else:
                            a, b = failed
                            # Correct the observed map: the believed edge
                            # (e.g. a corrupted/hallucinated shortcut) is
                            # retracted with the update grammar the prompts
                            # teach.
                            self.partial_graph.graph.remove_edge(a, b)
                            self.updator.update(removed_connections=[[a, b]])
                            self.partial_graph.update_location(reached)
                            self.updator.update(location_updates=[reached])
                            self.updator.update(freeform_updates=[
                                f"goto({location}) failed en route: there is "
                                f"no edge between {a} and {b}. You are now at "
                                f"{reached}. Replan from {reached} using only "
                                f"edges that exist."])
                            self.have_updates = True

        # Valid routes are goto-chains, so the repeat-action nag (meant for
        # explore spam) would fire on every step of a normal route once
        # ratification can interrupt plans; exempt goto while ratifying.
        nag_exempt = action == "goto" and not _goto_ratify_disabled()
        if (not nag_exempt and len(self.action_history) > 0
                and action in self.action_history[-3:]):
            self.updator.update(
                freeform_updates=[
                    f"You are calling {action} multiple times in a row, which will not help you solve the task. Try calling something else. If you have tried all options, answer the user with your results."
                ]
            )
        self.action_history.append(action)

        return self.have_updates

    def __str__(self):
        out = ""

        out += f"full graph\n---\nn_nodes: {len(self.graph.graph.nodes)}\nn_edges: {len(self.graph.graph.edges)}"
        out += f"\n===\npartial graph\n---\nn_nodes: {len(self.partial_graph.graph.nodes)}\nn_edges: {len(self.partial_graph.graph.edges)}"

        return out
