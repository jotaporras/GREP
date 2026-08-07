import json
import os
import re
import sys
from typing import List, Optional

from prism.data import local_llm, utils, vllm_llm

FILL_TOKEN = "__FILL__"

# Skeleton names the LLM must replace (region_1, object_12, ...).
PLACEHOLDER_NAME = re.compile(r"^(region|object)_\d+$")

# `type_N` node ids as they appear inside task answers/criteria (hub_1,
# cell_block_3, ...). Used to catch tasks that reference a node the rename map
# never produced.
NODE_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)*_\d+\b")


def _validate_tasks(json_content: dict) -> None:
    """Best-effort post-generation viability gate.

    Reuses the deterministic smoke test (``scripts/smoke_test_graph_solvable``)
    to flag, at generation time, any task that is unsolvable (entity missing /
    unreachable / non-traversable route) or whose ``answer`` regex leaks (rewards
    a wrong-polarity or premise-echo answer). Failures are logged, not raised, so
    a validation hiccup never aborts a generation run; the leaky/narrow regex is
    later canonicalised by ``scripts/fix_answer_regexes.py``.
    """
    try:
        scripts_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "scripts",
        )
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from smoke_test_graph_solvable import build_graph, check_task

        G, nodes, _ = build_graph(json_content["graph"])
        for i, task in enumerate(json_content.get("tasks", [])):
            r = check_task(G, nodes, task)
            if r["verdict"] == "FAIL":
                print(
                    f"[graph-gen][VALIDATION FAIL] task {i}: {r.get('note', '')} "
                    f"| {task.get('task', '')[:80]}"
                )
    except Exception as e:  # never let validation break generation
        print(f"[graph-gen] task validation skipped ({type(e).__name__}: {e})")


def _base_nodes(base_graph: dict) -> List[dict]:
    return base_graph["graph"]["objects"] + base_graph["graph"]["regions"]


def _graph_node_names(graph: dict) -> set:
    return {n["name"] for n in graph["objects"] + graph["regions"]}


def validate_rename_map(rename: dict, base_graph: dict) -> None:
    """Reject a rename map that would not survive mechanical application.

    The map is the ONLY thing the LLM says about topology, so every failure
    mode of the old free-form contract (dangling edges, dropped nodes, renamed
    coordinates) reduces to a defect in this dict. Raises ``ValueError`` so the
    caller's retry loop regenerates the graph.
    """
    if not isinstance(rename, dict):
        raise ValueError(f"rename map must be a dict, got {type(rename).__name__}")

    expected = {node["name"] for node in _base_nodes(base_graph)}
    got = set(rename)
    missing = expected - got
    extra = got - expected
    if missing or extra:
        raise ValueError(
            f"rename map does not cover the base nodes: "
            f"{len(missing)} missing {sorted(missing)[:8]}, "
            f"{len(extra)} unknown {sorted(extra)[:8]}"
        )

    bad_type = {k for k, v in rename.items() if not isinstance(v, str) or not v.strip()}
    if bad_type:
        raise ValueError(
            f"rename map has empty/non-string names for {sorted(bad_type)[:8]}"
        )

    seen: dict = {}
    collisions = []
    for old, new in rename.items():
        if new in seen:
            collisions.append(f"{seen[new]!r} and {old!r} -> {new!r}")
        seen[new] = old
    if collisions:
        raise ValueError(f"rename map is not injective: {collisions[:8]}")

    leftover = sorted(v for v in rename.values() if PLACEHOLDER_NAME.match(v))
    if leftover:
        raise ValueError(f"rename map keeps skeleton placeholder names: {leftover[:8]}")


def validate_descriptions(descriptions: dict, base_graph: dict) -> None:
    """Descriptions may only be supplied for nodes marked ``__FILL__``."""
    if not isinstance(descriptions, dict):
        raise ValueError(
            f"descriptions must be a dict, got {type(descriptions).__name__}"
        )

    expected = {
        node["name"] for node in _base_nodes(base_graph)
        if node["description"] == FILL_TOKEN
    }
    got = set(descriptions)
    missing = expected - got
    extra = got - expected
    if missing or extra:
        raise ValueError(
            f"descriptions must cover exactly the {len(expected)} __FILL__ nodes: "
            f"missing {sorted(missing)[:8]}, unknown {sorted(extra)[:8]}"
        )

    bad = {
        k for k, v in descriptions.items() if not isinstance(v, str) or not v.strip()
    }
    if bad:
        raise ValueError(f"descriptions are empty/non-string for {sorted(bad)[:8]}")


def apply_rename(base_graph: dict, rename: dict, descriptions: dict) -> dict:
    """Build the populated graph from the base graph and a validated map.

    Coordinates, connections and node counts are copied from the base graph, so
    the LLM cannot perturb them; only names and ``__FILL__`` descriptions come
    from the model.
    """
    def renamed_node(node: dict) -> dict:
        out = dict(node)
        out["name"] = rename[node["name"]]
        if node["description"] == FILL_TOKEN:
            out["description"] = descriptions[node["name"]]
        return out

    graph = base_graph["graph"]
    return {
        "objects": [renamed_node(n) for n in graph["objects"]],
        "regions": [renamed_node(n) for n in graph["regions"]],
        "object_connections": [
            [rename[a], rename[b]] for a, b in graph["object_connections"]
        ],
        "region_connections": [
            [rename[a], rename[b]] for a, b in graph["region_connections"]
        ],
        "robot_location": rename[graph["robot_location"]],
    }


def assert_graph_refs_resolve(graph: dict) -> None:
    """Final backstop: every connection endpoint names an existing node."""
    names = _graph_node_names(graph)
    for key in ("object_connections", "region_connections"):
        for edge in graph[key]:
            assert len(edge) == 2, (
                f"{key} entry {edge} has {len(edge)} endpoints, want 2"
            )
            for endpoint in edge:
                assert endpoint in names, f"{key} references unknown node {endpoint!r}"
    assert graph["robot_location"] in names, (
        f"robot_location {graph['robot_location']!r} is not a node"
    )


def validate_task_refs(tasks: List[dict], graph: dict) -> None:
    """Every node a task names must exist in the renamed graph.

    ``init_node`` is loaded straight into ``GraphHandler.reset`` at rollout
    time, and the deterministic grader resolves the goal/waypoints out of
    ``acceptance_criterion``, so a hallucinated id there yields an ungradeable
    task rather than a loud failure.
    """
    names = _graph_node_names(graph)
    region_names = {n["name"] for n in graph["regions"]}

    for i, task in enumerate(tasks):
        if task["init_node"] not in region_names:
            raise ValueError(
                f"task {i}: init_node {task['init_node']!r} is not a region"
            )
        # Answer regexes carry escapes (\b...\b); drop them so the escape letter
        # is not read as part of the following node id.
        text = re.sub(
            r"\\.", " ", f"{task['answer']} {task['acceptance_criterion']}"
        )
        unknown = set(NODE_TOKEN.findall(text)) - names
        if unknown:
            raise ValueError(
                f"task {i} references nodes that are not in the graph: {sorted(unknown)}"
            )


UPDATED_QUERY = r"""
You are generating data for training an LLM-based planner, like the SPINE paper from Ravichandran et al.

You will be given a base scene graph in the following format:

{
  "graph": {
    "objects": [{"name": "object_1_name", "coords": [west_east_coordinate, south_north_coordinate], "description": "" | "__FILL__"}, ...],
    "regions": [{"name": "region_1_name", "coords": [west_east_coordinate, south_north_coordinate], "description": ""}, ...],
    "object_connections": [["object_name", "region_name"], ...],
    "region_connections": [["some_region_name", "other_region_name"], ...],
    "robot_location": "region_of_robot_location"
  },
  "tasks": [],
  "_metadata": {
    "n_communities": <int>,
    "community_assignments": {"region_name": <community_id>, ...},
    "n_tasks": <int>
  }
}

The `_metadata` field tells you how many communities the graph has, which regions belong to which community, and how many tasks to generate. Use this to guide your theme and naming choices.

For example,

{
"graph": {
  "objects":
  [
      {"name": "object_0", "coords": [78, 9], "description": "__FILL__"},
      {"name": "object_1", "coords": [52, -56], "description": ""}
  ],
  "regions": [
      {"name": "region_0", "coords": [0, 0], "description": ""},
      {"name": "region_1", "coords": [5.7, -8.3], "description": ""},
      {"name": "region_2", "coords": [19.3, -6.5], "description": ""},
      {"name": "region_3", "coords": [35.7, -12.1], "description": ""},
      {"name": "region_4", "coords": [52.7, -20], "description": ""},
      {"name": "region_5", "coords": [57.2, -31.6], "description": ""},
      {"name": "region_6", "coords": [54.3, -46.7], "description": ""},
      {"name": "region_7", "coords": [52.4, -56.5], "description": ""},
      {"name": "region_8", "coords": [78.4, 9.1], "description": ""}
  ],
  "object_connections": [
      ["object_0", "region_8"],
      ["object_1", "region_7"]
  ],
  "region_connections":[
      ["region_0", "region_1"],
      ["region_1", "region_2"],
      ["region_2", "region_3"],
      ["region_3", "region_4"],
      ["region_4", "region_5"],
      ["region_5", "region_6"],
      ["region_6", "region_7"],
      ["region_7", "region_8"]
  ],
  "robot_location": "region_0"
},
"tasks": [],
"_metadata": {
    "n_communities": 2,
    "community_assignments": {"region_0": 0, "region_1": 0, "region_2": 0, "region_3": 0, "region_4": 1, "region_5": 1, "region_6": 1, "region_7": 1, "region_8": 1},
    "n_tasks": 4
}
}

You do NOT rewrite this graph. You return a rename map, the descriptions to fill in, and the tasks. The graph itself — coordinates, connections, node counts, robot_location — is rebuilt mechanically by applying your rename map to the base graph, so you cannot change the topology and must not try.

For the base graph above, your entire output would be:

{
  "rename": {
      "object_0": "shed_1",
      "object_1": "gate_1",
      "region_0": "ground_1",
      "region_1": "road_1",
      "region_2": "road_2",
      "region_3": "trail_1",
      "region_4": "highway_1",
      "region_5": "highway_2",
      "region_6": "bridge_1",
      "region_7": "intersection_1",
      "region_8": "driveway_1"
  },
  "descriptions": {
      "object_0": "rusted"
  },
  "tasks": [...]
}

Applying that map yields `["shed_1", "driveway_1"]` from `["object_0", "region_8"]`, `robot_location: "ground_1"` from `"region_0"`, and every coordinate untouched.

You must populate the base graph using the following steps:

### CRITICAL OUTPUT INVARIANTS (MUST NOT BE VIOLATED)

- The "rename" map must contain EXACTLY one entry per node in the base graph — every region AND every object, keyed by its ORIGINAL name. No missing keys, no invented keys.
- Every new name must be unique: two original nodes may never map to the same new name.
- No new name may look like a skeleton placeholder (`region_4`, `object_7`).
- DO NOT output "objects", "regions", "object_connections", "region_connections", "robot_location", "coords", or "_metadata". Any of those in your output is INVALID.

### Step 1: Choose theme

**Independence rule:** Each graph must be filled completely independently. Do NOT reuse themes, region types, object types, naming patterns, or task wordings from any previously filled graph in this conversation or any other. Treat each skeleton as if it is the only one you have ever seen. Choose a fresh, distinct theme every time.

If the user provides a theme, use it. Otherwise, infer a coherent theme from the topology (e.g., a graph with 3-4 communities of 5-10 regions suggests a rural area with distinct zones like fields, roads, and wooded areas). Vary your theme choices widely — do not default to the same genre (e.g., rural farmland) across multiple invocations.

### Step 2: Rename regions

Each community gets a coherent region type. Names follow the `type_N` convention with **globally unique names across all regions AND objects**.

**Uniqueness rule:** Every node name in the entire graph must be unique. A name like `field_1` may only appear once — it cannot be both a region and an object. Each `type` string (the prefix before `_N`) may appear at most **three times** in the entire graph (regions + objects combined). For example, you may have `desert_1`, `desert_2`, and `desert_3`, but not `desert_4`. If a community has more nodes than 3, use multiple distinct types within that community (e.g., `field_1`, `field_2`, `field_3`, `meadow_1`, `clearing_1`).

**Anti-pattern — monotonic naming is BANNED:** Do NOT name nodes by repeating the same type with an incrementing counter. For example, `shed_0, shed_1, shed_2, ..., shed_100` is INVALID — it violates the three-per-prefix limit and produces an unrealistic, homogeneous graph. Instead, vary the types:

BAD (invalid):  `shed_1`, `shed_2`, `shed_3`, `shed_4`, `shed_5`
GOOD (valid): `shed_1`, `shed_2`, `shed_3`, `barn_1`, `silo_1`

Examples of community → type mappings:
- Community 0 (5 nodes) → `field_1`, `field_2`, `field_3`, `meadow_1`, `clearing_1`
- Community 1 (5 nodes) → `road_1`, `road_2`, `road_3`, `highway_1`, `intersection_1`
- Community 2 (5 nodes) → `trail_1`, `trail_2`, `path_1`, `path_2`, `bridge_1`

Choose types that make sense for the theme. Types used across different communities should be distinct where possible. Record each choice in the `rename` map: `{"region_1": "field_1", "region_2": "meadow_1", ...}`.

### Step 3: Rename objects

Give objects realistic names matching their host region context. Use the `type_N` convention.

Examples: `pickup_truck_1`, `cabin_1`, `shed_1`, `light_pole_1`, `sail_boat_1`, `internet_tower_1`.

**Uniqueness rule (same as regions):** Each object name must be globally unique across all regions AND objects. Each `type` prefix may appear at most **three times** in the entire graph. Maximize diversity — avoid repeating the same type for every object. Consider what makes sense near each region type.

**Same anti-pattern applies to objects:** `crate_1, crate_2, crate_3, crate_4, ..., crate_50` is INVALID. Use diverse, contextually appropriate types like `crate_1`, `crate_2`, `barrel_1`, `toolbox_1`, `generator_1`, etc.

### Step 4: Fill descriptions (STRICT)

The `descriptions` map holds one entry for each node whose base-graph description is `"__FILL__"`, keyed by its ORIGINAL name:

`"descriptions": {"object_3": "rusted", "object_9": "locked"}`

Rules:
- Include EVERY node marked `"__FILL__"` and NOTHING else. Nodes whose description is `""` keep it and must NOT appear in the map.
- If no node is marked `"__FILL__"`, return `"descriptions": {}`.

Allowed description values are short attributes such as:
"damaged", "not damaged", "locked", "empty", "operational", "rusted", "overgrown", "flooded", "collapsed", "active"

### Step 5: Planning

Every node id you write in a task — in `init_node`, in `answer`, in `acceptance_criterion` — must be a NEW name from your `rename` map, never a base-graph name like `region_4`. Read the topology (`region_connections`, `object_connections`) off the BASE graph and translate each node through your map as you write. A task naming a node that is not a value in the map is INVALID.

Generate EXACTLY n_tasks tasks (specified in `_metadata`). The tasks should present interesting planning scenarios that assess the ability of the planner to do one of the following:

1. **Existence** — understanding node existence: is a semantic type in the graph?
   These tasks ask whether a particular kind of object or entity exists somewhere in the scene. They range from straightforward to indirect and inferential.
   Simple examples:
   - "Is there a boat in the scene?"
   - "Does the complex include a medical supply kit?"
   - "Is there something I can use to catch fish?"
   Complex examples:
   - "Is there a single storage area holding a transit crate, rescue blanket, and tag bundle together?"
   - "Is there an airport area that contains both a hold-short sign and a runway beacon?"
   - "Does the place with the torque wrench also hold a socket set?"

2. **Positionality** — understand the position of a node: what is the northmost region, which area contains an object, etc.?
   These tasks ask about spatial location, containment, counting, or finding entities by attribute. They range from direct lookups to multi-constraint searches that combine location with condition or co-presence.
   Simple examples:
   - "Where is the satellite phone stored?"
   - "Which area contains the fuel tank?"
   - "I lost my keys. I last saw them when I parked my truck."
   Complex examples:
   - "Name the item in the airfield avionics work area that is marked operational."
   - "Which other item is stored in the same area as the charger bank?"
   - "Where are the charger bank and spare battery kept together?"

3. **Reachability** — reach a target and SHOW the route: which edges connect the start to the target?
   These tasks ask the planner to reach a target area (often a short hop away) and report the connecting edge(s) and the route taken — never a bare yes/no. They range from naming both areas explicitly to referencing the robot's current position or object descriptions.
   Simple examples:
   - "Reach the cold storage area from the parking pay station and give the connecting edges and route."
   - "Show how the pump house connects to the radar pier; list the edge and the path."
   - "Get from the main harbor plaza to the chart archive and report the route taken."
   Complex examples:
   - "From the southern aircraft service apron, reach the weather-check balloon and give the edges traversed."
   - "Route from the waterfront pier to the remote cache and list each connecting edge."
   - "Reach the damaged meteorological balloon from the starting area and show the path."

4. **Navigability** — multi-hop routing: give the full path from area a to area b.
   These tasks ask for a multi-step route and the path to take, with optional constraints (required waypoints or avoided areas).
   Simple examples:
   - "Give the route from the starting area to the area containing the air purifier."
   - "Route the robot from the dockyard gate to the oxygen cart and list the path."
   - "From the starting area, give the path to the area with the medical pack."
   Complex examples:
   - "Give the two-step route from the starting area to the quadcopter's area and name the intermediate area."
   - "Route from the ice-core storage area to the locked satellite phone by passing through the communications bunker and cable vault; give the full path."
   - "From the cargo-pallet area, give the route to the laboratory with the spectrometer without using the main command area."

**Existence tasks are NOT allowed** (bare yes/no node-existence is a semantic-matching false positive). Generate only Positionality, Reachability, and Navigability — every task must require graph topology to solve and must be answered with edges and (for Reachability/Navigability) a route, never a polar yes/no.

**Required answer content — EDGES and PATHS (applies to every task):**
- Phrase each `task` so the planner must report the relevant connecting edges as `A <=> B` and, for Reachability/Navigability, the full route as `A -> B -> C`. End reachability/navigability prompts with a clause such as "give the connecting edges and the route".
- Grading is deterministic over the graph: a correct answer must STATE the containment/adjacency edge(s) (`A <=> B`) and, for routes, a valid walk from the start region to the destination region (honouring any required waypoint and avoided area). No yes/no.

Each task must be a JSON object with the following structure:

{
  "task": "the natural language question for the planner",
  "answer": "a regex pattern that matches correct answers",
  "init_node": "the renamed region where the robot starts for this task",
  "acceptance_criterion": "one sentence describing what a correct response must convey"
}

For example, if the graph contains a `fuel_depot_1` region holding a `fuel_tank_1` object, reached from `clearing_1` via `comm_bunker_1`:

{
  "task": "Reach the area holding the fuel tank from the starting area and report the connecting edges and the route.",
  "answer": "(?i)\\bclearing_1\\b.*\\bfuel_depot_1\\b",
  "init_node": "clearing_1",
  "acceptance_criterion": "A correct answer gives a valid route from clearing_1 to fuel_depot_1 (which contains fuel_tank_1), stating the edge fuel_depot_1 <=> fuel_tank_1."
}

**Answer regex rules (a coarse parallel check — grading is deterministic over the graph):**
The deterministic grader reads the goal region, any waypoints, the avoided areas, and the containment edge(s) directly from the `acceptance_criterion` (and this `answer`), then validates the planner's stated edges and route against the NetworkX graph. The regex is a secondary signal — keep it SIMPLE and node-naming, never a bare polarity.

- Wrap every node name in `\b` word boundaries: `(?i)\bfuel_depot_1\b`, never bare `fuel_depot_1` (which also matches `fuel_depot_10`). In the JSON output the backslash must itself be escaped — write `\\b`, since a bare `\b` in a JSON string is the backspace character and silently breaks the regex.
- NEVER use a bare yes/no regex for Positionality, Reachability, or Navigability — those tasks are graded on edges and paths, not polarity.
- Positionality — match the answer (destination) region name: `(?i)\bfuel_depot_1\b`.
- Reachability/Navigability — match only the route's FIRST and LAST hop (start region then destination), in order: `(?i)\bcrew_quarters_1\b.*\bcoolant_station_1\b`. If the task names a required waypoint, include it: `(?i)\bcrew_quarters_1\b.*\bpower_conduit_1\b.*\bcoolant_station_1\b`. Never encode the full path in the regex — the full walk is checked deterministically against the graph.
- Do NOT anchor with `^` or `$`. The regex should match anywhere in the response.
- Substring traps — even with `\b`, a word can be a substring of its own negation, so an affirmative token silently accepts the wrong answer. Never key a positive match on a word that also occurs in the negative phrasing: `\bcan\b`/`\bcan reach\b` appear inside "cannot", `\bpossible\b` inside "impossible"/"not possible", `\breachable\b` inside "not reachable", `\bable to reach\b` inside "not able to reach". If such a word is unavoidable, guard it with a negative lookbehind, e.g. `(?<!not )\breachable\b`. Bare `no`/`not` match "node"/"north"/"not in one move" and must never appear unanchored.
- Narration leak — `\ba path\b` and `\ba route\b` match a mere restatement of the question ("…determine whether there is a path to fuel_depot_1") which is NOT an answer. Never let these (or any prompt noun) carry a match on their own; key on node ids instead.
- Self-test before finalizing: if the regex contains any English word, mentally run it against (a) the WRONG-polarity answer and (b) a premise echo that only repeats the question. If either matches, the regex is invalid — rewrite it to key on node ids and the route.

**Acceptance criterion rules (these drive the deterministic grader):**
- Write ONE sentence describing what a correct planner response must convey.
- Name, by node id, the destination region (the goal), every required waypoint (write "via <node>" / "passing through <node>"), every avoided area (write "without using <node>"), and each object whose containment is part of the answer — so the grader can resolve the goal, constraints, and required `region <=> object` edges deterministically.
- An avoided area must be a REAL region that is neither the start nor the destination and that the intended route genuinely bypasses. Never write a vacuous constraint: do not reference a non-existent edge, do not hedge ("if it existed"), and do not name the start/goal/waypoint in the "without using" clause.
- Only write "via <node>" / "passing through <node>" when the TASK text itself imposes that waypoint. For an UNCONSTRAINED reach/route task, require only "a valid route" — do NOT name a specific intermediate region and do NOT spell out a full example path (e.g. "the route a -> b -> c") in the criterion. Many valid routes exist; the deterministic grader accepts any valid walk to the goal, and a baked-in intermediate wrongly fails a correct alternative.
- For Reachability/Navigability the criterion must name the start region, the destination region, and require a valid route between them (plus any waypoint/avoid). For Positionality it must name the region and the contained object(s) so the containment edge is required.
- Do NOT restate the task; describe the *answer*. The criterion is for offline grading ONLY — it is never shown to the planner.

Examples of good acceptance criteria:
- "A correct answer identifies fuel_depot_1 as the region containing fuel_tank_1, stating the edge fuel_depot_1 <=> fuel_tank_1."
- "A correct answer gives a route from clearing_1 to coolant_station_1 via power_conduit_1 and lists each connecting edge."
- "A correct answer routes from decon_chamber_1 to quarantine_bay_1 without using ventilation_hub_1 and shows the full path."

**Task generation instructions:**
- DO NOT reference specific objects or nodes by id in the task text. Make the planner infer these from semantic content.
- Phrase each task so the planner must output the connecting edges (`A <=> B`) and, for Reachability/Navigability, the route (`A -> B -> C`).
- Each task must be solvable from the graph alone.
- Mix the three allowed types (Positionality, Reachability, Navigability). Do not generate all tasks of the same type, and never generate an Existence (yes/no) task.

### Step 6: Validation (REQUIRED)

Before producing your final output, verify ALL of the following. If any condition fails, fix it before outputting.

1. `rename` has exactly one key per base-graph node — count the regions and objects in the input and count your keys; they must match
2. Every key in `rename` is spelled exactly as in the base graph, and no key is invented
3. All new names are globally unique (across both regions and objects); no two keys share a value
4. Each type prefix (e.g., `field` in `field_1`) appears at most THREE times across all new names
5. No new name is a skeleton placeholder like `region_0` or `object_0`
6. `descriptions` covers exactly the `"__FILL__"` nodes — no more, no fewer
7. The number of tasks equals exactly n_tasks from `_metadata`
8. Every `init_node` is a new name that you mapped from a REGION (not an object)
9. Every node id inside `answer` and `acceptance_criterion` is a value of your `rename` map
10. Every task has a non-empty `acceptance_criterion` that names the answer entity by its node name
11. Every `answer` regex uses `\b` word boundaries and, for Navigability tasks, matches only the first and last hop (start and destination regions, plus any required waypoint), never a full path

### Output format

Return ONLY valid JSON in the following format — no extra text, no reasoning, no commentary:

{
  "rename": {"<original name>": "<new name>", ...},
  "descriptions": {"<original name of a __FILL__ node>": "<short attribute>", ...},
  "tasks": [...]
}

"""

class TaskGraphGen:
    def __init__(self, client=None):
        # Backend selected by PRISM_LLM_BACKEND (default "openai"). Set it to
        # "hf" to populate graphs/tasks with a local Gemma 4 model, or "vllm"
        # to run the same model through a continuously-batched vLLM engine.
        # An explicit `client` always wins.
        if client is not None:
            self.client = client
        elif vllm_llm.vllm_backend_enabled():
            self.client = vllm_llm.VLLMQueryClient()
        elif local_llm.hf_backend_enabled():
            self.client = local_llm.LocalHFQueryClient()
        else:
            self.client = utils.GPTQueryClient()  # OpenAI()

    def build_prompt(
        self,
        base_graph: str,
        n_tasks: Optional[int] = 2,
        prior: Optional[str] = "",
        previous_tasks: Optional[str] = "",
        task_types: Optional[List[int]] = None,
        task_complexities: Optional[List[int]] = None,
    ):
        query = (
            UPDATED_QUERY
            + f"\nYour graph should populate the base graph provided below and you should generate {n_tasks} tasks.\nBase graph:\n{base_graph}"
        )

        if task_types is not None:
            query += (
                "\n\nTask Taxonomy (Existence is DISALLOWED — treat any 0 as Positionality)\n"
                "0. Positionality (Existence is banned; generate Positionality instead)\n"
                "1. Positionality (within graph) — answer with containment edge(s)\n"
                "2. Reachability (reach the target; answer with the edge(s) and the route)\n"
                "3. Navigability (multi-hop; answer with the full route and its edges)\n"
                f"\nHere is a list of the types for the tasks: {task_types}\n"
                "Generate tasks in order, matching each entry to the type above. Every "
                "task must be answered with edges (A <=> B) and, for Reachability/"
                "Navigability, a route (A -> B -> C) — never a bare yes/no."
            )

        if task_complexities is not None:
            query += (
                "\n\nTask Complexity\n"
                "0. Simple\n"
                "1. Complex\n"
                f"\nHere is a list of the complexities for the tasks: {task_complexities}\n"
                "Generate tasks in order, matching each entry in the list to the "
                "corresponding complexity above."
            )

        if previous_tasks != "":
            query += f"\nPrevious tasks are: {previous_tasks}\nTry not to duplicate"

        if prior != "":
            query += f"\nYour tasks and scene should be like the following: {prior}"

        return query

    def get_tasks(
        self,
        base_graph: str,
        n_tasks: int = 10,
        description="",
        previous_tasks: str = "",
        task_types: Optional[List[int]] = None,
        task_complexities: Optional[List[int]] = None,
        reasoning_effort: str = "low",
    ) -> List[str]:
        """Get GPT generated tasks for putting planner data

        Parameters
        ----------
        n_regions : int, optional
            Number of regions in the graph, by default 10
        n_objects : int, optional
            Number of objects in the graph, by default 10
        n_tasks : int, optional
            Number of tasks to generate, by default 2
        description : str, optional
            an example/prior scene description to give the LLM to base the tasks on.
        task_types : list[int], optional
            Per-task type labels (see TASK_TAXONOMY) to steer the task mix.
        task_complexities : list[int], optional
            Per-task complexity labels (0=Simple, 1=Complex).
        reasoning_effort : str
            Reasoning effort passed through to the LLM client.

        Returns
        -------
        List[str]
            list of tasks
        """
        response = self.client.query_gpt(
            query=self.build_prompt(
                n_tasks=n_tasks,
                prior=description,
                base_graph=base_graph,
                previous_tasks=previous_tasks,
                task_types=task_types,
                task_complexities=task_complexities,
            ),
            reasoning_effort=reasoning_effort,
        )

        return self.parse_response(
            response, base_graph, description=description, n_tasks=n_tasks
        )

    def parse_response(
        self,
        response: str,
        base_graph: str,
        description: str = "",
        n_tasks: Optional[int] = None,
    ) -> dict:
        """Rebuild the populated graph from a raw LLM response and the skeleton.

        The response carries only a rename map, the ``__FILL__`` descriptions
        and the tasks; the graph is reconstructed from ``base_graph`` so
        topology and coordinates are the skeleton's by construction. Anything
        the model got wrong about the map raises ``ValueError`` for the
        caller's retry loop.

        When ``n_tasks`` is given, the parsed ``tasks`` list is hard-capped to
        that many entries — the LLM does not reliably honour the requested
        count, so this guarantees at most ``n_tasks`` tasks per graph.
        """
        print(response)
        base = json.loads(base_graph)
        llm_content = json.loads(response)

        validate_rename_map(llm_content["rename"], base)
        validate_descriptions(llm_content["descriptions"], base)
        graph = apply_rename(base, llm_content["rename"], llm_content["descriptions"])
        assert_graph_refs_resolve(graph)

        json_content = {
            "graph": graph,
            "tasks": llm_content["tasks"],
            "description": description,
        }

        # Enforce the requested task count: the model may over-produce (e.g. 21
        # when asked for 20), so truncate to exactly n_tasks.
        if n_tasks is not None:
            n_got = len(json_content["tasks"])
            if n_got != n_tasks:
                print(
                    f"[graph-gen] model returned {n_got} tasks; "
                    f"capping to requested n_tasks={n_tasks}"
                )
            json_content["tasks"] = json_content["tasks"][:n_tasks]

        validate_task_refs(json_content["tasks"], graph)
        _validate_tasks(json_content)
        return json_content
