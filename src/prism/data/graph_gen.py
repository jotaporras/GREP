import json
from typing import List, Optional

from spine.mapping.graph_util import GraphHandler

from prism.data import utils


QUERY = r"""
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

After populating, the same graph might look like this (note: coordinates are unchanged, names are realistic, `__FILL__` is replaced, `""` is untouched, and `_metadata` is removed):

{
"graph": {
  "objects":
  [
      {"name": "shed_1", "coords": [78, 9], "description": "rusted"},
      {"name": "gate_1", "coords": [52, -56], "description": ""}
  ],
  "regions": [
      {"name": "ground_1", "coords": [0, 0], "description": ""},
      {"name": "road_1", "coords": [5.7, -8.3], "description": ""},
      {"name": "road_2", "coords": [19.3, -6.5], "description": ""},
      {"name": "trail_1", "coords": [35.7, -12.1], "description": ""},
      {"name": "highway_1", "coords": [52.7, -20], "description": ""},
      {"name": "highway_2", "coords": [57.2, -31.6], "description": ""},
      {"name": "bridge_1", "coords": [54.3, -46.7], "description": ""},
      {"name": "intersection_1", "coords": [52.4, -56.5], "description": ""},
      {"name": "driveway_1", "coords": [78.4, 9.1], "description": ""}
  ],
  "object_connections": [
      ["shed_1", "driveway_1"],
      ["gate_1", "intersection_1"]
  ],
  "region_connections":[
      ["ground_1", "road_1"],
      ["road_1", "road_2"],
      ["road_2", "trail_1"],
      ["trail_1", "highway_1"],
      ["highway_1", "highway_2"],
      ["highway_2", "bridge_1"],
      ["bridge_1", "intersection_1"],
      ["intersection_1", "driveway_1"]
  ],
  "robot_location": "ground_1"
},
"tasks": [...]
}

You must populate the base graph using the following steps:

### CRITICAL GRAPH INVARIANTS (MUST NOT BE VIOLATED)

You are given a base graph. You MUST preserve the following exactly:

- DO NOT modify any coordinates under any circumstance.
- DO NOT reorder or alter coordinate values.
- DO NOT add, remove, or perturb coordinates.
- The "coords" field for every region and object must remain EXACTLY as provided.
- DO NOT add or remove nodes. The number of regions and objects must match the input exactly.
- DO NOT change the graph topology. All connections (both object_connections and region_connections) must remain exactly as given, only with names updated to reflect renames.

The ONLY allowed modifications are:
- renaming nodes (name field)
- filling descriptions (only replacing "__FILL__" — never modifying "")
- generating tasks
- updating robot_location to its renamed value

If any coordinate, connection, or node count changes, the output is INVALID.

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

Choose types that make sense for the theme. Types used across different communities should be distinct where possible. Maintain a rename map: `{"region_1": "field_1", "region_2": "meadow_1", ...}`.

### Step 3: Rename objects

Give objects realistic names matching their host region context. Use the `type_N` convention.

Examples: `pickup_truck_1`, `cabin_1`, `shed_1`, `light_pole_1`, `sail_boat_1`, `internet_tower_1`.

**Uniqueness rule (same as regions):** Each object name must be globally unique across all regions AND objects. Each `type` prefix may appear at most **three times** in the entire graph. Maximize diversity — avoid repeating the same type for every object. Consider what makes sense near each region type.

**Same anti-pattern applies to objects:** `crate_1, crate_2, crate_3, crate_4, ..., crate_50` is INVALID. Use diverse, contextually appropriate types like `crate_1`, `crate_2`, `barrel_1`, `toolbox_1`, `generator_1`, etc.

### Step 4: Fill descriptions (STRICT)

Each node has a "description" field that is either:
- "__FILL__" → MUST be replaced with a short attribute string
- "" (empty string) → MUST remain EXACTLY "" (do not modify)

Rules:
- DO NOT add descriptions to entries with "".
- DO NOT change "" to any other value.
- DO NOT remove descriptions that already exist.
- ONLY replace "__FILL__".

Allowed description values are short attributes such as:
"damaged", "not damaged", "locked", "empty", "operational", "rusted", "overgrown", "flooded", "collapsed", "active"

If you modify an empty string "", the output is INVALID.

### Step 5: Planning

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

3. **Reachability** — assess reachability: is one node directly connected to another node, i.e. one hop?
   These tasks ask whether the robot can move directly between two areas in a single step — one edge traversal. They range from naming both areas explicitly to referencing the robot's current position or object descriptions.
   Simple examples:
   - "Can the robot move directly from the cold storage area to the area with the parking pay station?"
   - "Is the pump house directly connected to the radar pier?"
   - "Can the robot move directly from the main harbor plaza to the chart archive?"
   Complex examples:
   - "From the southern aircraft service apron, can the robot reach the weather-check balloon in one move?"
   - "From the waterfront pier, is there a direct link to the remote cache?"
   - "Can the robot directly reach the damaged meteorological balloon from its starting area?"

4. **Navigability** — multi-hop routing: is there a route from one area to another, and what is it?
   Phrase EVERY Navigability task as a yes/no question — "Is there a route from <A> to <B>?" — and ask the planner to give the route if one exists. The planner answers "yes" and then lists the path. Choose the start, end, and any constraint so that a route DOES exist — the rollout should demonstrate a valid path.
   **Complex Navigability tasks SHOULD add a constraint** — either a required waypoint ("by way of ...", "passing through ...") or an avoided area ("without passing through ..."). Favor these constrained variants for complex tasks; they enrich the dataset.
   Simple examples:
   - "Is there a route from the starting area to the area with the air purifier? If so, give it."
   - "Can the robot get from the dockyard gate to the oxygen cart? Provide the route."
   - "Is there a path from the crew quarters to the coolant service point? List the route."
   Complex examples:
   - "Is there a route from the ice-core store to the locked satellite phone that passes through the communications bunker? If so, give it."
   - "Can the robot reach the spectrometer lab from the cargo-pallet area without going through the main command area? Provide the route."
   - "Is there a route from the waste-processing area to the medical lab by way of the crew recreation area? Give the route."

**Task difficulty (simple vs complex)**

Every task is either **simple** or **complex**. Use the examples above as style guides for each level within its task type.

**Difficulty mix (required):** Unless a per-task difficulty list is provided below, generate tasks with a **1:1 ratio of simple to complex** (50% simple, 50% complex). For `n_tasks` tasks, the counts must be exactly `floor(n_tasks / 2)` simple and `n_tasks - floor(n_tasks / 2)` complex. Do not choose your own ratio.

Each task must be a JSON object with the following structure:

{
  "task": "the natural language question for the planner",
  "answer": "a regex pattern that matches correct answers",
  "init_node": "the renamed region where the robot starts for this task",
  "acceptance_criterion": "one sentence describing what a correct response must convey"
}

For example, if the graph contains a `fuel_depot_1` region with a `fuel_tank_1` object:

{
  "task": "Is there any fuel storage facility nearby?",
  "answer": "(?i)\byes\b",
  "init_node": "clearing_1",
  "acceptance_criterion": "A correct answer affirms that fuel_depot_1 exists and contains fuel_tank_1."
}

**Answer regex rules:**
The `answer` regex is only a coarse automatic check — the `acceptance_criterion` is the authoritative grader. Keep the regex SIMPLE; it must accept correct answers and reject wrong ones.

- Wrap every word or phrase in `\b` word boundaries (e.g. `\byes\b`, `\bcan\b`). Without them a token matches inside larger words and silently accepts wrong answers — bare `can` matches "cannot", `correct` matches "incorrect", `no` matches "node"/"north".
- Yes/no tasks — POLARITY LEAK is the most common bug: never use a phrase for one polarity that also appears inside the opposite answer. `there is` is INVALID for a "yes" answer because it occurs in the "no" answer "there is no ..."; likewise never key on `reachable`, `connected`, or `correct` (they occur in "not reachable", "not connected", "not correct"). The planner reliably says "yes" or "no", so use exactly:
  - "yes" answer → `(?i)\byes\b`
  - "no" answer → `(?i)\bno\b`
  Then test the regex against the WRONG-polarity answer (e.g. "No, it cannot ..." for a yes-task); if it still matches, the regex is invalid — fix it.
- Navigability tasks — the regex matches only the route's FIRST and LAST hop: the start (init_node) region and the destination region, in order — e.g. `(?i)\bcrew_quarters_1\b.*\bcoolant_station_1\b`. If the task names a required waypoint, include it too (it is equally mandatory): `(?i)\bcrew_quarters_1\b.*\bpower_conduit_1\b.*\bcoolant_station_1\b`. NEVER encode the full path — many valid routes exist and a full-path regex rejects correct alternatives. The yes/no correctness and the middle of the route are checked by the judge, not the regex.
- Existence/location tasks — match the answer node name with `\b` boundaries: `(?i)\bfuel_depot_1\b`, never bare `fuel_depot_1` (which also matches `fuel_depot_10`).
- Do NOT anchor with `^` or `$`. The regex should match anywhere in the response.

**Acceptance criterion rules:**
- Write ONE sentence describing what a correct planner response must convey
- Reference the answer entity by its node name (e.g., `fuel_depot_1`) so an LLM judge can verify without re-solving the task
- For yes/no tasks: state the correct polarity plus the supporting entity
- For Navigability tasks: the criterion must tell the judge to check (1) the yes/no answer is correct, and (2) the planner output a route whose first hop leaves the init_node region and whose last hop reaches the destination region, plus any required waypoint or avoided area named in the task. Full edge-by-edge path validation is deferred to a later solvability check, so do not require it here
- Do NOT restate the task; describe the *answer*
- The criterion is for offline grading ONLY — it will never be shown to the planner

Examples of good acceptance criteria:
- "A correct answer identifies fuel_depot_1 as the region containing fuel_tank_1."
- "A correct answer confirms a route exists and outputs a path whose first hop leaves clearing_1 and whose last hop reaches fuel_depot_1."
- "A correct answer identifies driveway_1 as the northmost region in the scene."

**Task generation instructions:**
- DO NOT refer to nodes by their renamed `name_N` form (e.g. `fuel_depot_1`, `satellite_phone_1`). Refer to them only by their semantic content — type words, role descriptions, contained objects, or attributes (e.g. "the fuel depot", "the satellite phone", "the area with the truck"). The planner must locate the actual node from this description.
- Tasks should request specific information, not general exploration. Make the planner map or inspect certain entities. For example, start tasks with phrases such as "what", "I heard", "find out", "map", "inspect", "Can I", "is there", and likewise.
- Each task must be solvable from the graph alone.
- A per-task list of task types (TASK_TYPES) will be provided below, with one entry per task. Generate tasks in the exact order of that list, matching each entry to the corresponding category above.
- Match each task's difficulty to the required simple/complex mix or per-task difficulty list. Do not make every task simple or every task complex unless the list says so.

### Step 6: Update robot location

After renaming all regions, update the `robot_location` field to use the new renamed region name. For example, if `region_0` was renamed to `clearing_1`, then `robot_location` must become `"clearing_1"`.

### Step 7: Remove metadata

Remove the `_metadata` field completely from the output. It is only used as input guidance and must not appear in the final JSON.

### Step 8: Validation (REQUIRED)

Before producing your final output, verify ALL of the following. If any condition fails, fix it before outputting.

1. ALL coordinates are EXACTLY unchanged from the input
2. NO "" descriptions were modified (they must still be "")
3. NO "__FILL__" placeholders remain — all must be replaced
4. All node names are globally unique (across both regions and objects)
5. Each type prefix (e.g., `field` in `field_1`) appears at most THREE times in the entire graph
6. All connections reference valid, renamed node names
7. No original placeholder names (like `region_0`, `object_0`) remain
8. The number of tasks equals exactly n_tasks from `_metadata`
9. The simple/complex counts match the required difficulty mix (or the per-task difficulty list, if provided)
10. `robot_location` references a valid renamed region.
11. All `init_node` values in tasks reference valid renamed regions. For Existence and Positionality tasks, `init_node` MUST NOT be the answer region itself (the region that contains the answer object, or is itself the answer), and should preferably not be directly adjacent to it either — the planner must have to traverse the graph to find the answer, not read it off its starting observation. For Reachability and Navigability tasks, `init_node` may be one of the endpoints referenced by the task
12. Every task has a non-empty `acceptance_criterion` that names the answer entity by its node name
13. Every `answer` regex uses `\b` word boundaries, rejects the opposite-polarity answer for yes/no tasks, and for Navigability tasks matches only the first and last hop (start and destination regions, plus any required waypoint), never a full path

### Output format

Return ONLY valid JSON in the following format — no extra text, no reasoning, no commentary:

{
  "graph": {
    "objects": [...],
    "regions": [...],
    "object_connections": [...],
    "region_connections": [...],
    "robot_location": "..."
  },
  "tasks": [...]
}

"""

UPDATED_QUERY = """
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

After populating, the same graph might look like this (note: coordinates are unchanged, names are realistic, `__FILL__` is replaced, `""` is untouched, and `_metadata` is removed):

{
"graph": {
  "objects":
  [
      {"name": "shed_1", "coords": [78, 9], "description": "rusted"},
      {"name": "gate_1", "coords": [52, -56], "description": ""}
  ],
  "regions": [
      {"name": "ground_1", "coords": [0, 0], "description": ""},
      {"name": "road_1", "coords": [5.7, -8.3], "description": ""},
      {"name": "road_2", "coords": [19.3, -6.5], "description": ""},
      {"name": "trail_1", "coords": [35.7, -12.1], "description": ""},
      {"name": "highway_1", "coords": [52.7, -20], "description": ""},
      {"name": "highway_2", "coords": [57.2, -31.6], "description": ""},
      {"name": "bridge_1", "coords": [54.3, -46.7], "description": ""},
      {"name": "intersection_1", "coords": [52.4, -56.5], "description": ""},
      {"name": "driveway_1", "coords": [78.4, 9.1], "description": ""}
  ],
  "object_connections": [
      ["shed_1", "driveway_1"],
      ["gate_1", "intersection_1"]
  ],
  "region_connections":[
      ["ground_1", "road_1"],
      ["road_1", "road_2"],
      ["road_2", "trail_1"],
      ["trail_1", "highway_1"],
      ["highway_1", "highway_2"],
      ["highway_2", "bridge_1"],
      ["bridge_1", "intersection_1"],
      ["intersection_1", "driveway_1"]
  ],
  "robot_location": "ground_1"
},
"tasks": [...]
}

You must populate the base graph using the following steps:

### CRITICAL GRAPH INVARIANTS (MUST NOT BE VIOLATED)

You are given a base graph. You MUST preserve the following exactly:

- DO NOT modify any coordinates under any circumstance.
- DO NOT reorder or alter coordinate values.
- DO NOT add, remove, or perturb coordinates.
- The "coords" field for every region and object must remain EXACTLY as provided.
- DO NOT add or remove nodes. The number of regions and objects must match the input exactly.
- DO NOT change the graph topology. All connections (both object_connections and region_connections) must remain exactly as given, only with names updated to reflect renames.

The ONLY allowed modifications are:
- renaming nodes (name field)
- filling descriptions (only replacing "__FILL__" — never modifying "")
- generating tasks
- updating robot_location to its renamed value

If any coordinate, connection, or node count changes, the output is INVALID.

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

Choose types that make sense for the theme. Types used across different communities should be distinct where possible. Maintain a rename map: `{"region_1": "field_1", "region_2": "meadow_1", ...}`.

### Step 3: Rename objects

Give objects realistic names matching their host region context. Use the `type_N` convention.

Examples: `pickup_truck_1`, `cabin_1`, `shed_1`, `light_pole_1`, `sail_boat_1`, `internet_tower_1`.

**Uniqueness rule (same as regions):** Each object name must be globally unique across all regions AND objects. Each `type` prefix may appear at most **three times** in the entire graph. Maximize diversity — avoid repeating the same type for every object. Consider what makes sense near each region type.

**Same anti-pattern applies to objects:** `crate_1, crate_2, crate_3, crate_4, ..., crate_50` is INVALID. Use diverse, contextually appropriate types like `crate_1`, `crate_2`, `barrel_1`, `toolbox_1`, `generator_1`, etc.

### Step 4: Fill descriptions (STRICT)

Each node has a "description" field that is either:
- "__FILL__" → MUST be replaced with a short attribute string
- "" (empty string) → MUST remain EXACTLY "" (do not modify)

Rules:
- DO NOT add descriptions to entries with "".
- DO NOT change "" to any other value.
- DO NOT remove descriptions that already exist.
- ONLY replace "__FILL__".

Allowed description values are short attributes such as:
"damaged", "not damaged", "locked", "empty", "operational", "rusted", "overgrown", "flooded", "collapsed", "active"

If you modify an empty string "", the output is INVALID.

### Step 5: Planning

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

3. **Reachability** — assess reachability: is one node directly connected to another node, i.e. one hop?
   These tasks ask whether the robot can move directly between two areas in a single step — one edge traversal. They range from naming both areas explicitly to referencing the robot's current position or object descriptions.
   Simple examples:
   - "Can the robot move directly from the cold storage area to the area with the parking pay station?"
   - "Is the pump house directly connected to the radar pier?"
   - "Can the robot move directly from the main harbor plaza to the chart archive?"
   Complex examples:
   - "From the southern aircraft service apron, can the robot reach the weather-check balloon in one move?"
   - "From the waterfront pier, is there a direct link to the remote cache?"
   - "Can the robot reach the damaged meteorological balloon from its starting area?"

4. **Navigability** — understand navigation: give a path from node a to node b, potentially multi-hop?
   These tasks ask whether a multi-step route exists, what path to take, or which intermediate areas to use. They range from simple reachability over multiple hops to constrained path planning with avoidance, specific waypoints, or structural graph queries.
   Simple examples:
   - "Can a route from the starting area reach the area containing the air purifier?"
   - "Can the robot get from the dockyard gate to the oxygen cart using mapped areas?"
   - "From the starting area, can the robot reach the area with the medical pack?"
   Complex examples:
   - "Which intermediate area gives a two-step route from the starting area to the quadcopter's area?"
   - "Can the robot get from the ice-core storage area to the locked satellite phone by passing through the communications bunker and cable vault?"
   - "Starting at the place with the cargo pallet, can the robot reach the laboratory with the spectrometer without using the main command area?"

Each task must be a JSON object with the following structure:

{
  "task": "the natural language question for the planner",
  "answer": "a regex pattern that matches correct answers",
  "init_node": "the renamed region where the robot starts for this task",
  "acceptance_criterion": "one sentence describing what a correct response must convey"
}

For example, if the graph contains a `fuel_depot_1` region with a `fuel_tank_1` object:

{
  "task": "Is there any fuel storage facility nearby?",
  "answer": "(?i)(yes|there is|fuel)",
  "init_node": "clearing_1",
  "acceptance_criterion": "A correct answer affirms that fuel_depot_1 exists and contains fuel_tank_1."
}

**Answer regex rules:**
- Include synonyms and common phrasings so the regex does not miss valid answers
- Avoid ambiguous substrings that would match incorrect responses (e.g., don't use `"no"` as a pattern — it matches "north", "node", etc.)
- Yes/no patterns must match the correct polarity only. For a "yes" answer, use `(?i)(yes|there is|it does)` — never a pattern that also matches "no"
- Do NOT anchor with `^` or `$`. The regex should match anywhere in the response.

**Acceptance criterion rules:**
- Write ONE sentence describing what a correct planner response must convey
- Reference the answer entity by its node name (e.g., `fuel_depot_1`) so an LLM judge can verify without re-solving the task
- For yes/no tasks: state the correct polarity plus the supporting entity
- Do NOT restate the task; describe the *answer*
- The criterion is for offline grading ONLY — it will never be shown to the planner

Examples of good acceptance criteria:
- "A correct answer identifies fuel_depot_1 as the region containing fuel_tank_1."
- "A correct answer affirms reachability and cites the path through storage_tent_1."
- "A correct answer is the number 2, the count of regions with exactly three neighbors."

**Task generation instructions:**
- DO NOT reference specific objects or nodes in the task text. Make the planner infer these.
- Tasks should request specific information, not general exploration. Make the planner map or inspect certain entities. For example, start tasks with phrases such as "what", "I heard", "find out", "map", "inspect", "Can I", "is there", and likewise.
- Each task must be solvable from the graph alone.
- Mix task types across all four categories. Do not generate all tasks of the same type.

### Step 6: Update robot location

After renaming all regions, update the `robot_location` field to use the new renamed region name. For example, if `region_0` was renamed to `clearing_1`, then `robot_location` must become `"clearing_1"`.

### Step 7: Remove metadata

Remove the `_metadata` field completely from the output. It is only used as input guidance and must not appear in the final JSON.

### Step 8: Validation (REQUIRED)

Before producing your final output, verify ALL of the following. If any condition fails, fix it before outputting.

1. ALL coordinates are EXACTLY unchanged from the input
2. NO "" descriptions were modified (they must still be "")
3. NO "__FILL__" placeholders remain — all must be replaced
4. All node names are globally unique (across both regions and objects)
5. Each type prefix (e.g., `field` in `field_1`) appears at most THREE times in the entire graph
6. All connections reference valid, renamed node names
7. No original placeholder names (like `region_0`, `object_0`) remain
8. The number of tasks equals exactly n_tasks from `_metadata`
9. `robot_location` references a valid renamed region
10. All `init_node` values in tasks reference valid renamed regions
11. Every task has a non-empty `acceptance_criterion` that names the answer entity by its node name

### Output format

Return ONLY valid JSON in the following format — no extra text, no reasoning, no commentary:

{
  "graph": {
    "objects": [...],
    "regions": [...],
    "object_connections": [...],
    "region_connections": [...],
    "robot_location": "..."
  },
  "tasks": [...]
}

"""

class TaskGraphGen:
    def __init__(self):
        self.client = utils.GPTQueryClient()  # OpenAI()

    def build_prompt(
        self,
        base_graph: str,
        n_tasks: Optional[int] = 2,
        prior: Optional[str] = "",
        previous_tasks: Optional[str] = "",
        task_types: Optional[List[int]] = None,
    ):
        query = (
            UPDATED_QUERY
            + f"\nYour graph should populate the base graph provided below and you should generate {n_tasks} tasks.\nBase graph:\n{base_graph}"
        )

        if task_types is not None:
            query += (
                "\n\nTask Taxonomy\n"
                "0. Existence (of a node)\n"
                "1. Positionality (within graph)\n"
                "2. Reachability (with one edge)\n"
                "3. Navigability (with multiple edges)\n"
                f"\nHere is a list of the types for the tasks: {task_types}\n"
                "Generate tasks in order, matching each entry in the list to the "
                "corresponding task type above."
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
            )
        )

        return self.parse_response(response, description=description)

    def parse_response(self, response: str, description: str = "") -> dict:
        """Validate and parse a raw GPT response string into a task dict."""
        print(response)
        json_content = json.loads(response)
        json_content["description"] = description
        graph_handle = GraphHandler(graph="")
        graph_handle.reset(
            json_content["graph"],
            current_location=json_content["graph"]["robot_location"],
        )

        # make sure GPT isn't hallucinating edges
        for [source, end] in graph_handle.graph.edges:
            assert source in graph_handle.graph.nodes, f"{source} not in graph"
            assert end in graph_handle.graph.nodes, f"{end} not in graph"

        return json_content


if __name__ == "__main__":
    gen = TaskGraphGen()

    rnd_data = gen.get_tasks()

    graph_handler = GraphHandler(graph_path="")

    whatdoihave = 0
