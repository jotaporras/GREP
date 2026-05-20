import json
from typing import List, Optional

from spine.mapping.graph_util import GraphHandler

from prism.data import utils

# Old query
# QUERY = """
# You are generating data for training an llm-based planner, like the SPINE paper from ravichandran et al.

# You will be given a base scene graph in the following format

# {
#         "objects": [{"name": "object_1_name", "coords": [west_east_coordinate, south_north_coordinate]}, ...],
#         "regions": [{"name": "region_1_name", "coords": [west_east_coordinate, south_north_coordinate]}, ...],
#         "object_connections: [["object_name", "region_name"], ...],
#         "region_connections": [["some_region_name", "other_region_name"], ...]
#         "robot_location": "region_of_robot_location
# }

# For example,

# {
# "objects":
# [
#     {"name": "shed_1", "coords": [78, 9]},
#     {"name": "gate_1", "coords": [52, -56]}
# ],
# "regions": [
#     {"name": "ground_1", "coords": [0, 0]},
#     {"name": "road_1", "coords": [5.7, -8.3]},
#     {"name": "road_2", "coords": [19.3, -6.5]},
#     {"name": "road_3", "coords": [35.7, -12.1]},
#     {"name": "road_4", "coords": [52.7, -20]},
#     {"name": "road_5", "coords": [57.2, -31.6]},
#     {"name": "bridge_1", "coords": [54.3, -46.7]},
#     {"name": "road_6", "coords": [52.4, -56.5]},
#     {"name": "driveway_1", "coords": [78.4, 9.1]}
# ],
# "object_connections": [
#     ["shed_1", "driveway_1"],
#     ["gate_1", "road_6"]
# ],
# "region_connections":[
#     ["ground_1", "road_1"],
#     ["road_1", "road_2"],
#     ["road_2", "road_3"],
#     ["road_3", "road_4"],
#     ["road_4", "road_5"],
#     ["road_5", "bridge_1"],
#     ["bridge_1", "road_6"],
#     ["road_6", "driveway_1"]
# ],
# "robot_location": "ground_1"
# }

# You must populate the base graph using the following steps:

# ### CRITICAL GRAPH INVARIANTS (MUST NOT BE VIOLATED)

# You are given a base graph. You MUST preserve the following exactly:

# - DO NOT modify any coordinates under any circumstance.
# - DO NOT reorder or alter coordinate values.
# - DO NOT add, remove, or perturb coordinates.
# - The "coords" field for every region and object must remain EXACTLY as provided.

# The ONLY allowed modifications are:
# - renaming nodes (name field)
# - adding descriptions (only when explicitly allowed)
# - generating tasks

# If any coordinate is changed, the output is INVALID.

# ### Step 1: Choose theme

# **Independence rule:** Each graph must be filled completely independently. Do NOT reuse themes, region types, object types, naming patterns, or task wordings from any previously filled graph in this conversation or any other. Treat each skeleton as if it is the only one you have ever seen. Choose a fresh, distinct theme every time.

# If the user provides a theme, use it. Otherwise, infer a coherent theme from the topology (e.g., a graph with 3-4 communities of 5-10 regions suggests a rural area with distinct zones like fields, roads, and wooded areas). Vary your theme choices widely — do not default to the same genre (e.g., rural farmland) across multiple invocations.

# ### Step 2: Rename regions

# Each community gets a coherent region type. Names follow the `type_N` convention with **globally unique names across all regions AND objects**.

# **Uniqueness rule:** Every node name in the entire graph must be unique. A name like `field_1` may only appear once — it cannot be both a region and an object. Each `type` string (the prefix before `_N`) may appear at most **twice** in the entire graph (regions + objects combined). For example, you may have `desert_1` and `desert_2`, but not `desert_3`. If a community has more nodes than 2, use multiple distinct types within that community (e.g., `field_1`, `field_2`, `meadow_1`, `meadow_2`, `clearing_1`).

# Examples of community → type mappings:
# - Community 0 (5 nodes) → `field_1`, `field_2`, `meadow_1`, `meadow_2`, `clearing_1`
# - Community 1 (5 nodes) → `road_1`, `road_2`, `highway_1`, `intersection_1`, `parking_lot_1`
# - Community 2 (5 nodes) → `trail_1`, `trail_2`, `path_1`, `path_2`, `bridge_1`

# Choose types that make sense for the theme. Types used across different communities should be distinct where possible. Maintain a rename map: `{"region_1": "field_1", "region_2": "meadow_1", ...}`.

# ### Step 3: Rename objects

# Give objects realistic names matching their host region context. Use the `type_N` convention.

# Examples: `pickup_truck_1`, `cabin_1`, `shed_1`, `light_pole_1`, `sail_boat_1`, `internet_tower_1`.

# **Uniqueness rule (same as regions):** Each object name must be globally unique across all regions AND objects. Each `type` prefix may appear at most **twice** in the entire graph. Maximize diversity — avoid repeating the same type for every object. Consider what makes sense near each region type.

# ### Step 4: Fill descriptions (STRICT)

# Each node has a "description" field that is either:
# - "__FILL__" → MUST be replaced with a short attribute string
# - "" (empty string) → MUST remain EXACTLY "" (do not modify)

# Rules:
# - DO NOT add descriptions to entries with "".
# - DO NOT change "" to any other value.
# - DO NOT remove descriptions that already exist.
# - ONLY replace "__FILL__".

# If you modify an empty string "", the output is INVALID.

# ### Step 4: Planning

# Then, provide tasks that present interesting planning scenarios. The tasks should assess the ability of the planner to do one of the following
# 1. understanding node existance (is a semantic type in the graph)?
# 2. understand the position of a node (what is the northmost region, etc.)?
# 3. assess reachability (is one node connected to another node)?
# 4. understand navigation (give a path from node a to node b)


# The planner should be able to respond to your tasks via the answer() function. This should not require mapping, navigation, etc.


# Provide your answer in the following JSON format:

# {
# reasoning: describe the type of scene you are creating,
# graph: <JSON GRAPH>,
# tasks: list of tasks that correspond to the graph.
# }


# Add a "description" attribute to each node that provides information.
# These will be hidden from the robot

# Task generation instructions
# - DO NOT reference specific objects or nodes. Make the planner infer these.
# - Tasks should request specific information, not general exploration. Make the planner map or inspect certain entities. For example, start tasks with phrases such as "what", "I heard", "find out", "map", "inspect", "Can I", "is there", and likewise

# """

SCENE_PRIOR = """We are improving the SPINE planner proposed by ravichandran et al.
You need to generate data for training. Describe scenes you would train in, such as regions, objects, and general scene description

Describe ONE example environment, including scene, regions, and objects.
Such as `semi-urban office park with fields, roads, parking lots, buildings, people...` and more.

You will be randomly sampled, so be creative but realistic.

Your response should be a JSON with a "description" key, the value be the description.
"""


QUERY = """
You are generating high-quality evaluation data for an LLM-based planner (PRISM-style).

You will be given a graph skeleton JSON. Your job is to transform it into a fully populated evaluation JSON.

You must follow ALL instructions exactly. Any violation makes the output invalid.

--------------------------------------------------
CRITICAL GRAPH INVARIANTS (MUST NOT BE VIOLATED)
--------------------------------------------------

You are given a base graph. You MUST preserve:

- DO NOT modify any coordinates under any circumstance
- DO NOT reorder coordinate arrays
- DO NOT change graph topology (connections)
- DO NOT add or remove nodes

The ONLY allowed changes:
- rename nodes (name fields)
- fill descriptions (ONLY when allowed)
- generate tasks
- update robot_location to renamed value

If any coordinate or connection changes, the output is INVALID.

--------------------------------------------------
INPUT FORMAT
--------------------------------------------------

You will receive a JSON:

{
  "graph": {
    "objects": [{"name": "...", "coords": [...], "description": "" | "__FILL__"}],
    "regions": [{"name": "...", "coords": [...], "description": ""}],
    "object_connections": [[...]],
    "region_connections": [[...]],
    "robot_location": "region_X"
  },
  "tasks": [],
  "_metadata": {...}
}

--------------------------------------------------
STEP 1: UNDERSTAND GRAPH
--------------------------------------------------

Use `_metadata`:
- number of communities
- region assignments
- number of tasks (n_tasks)

--------------------------------------------------
STEP 2: CHOOSE THEME
--------------------------------------------------

Each graph must be completely independent.

- DO NOT reuse themes or naming patterns
- Choose a distinct, realistic environment
- Infer theme from topology if not provided

--------------------------------------------------
STEP 3: RENAME REGIONS
--------------------------------------------------

Rename all regions using:

type_N format

Rules:
- All names globally unique (regions + objects)
- Each type prefix may appear at most TWICE
- Use multiple types per community if needed

--------------------------------------------------
STEP 4: RENAME OBJECTS
--------------------------------------------------

Rename objects using realistic names:

Examples:
pickup_truck_1, antenna_1, generator_1

Rules:
- Same uniqueness + prefix limits as regions
- Must match region context

--------------------------------------------------
STEP 5: FILL DESCRIPTIONS (STRICT)
--------------------------------------------------

Each object description is either:
- "__FILL__" → MUST replace with short attribute
- "" → MUST remain EXACTLY "" (DO NOT CHANGE)

Allowed values:
"damaged", "not damaged", "locked", "empty", "operational", etc.

If you modify "" → INVALID

--------------------------------------------------
STEP 6: GENERATE TASKS
--------------------------------------------------

Generate EXACTLY n_tasks tasks:

Each task:
{
  "task": "...",
  "answer": "...",
  "init_node": "...",
  "acceptance_criterion": "..."
}

Rules:

- NO node names in task text
- Must be solvable from graph
- Mix types:
  - existence
  - location
  - condition
  - reachability

Answer regex rules:
- Include synonyms
- Avoid ambiguous substrings
- No false positives
- Yes/no must match correct polarity only
- Do NOT anchor with ^ or $. Match anywhere in the response.

Acceptance criterion rules:
- ONE sentence describing what a correct planner response must convey.
- Reference the answer entity by its node name (e.g. fuel_depot_1) so an
  LLM judge can verify without re-solving the task.
- For yes/no tasks: state the correct polarity plus the supporting entity.
- Do NOT restate the task; describe the *answer*.
- The criterion is for offline grading ONLY. It will never be shown to the
  planner.
- Examples:
    "A correct answer identifies fuel_depot_1 as the region containing fuel_tank_1."
    "A correct answer affirms reachability and cites the path through storage_tent_1."
    "A correct answer is the number 2, the count of regions with exactly three neighbors."

--------------------------------------------------
STEP 7: UPDATE ROBOT LOCATION
--------------------------------------------------

Rename robot_location to new region name

--------------------------------------------------
STEP 8: REMOVE METADATA
--------------------------------------------------

Remove "_metadata" completely

--------------------------------------------------
STEP 9: VALIDATION (REQUIRED)
--------------------------------------------------

Before output, verify:

1. ALL coordinates EXACTLY unchanged
2. NO "" descriptions modified
3. NO "__FILL__" remains
4. All names unique globally
5. Each type prefix appears ≤ 2 times
6. All connections reference valid nodes
7. No original placeholder names remain
8. tasks length == n_tasks
9. robot_location valid
10. All init_node values valid
11. Every task has a non-empty acceptance_criterion that names the answer
    entity (or value, for counting tasks) by its node name.

If any condition fails → fix before output

--------------------------------------------------
OUTPUT FORMAT
--------------------------------------------------

Return ONLY valid JSON:

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

NO extra text.
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
