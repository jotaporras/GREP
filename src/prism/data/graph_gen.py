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


class TaskGraphGen:
    def __init__(self):
        self.client = utils.GPTQueryClient()  # OpenAI()

    def _build_prompt(
        self,
        base_graph: str,
        n_tasks: Optional[int] = 2,
        prior: Optional[str] = "",
        previous_tasks: Optional[str] = "",
    ):
        query = (
            QUERY
            + f"\nYour graph should populate the base gras provided below and you should generate {n_tasks} tasks.\nBase graph:\b{base_graph}"
        )

        if previous_tasks is not "":
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
            query=self._build_prompt(
                n_tasks=n_tasks,
                prior=description,
                base_graph=base_graph,
                previous_tasks=previous_tasks,
            )
        )

        # try to load the graph for error handling
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
