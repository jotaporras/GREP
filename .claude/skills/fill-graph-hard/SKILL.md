---
name: fill-graph-hard
description: Use this when the user wants to fill a graph skeleton with HARD graph-reasoning tasks only (no node existence). Trigger on "/fill-graph-hard <path>", "fill graph hard", "hard fill", or when the user wants tasks that require graph structure understanding (positionality, reachability, navigability).
---

# Semantic Fill for PRISM Eval Graph Skeletons — Graph Reasoning Only

You transform a programmatically generated graph skeleton (from `scripts/generate_eval_graphs.py`) into a complete eval JSON that can be consumed by `python -m prism.eval.scalability_evaluation` and `src/prism/eval/evaluate.py`.

**This skill generates ONLY graph-reasoning tasks.** Every task must require understanding of graph topology (connections, paths, adjacency, spatial layout) to solve. Node existence tasks ("is there a boat?", "is X damaged?") are BANNED — they test semantic matching, not graph understanding, and are false positives for graph reasoning evaluation.

## Input

The user provides a path to a skeleton JSON file and optionally a scene theme (e.g., "rural farmland", "urban campus", "disaster zone"). Read that file.

The skeleton has this structure:

```json
{
  "graph": {
    "objects": [{"name": "object_1", "coords": [...], "description": "" | "__FILL__"}, ...],
    "regions": [{"name": "region_1", "coords": [...], "description": ""}, ...],
    "object_connections": [["object_1", "region_3"], ...],
    "region_connections": [["region_1", "region_2"], ...],
    "robot_location": "region_1"
  },
  "tasks": [],
  "_metadata": {
    "n_communities": 3,
    "nodes_per_community": 5,
    "community_assignments": {"region_1": 0, "region_2": 0, ...},
    "n_tasks": 10,
    "seed": 42,
    ...
  }
}
```

## Procedure

### Step 1: Read and understand

Read the skeleton JSON. Extract `_metadata` to understand:
- How many communities exist and which regions belong to each
- How many tasks to generate (`n_tasks`)
- The graph topology (region_connections) and object placement (object_connections)

### Step 2: Choose theme

**Independence rule:** Each graph must be filled completely independently. Do NOT reuse themes, region types, object types, naming patterns, or task wordings from any previously filled graph in this conversation or any other. Treat each skeleton as if it is the only one you have ever seen. Choose a fresh, distinct theme every time.

If the user provides a theme, use it. Otherwise, infer a coherent theme from the topology (e.g., a graph with 3-4 communities of 5-10 regions suggests a rural area with distinct zones like fields, roads, and wooded areas). Vary your theme choices widely — do not default to the same genre (e.g., rural farmland) across multiple invocations.

### Step 3: Rename regions

Each community gets a coherent region type. Names follow the `type_N` convention with **globally unique names across all regions AND objects**.

**Uniqueness rule:** Every node name in the entire graph must be unique. A name like `field_1` may only appear once — it cannot be both a region and an object. Each `type` string (the prefix before `_N`) may appear at most **twice** in the entire graph (regions + objects combined). For example, you may have `desert_1` and `desert_2`, but not `desert_3`. If a community has more nodes than 2, use multiple distinct types within that community (e.g., `field_1`, `field_2`, `meadow_1`, `meadow_2`, `clearing_1`).

Examples of community → type mappings:
- Community 0 (5 nodes) → `field_1`, `field_2`, `meadow_1`, `meadow_2`, `clearing_1`
- Community 1 (5 nodes) → `road_1`, `road_2`, `highway_1`, `intersection_1`, `parking_lot_1`
- Community 2 (5 nodes) → `trail_1`, `trail_2`, `path_1`, `path_2`, `bridge_1`

Choose types that make sense for the theme. Types used across different communities should be distinct where possible. Maintain a rename map: `{"region_1": "field_1", "region_2": "meadow_1", ...}`.

### Step 4: Rename objects

Give objects realistic names matching their host region context. Use the `type_N` convention.

Examples: `pickup_truck_1`, `cabin_1`, `shed_1`, `light_pole_1`, `sail_boat_1`, `internet_tower_1`.

**Uniqueness rule (same as regions):** Each object name must be globally unique across all regions AND objects. Each `type` prefix may appear at most **twice** in the entire graph. Maximize diversity — avoid repeating the same type for every object. Consider what makes sense near each region type.

### Step 5: Fill descriptions

For objects with `description: "__FILL__"`, assign short attribute strings that create interesting planning scenarios:
- `"damaged"`, `"not damaged"`, `"has keys"`, `"locked"`, `"empty"`, `"operational"`

For objects with `description: ""`, leave them as `""`.

### Step 6: Generate tasks — GRAPH REASONING ONLY

Create exactly `n_tasks` tasks (from `_metadata`). **Every task must be one of these three graph-reasoning types — NO node existence tasks.**

Each task is:

```json
{
  "task": "natural language question (NO node names)",
  "answer": "regex pattern with \\b word boundaries",
  "init_node": "renamed_region_name",
  "acceptance_criterion": "one sentence describing what a correct response must convey"
}
```

#### Banned task type: Existence

Do NOT generate any task that can be answered by simply confirming an object or entity exists, checking its condition/status, or determining if something is present. These are semantic matching tasks, not graph reasoning tasks. They are false positives when evaluating graph understanding.

**Banned patterns include:**
- "Is there a boat in the scene?"
- "Is X damaged/locked/operational?"
- "Does the complex include a medical supply kit?"
- "Is there something I can use to catch fish?"
- "What condition is reported for the fire engine?"
- "Is there a way to communicate with the surface?"
- Any task answerable without traversing edges or understanding adjacency

#### Allowed task types

**1. Positionality** — understand the position of a node within the graph structure: what contains what, spatial relationships, counting co-located items, finding items by graph-structural properties.

These tasks require understanding where nodes sit in the graph topology — containment, co-location, neighborhood size, or spatial extremes. The answer depends on graph structure, not just whether something exists.

Simple examples:
- "Where is the satellite phone stored?"
- "Which area contains the fuel tank?"
- "I lost my keys. I last saw them when I parked my truck."

Complex examples (preferred — these truly test graph understanding):
- "Name the item in the airfield avionics work area that is marked operational."
- "Which other item is stored in the same area as the charger bank?"
- "Where are the charger bank and spare battery kept together?"
- "Which area contains both the fuel filter case and the damaged battery crate?"
- "Which area is the only one with two listed objects in the same area?"
- "Among all areas, which area has the largest number of immediate neighbors?"
- "How many areas have exactly three listed items?"
- "How many areas contain exactly two objects?"
- "Which workroom holds the microscope, sample freezer, and acoustic recorder together?"
- "How many listed items have a filled-in condition description?"
- "Which area holds both a ration supply and a satellite phone?"

**2. Reachability** — reach a target (often a short hop away) and SHOW the connecting edge(s) and the route.

These tasks require understanding the edge structure of the graph. The planner must reach the target and report the connecting edge(s) as `A <-> B` and the route taken as `A -> B -> C` — **never a bare yes/no**.

Simple examples:
- "Reach the cold storage area from the parking pay station and give the connecting edge and route."
- "Show how the pump house connects to the radar pier and list the path."
- "Get from the main harbor plaza to the chart archive and report the route."

Complex examples (preferred — these combine routing with object/attribute references):
- "From the southern aircraft service apron, reach the weather-check balloon and list the edges traversed."
- "Route from the waterfront pier to the remote cache and give each connecting edge."
- "Reach the damaged meteorological balloon from the starting area and show the path."
- "From the starting work bay, reach the area with the tool chest and give the route."
- "Starting where the welding cart is kept, reach the area with the fuel meter and list the connecting edges."
- "From the command post, route to the place with the ambulance and show the path."
- "From the medical communications hut, give the route to the landing deck."

**3. Navigability** — multi-hop routing: give the full route from one area to another.

Phrase EVERY Navigability task as a request for the route — "Give the route from <A> to <B>" — never as a yes/no. Choose start, end, and any constraint so that a route DOES exist.

**Complex Navigability tasks SHOULD add a constraint** — a required waypoint ("by way of ...", "passing through ...") or an avoided area ("without passing through ..."). Favor constrained variants.

Simple examples:
- "Is there a route from the starting area to the area with the air purifier? If so, give it."
- "Can the robot get from the dockyard gate to the oxygen cart? Provide the route."
- "Is there a path from the crew quarters to the coolant service point? List the route."

Complex examples (preferred — these add waypoint/avoidance constraints or structural queries):
- "Is there a route from the ice-core store to the locked satellite phone that passes through the communications bunker? If so, give it."
- "Can the robot reach the spectrometer lab from the cargo-pallet area without going through the main command area? Provide the route."
- "Is there a route from the waste-processing area to the medical lab by way of the crew recreation area and the coolant station? Give the route."
- "Give a valid area sequence from the animal holding area to the waterfront platform that avoids the supply building."
- "From the current starting point, can the hydraulic jack be reached by going through the fuel yard and then the parts storage area?"
- "Starting at the place with the cargo pallet, can the robot reach the laboratory with the spectrometer without using the main command area?"
- "From the freezer aisle, can the robot get to the damaged lidar device by crossing sector 164 and sector 161?"
- "Which intermediate area gives a two-step route from the starting area to the quadcopter's area?"
- "From the robot's starting area, give the two-hop route to the sample quay that uses the drone staging apron as the middle stop."
- "Which area is directly connected to the battery shed, the fuel quay, and the control bunker?"
- "Which reserve entry area is the only gateway between the facility side and the habitat loop?"
- "Which non-entry wetland habitat area directly links both the viewing shelter and the nesting patch?"

#### Task distribution

Distribute the `n_tasks` tasks across the three allowed types as follows:
- **Positionality:** ~30% of tasks
- **Reachability:** ~30% of tasks
- **Navigability:** ~40% of tasks

For `n_tasks = 10`: 3 positionality, 3 reachability, 4 navigability.

**Difficulty mix:** Generate tasks with a **1:1 ratio of simple to complex** (50% simple, 50% complex) within each type. Complex tasks are strongly preferred — they are the ones that truly test graph understanding.

#### Task field rules

- `task`: A natural language question or request. Must NOT contain any node names — the planner must discover them by exploring the graph.
- `answer`: A node-naming regex (NEVER a bare yes/no). **Must use `\b` word boundaries** (e.g. `(?i)\bfuel_depot_1\b`, or `(?i)\bclearing_1\b.*\bfuel_depot_1\b` for a route). See regex rules below.
- `init_node`: A valid renamed region name.
- `acceptance_criterion`: One sentence describing what a correct response must convey. Reference the answer entity by its node name so an LLM judge can verify.

#### Answer regex rules (grading is deterministic over the graph)

The deterministic grader reads the goal region, waypoints, avoided areas, and the containment edge(s) from the `acceptance_criterion` (and this `answer`), then validates the planner's stated edges and route against the NetworkX graph. The `answer` regex is only a coarse parallel signal — keep it SIMPLE and **node-naming, never a bare polarity**.

- Wrap every node name in `\b` word boundaries: `(?i)\bfuel_depot_1\b`, never bare `fuel_depot_1`.
- **NEVER use a yes/no regex** for Positionality, Reachability, or Navigability — they are graded on edges and paths, not polarity.
- **Positionality** — match the answer (destination) region name: `(?i)\bfuel_depot_1\b`.
- **Reachability / Navigability** — match only the route's FIRST and LAST hop (start region then destination), in order: `(?i)\bcrew_quarters_1\b.*\bcoolant_station_1\b`. Include a required waypoint when the task names one: `(?i)\bcrew_quarters_1\b.*\bpower_conduit_1\b.*\bcoolant_station_1\b`. NEVER encode the full path — the full walk is checked deterministically.
- Do NOT anchor with `^` or `$`. The regex should match anywhere in the response.

#### Acceptance criterion rules (these drive the deterministic grader)

- Write ONE sentence describing what a correct planner response must convey.
- Name, by node id, the destination region (the goal), every required waypoint (write "via <node>" / "passing through <node>"), every avoided area (write "without using <node>"), and each object whose containment is part of the answer — so the grader resolves the goal, constraints, and required `region <-> object` edges deterministically.
- An avoided area must be a REAL region that is neither the start nor the destination and that the intended route genuinely bypasses. Never write a vacuous constraint: do not reference a non-existent edge, do not hedge ("if it existed"), and do not name the start/goal/waypoint in the "without using" clause.
- Only write "via <node>" / "passing through <node>" when the TASK text itself imposes that waypoint. For an UNCONSTRAINED reach/route task, require only "a valid route" — do NOT name a specific intermediate region and do NOT spell out a full example path (e.g. "the route a -> b -> c") in the criterion. Many valid routes exist; the deterministic grader accepts any valid walk to the goal, and a baked-in intermediate wrongly fails a correct alternative.
- For Reachability/Navigability: name the start region, the destination region, and require a valid route between them (plus any waypoint/avoid), with the connecting edges. For Positionality: name the region and its contained object(s) so the containment edge is required.
- Do NOT restate the task; describe the *answer*.

Examples:
- "A correct answer identifies fuel_depot_1 as the region containing fuel_tank_1, stating the edge fuel_depot_1 <-> fuel_tank_1."
- "A correct answer gives a route from clearing_1 to fuel_depot_1 and lists each connecting edge."
- "A correct answer routes from clearing_1 to fuel_depot_1 via comm_bunker_1 and shows the full path."
- "A correct answer routes from decon_chamber_1 to quarantine_bay_1 passing through supply_corridor_1 and ventilation_hub_1, without using the blocked area, and shows the path."
- "A correct answer identifies driveway_1 as the northmost region in the scene."

### Step 7: Update robot_location

Replace `"robot_location": "region_1"` with the renamed version of region_1.

### Step 8: Strip _metadata

Remove the `_metadata` key from the output.

### Step 9: Validate

Before writing, verify:
1. All names in `object_connections` and `region_connections` exist in the `objects` and `regions` lists
2. No placeholder names remain (`region_N`, `object_N`)
3. No `__FILL__` descriptions remain
4. `tasks` has exactly `n_tasks` entries
5. `robot_location` is a valid region name
6. Every task's `init_node` is a valid region name
7. **Every node name is globally unique** — no name appears more than once across all regions and objects
8. **No type prefix appears more than twice** — count occurrences of each `type` (the part before `_N`) across all node names; none may exceed 2
9. **Reachability:** BFS from `robot_location` over `region_connections` must visit every region. Every object's host region (from `object_connections`) must be in the visited set. Every task's `init_node` must also be in the visited set. If any node is unreachable, flag it as an error — do NOT write the output.
10. **Task solvability:** For each task, verify that at least one object or region matching the `answer` regex is reachable from that task's `init_node` via BFS over `region_connections`. If a task is unsolvable, fix or replace it before writing.
11. **NO EXISTENCE TASKS:** Re-read every task and confirm it requires graph-structural reasoning (adjacency, paths, containment-within-topology, spatial layout). If any task can be answered by simply confirming something exists or checking its condition without traversing the graph, replace it with a graph-reasoning task. This is the critical validation — existence tasks are false positives for graph understanding evaluation.
12. **All `init_node` values for Positionality tasks must NOT be the answer region itself** — the planner must traverse the graph to find the answer, not read it off its starting observation. For Reachability and Navigability tasks, `init_node` may be one of the endpoints referenced by the task.
13. **No yes/no answers:** every `answer` regex uses `\b` word boundaries and NAMES NODES — never a bare `\byes\b`/`\bno\b`. Positionality matches the destination region; Reachability/Navigability match only the first and last hop (plus any required waypoint), never a full path.
14. **Every task has a non-empty `acceptance_criterion`** that names, by node id, the goal region, every waypoint/avoided area, and any contained object — so the deterministic grader can resolve the goal, constraints, and required `region <-> object` edges.

### Step 10: Write output

Write the final JSON to the same path (overwriting the skeleton) or to a user-specified path.

## Output Format

The output must match this schema exactly:

```json
{
  "graph": {
    "objects": [...],
    "regions": [...],
    "object_connections": [...],
    "region_connections": [...],
    "robot_location": "field_1"
  },
  "tasks": [
    {"task": "...", "answer": "...", "init_node": "...", "acceptance_criterion": "..."},
    ...
  ]
}
```

This file is directly consumable by:
- `python -m prism.eval.scalability_evaluation --checkpoint <checkpoint> --graphs <this_file>`
- `src/prism/eval/evaluate.py` via `EvalSample` (build with `evaluate.eval_samples_from_dict`)

## IMPORTANT RESTRICTIONS

- Do NOT add any keys beyond what's in the reference schema
- Do NOT modify coordinates — they encode spatial structure from the SBM
- Do NOT add or remove nodes or edges — only rename them
- Task text must NEVER contain node names
- All region/object names must be globally unique — no duplicates across the entire graph
- Each type prefix (e.g., `field`, `cabin`) may appear at most twice across all nodes (regions + objects)
- Each graph fill must be **semantically independent** — never carry over themes, naming patterns, object types, or task styles from a previous fill
- **ABSOLUTELY NO NODE EXISTENCE TASKS** — every task must require graph-structural reasoning to solve. A task that can be answered by "yes it exists" or "it is damaged" without understanding connections/paths/adjacency is INVALID for this skill.
- Use the Read tool to read the file, the Write tool to write it
