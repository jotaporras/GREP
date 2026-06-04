---
name: fill-graph
description: Use this when the user wants to fill a graph skeleton with semantic content. Trigger on "/fill-graph <path>", "fill graph", "fill skeleton", "name the graph", or when the user provides a generated graph JSON and wants realistic names, descriptions, and tasks added.
---

# Semantic Fill for PRISM Eval Graph Skeletons

You transform a programmatically generated graph skeleton (from `scripts/generate_eval_graphs.py`) into a complete eval JSON that can be consumed by `python -m prism.eval.scalability_evaluation` and `src/prism/eval/evaluate.py`.

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

### Step 6: Generate tasks

Create exactly `n_tasks` tasks (from `_metadata`). Each task is:

```json
{
  "task": "natural language question (NO node names)",
  "answer": "pipe_separated|regex|pattern",
  "init_node": "renamed_region_name"
}
```

**Task requirements:**
- `task`: A natural language question or request. Must NOT contain any node names — the planner must discover them by exploring the graph.
- `answer`: A pipe-separated regex pattern that the planner's response must match. Keep patterns simple: key object names, yes/no, condition words.
- `init_node`: A valid renamed region name. Should usually be the robot's start location but can vary.
- Tasks must be **solvable from the graph structure** — they should reference objects/descriptions/locations that actually exist.
- Mix task types: existence queries ("is there a..."), location queries ("find me..."), condition queries ("is X damaged?"), reasoning queries ("why is...").

**Robust answer regex design — no false positives, no false negatives:**

The `answer` regex is matched against the planner's free-text response. A poorly designed regex will either miss correct answers (false negative) or accept wrong ones (false positive). Follow these rules:

1. **Include all synonyms and phrasings a correct response could contain.** If the target is a truck, use `truck|pickup` — not just `truck`. If the answer is "no", use `no|not|did not find|unable` to catch negative phrasings.
2. **Never use a regex term that could match an unrelated node or concept in the graph.** For example, if the graph contains both a `cabin` and a `cable_car`, do not use `cab` as an answer term — it would match both. Use the full, unambiguous name.
3. **For yes/no questions, the regex must only match the correct polarity.** A "is X damaged?" task where X is not damaged must have an answer like `no|not|did not find|unable` — never include `yes` or the object name alone, which would match a response that merely mentions the object without answering the question.
4. **For existence/location tasks, include the target object type as a term.** The planner's correct response will name the object, so the regex should match it (e.g., `boat|sailboat` for "is there a boat?").
5. **Test each regex mentally against two scenarios:** (a) a correct planner response — does it match? (b) a plausible but wrong response — does it accidentally match? If either fails, revise.

**Reference examples** (from `data/eval/eval_1_multi_step.json`):

| Task | Answer regex |
|------|-------------|
| is there a boat in the scene? | `yes\|boat\|sailboat` |
| Find me something I can drive in | `truck` |
| communications are down. Why? | `internet\|damaged` |
| Is the truck damaged? | `no\|i did not find\|not\|unable` |
| Find me a place to stay for the night? | `cabin` |

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

### Step 10: Write output

Write the final JSON to the same path (overwriting the skeleton) or to a user-specified path.

## Output Format

The output must match the schema of `data/eval/eval_1_multi_step.json` exactly:

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
    {"task": "...", "answer": "...", "init_node": "..."},
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
- Use the Read tool to read the file, the Write tool to write it
