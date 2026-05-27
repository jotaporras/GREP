---
name: verify-tasks
description: Use this when the user wants to audit generated training-data tasks for solvability and answer-regex correctness. Trigger on "/verify-tasks", "verify tasks", "audit tasks", "check task quality", "are the tasks solvable", "check generated data", or when the user provides a path to populated graphs and wants quality assurance before running rollouts.
---

# Task Quality Auditor for PRISM Training Data

You verify that LLM-generated tasks in PRISM training data are (a) **solvable** from the accompanying graph and (b) **verifiable** by the supplied answer regex. Use this BEFORE expensive SPINE planner rollouts to catch ground-truth labeling bugs.

## Input

The user provides one of:

- A path to a directory containing `data_gen_*.json` files (e.g. `data/gen/spine_single_rollout/populated_graphs/` or `data/training_data_graphs_20260422/grep_training_data/graphs/`).
- A single `data_gen_*.json` file path.

Each `data_gen_*.json` is the LLM-populated output of Phase 1 from `scripts/training_data_generation/generate_data_spine.py`. Schema:

```json
{
  "graph": {
    "objects": [{"name": "...", "coords": [...], "description": "..."}],
    "regions": [{"name": "...", "coords": [...], "description": "..."}],
    "object_connections": [["object_name", "region_name"], ...],
    "region_connections": [["region_a", "region_b"], ...],
    "robot_location": "region_X"
  },
  "tasks": [
    {"task": "<natural-language question>",
     "answer": "<python regex>",
     "init_node": "<starting node>"}
  ],
  "description": "..."   // optional
}
```

## Procedure

### 1. Sample

**Sample size.** If invoked as `/verify-tasks <path> --sample-size N`, use N as the sample size; otherwise default to **5**.

If given a directory, list `data_gen_*.json` files and randomly sample up to the configured size. Use a shuffled random sample, not the first N. A simple way:

```bash
SAMPLE_SIZE="${SAMPLE_SIZE:-5}"
ls <dir>/data_gen_*.json | shuf -n "$SAMPLE_SIZE"
```

If fewer than N exist, audit all of them. If given a single file, audit that one.

Report the sampled file paths up front so the user knows which were inspected.

### 2. Per-task checks

For each sampled file, iterate over every task and run all checks below. Do NOT skip tasks — every task in every sampled file is audited.

#### Solvability checks

The task must be answerable from the graph (possibly via `inspect` / `map_region` / `goto` traversal — partial observability is fine, but the **information must exist somewhere in the ground-truth graph**).

- **S1. Init node exists.** `init_node` is in `regions` (or, rarely, an object). If not → **FAIL**.
- **S2. Init node connected.** A path exists from `init_node` to every node referenced in the canonical answer using `region_connections` ∪ `object_connections`. If unreachable → **FAIL**.
- **S3. Answer entity exists in graph.** For tasks of form:
  - existence ("is there an X") → the graph contains an entity whose name or description matches X.
  - location ("where is X") → X is in the graph AND has at least one connection.
  - condition / attribute ("is X damaged", "is X locked") → X exists AND the relevant attribute is present in X's `description`. If the attribute isn't in any description, the planner can't observe it → **FAIL** (unless the task is asking about absence and "no" is the correct answer — note in the report).
  - reachability / navigation ("can I get from A to B") → A and B exist; existence of path determines the yes/no answer.
- **S4. No phantom entities.** The task wording (rephrased generically per STEP 6 rules) must not require entities that don't appear in the graph. Example: "Find the helicopter" when no helicopter is in objects → **FAIL**.

#### Verifiability checks

The `answer` regex must correctly classify a planner's natural-language final answer.

- **V1. Regex compiles.** Try compiling with `re.compile(pattern, re.IGNORECASE)`. If invalid → **FAIL**.
- **V2. Canonical answer passes.** Construct a plausible canonical correct answer sentence in natural language (1–2 sentences mentioning the target entity by name). Verify `re.search(pattern, canonical, re.IGNORECASE)` is truthy. If not → **FAIL** (regex too narrow).
- **V3. Premise-only does NOT pass.** Construct a "premise echo" — a sentence that restates the question and mentions only the entities the question describes, but does NOT identify the answer entity. Verify the regex does NOT match. If it does → **FAIL: answer leak / regex matches premise**. (This is the bug pattern we found in task 5 / task 9: "Name a region connected to both A and B" with regex `\bA\b|\bB\b` — premise-only passes.)
- **V4. Wrong polarity does NOT pass** (yes/no questions). Construct the wrong-polarity answer ("No, there is no fuel storage" when correct is yes). Verify the regex does NOT match. If it does → **FAIL: polarity error**.
- **V5. Multiple-correct enumeration.** If the graph admits multiple valid answers (e.g., several connectors), check the regex matches *each* of them via alternation. If only one is hard-coded → **WARN: regex too narrow for valid alternatives**.
- **V6. No spurious substring matches.** Word boundaries (`\b`) around node names. If `fuel_depot_1` is bare in the regex (no `\b`), it can match inside `fuel_depot_10` etc. → **WARN**.

### 3. Verdict

Each task gets one of:
- **PASS** — solvable and verifiable.
- **WARN** — solvable + answer correct, but regex is brittle (narrow / no word boundaries / similar). Usable but flag for cleanup.
- **FAIL** — unsolvable OR regex misclassifies (leak, wrong polarity, doesn't match canonical, references phantom entity).

## Output format

### Per-file table

One section per sampled file:

```
### File: data_gen_017.json  (8 tasks)

| # | Task (first ~60 chars)          | Type        | Init node       | Verdict | Notes                                      |
|---|---------------------------------|-------------|-----------------|---------|--------------------------------------------|
| 0 | Is there any fuel storage in... | existence   | central_corr_1  | PASS    |                                            |
| 1 | Name a region connected to b... | location    | cargo_bay_1     | FAIL    | V3: regex `A\|B` matches premise           |
| 2 | Can a robot reach any cargo...  | reachability| control_room_1  | PASS    |                                            |
| ... |                               |             |                 |         |                                            |
```

Keep Notes to a single short phrase; if multiple issues, comma-separate (e.g., `V3 leak, V6 no \b`).

### Summary

```
| Metric              | Count |
|---------------------|-------|
| Files audited       | 5     |
| Total tasks         | 47    |
| PASS                | 38    |
| WARN                | 5     |
| FAIL                | 4     |
| FAIL: regex leak    | 3     |
| FAIL: unreachable   | 1     |
| FAIL: phantom node  | 0     |
| FAIL: regex broken  | 0     |
| FAIL: bad polarity  | 0     |
```

### Recommended actions

A short bulleted list of what to fix in the data generator prompt or post-process before running rollouts. Only include items actually triggered by the audit.

## Implementation notes

- Use the **Read** tool to load each JSON file. Don't load all at once — read one file at a time, run all checks for it, emit the table row, then move on.
- For regex testing (V1–V6), use **Bash** with a short python heredoc (`python3 -c '...'`) so the actual regex behavior is verified, not just inferred from inspection. Example:
  ```bash
  python3 -c "import re; print(bool(re.search(r'\bA\b|\bB\b', 'Yes, A and B are both endpoints', re.IGNORECASE)))"
  ```
- For reachability (S2), run a quick BFS over `region_connections + object_connections` in a python heredoc — do not eyeball topology for non-trivial graphs.
- Do NOT attempt to *fix* the tasks — this skill is read-only audit. Recommend fixes in the final section.

## Restrictions

- Read-only. Do not modify any `data_gen_*.json` files.
- Do not run SPINE planner or generate new data — those are separate scripts.
- If sampled files are very large (>20 KB), still audit every task; do not skip for size.
