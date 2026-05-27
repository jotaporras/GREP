---
name: judge-eval
description: Use this whenever the user wants to judge, review, or grade an evaluation log from PRISM planning eval. Trigger on phrases like "judge eval", "grade the eval", "review eval results", "check eval accuracy", or when the user provides an eval log JSON path and wants assessment. Also trigger when the user says /judge-eval.
---

# LLM-as-Judge for PRISM Planning Eval Logs

You are an LLM-as-judge evaluator for PRISM planning evaluation logs.

## Input

The user provides a path to an eval log file (e.g. `eval_logs/step_001782_epoch_2.000.json`, `results.json`, or a `.txt` log file). Read that file.

**MANDATORY: Always read files using the Read tool with `offset` and `limit` parameters in multiple passes.** Never use the Agent tool to read files. Start with a small chunk (e.g., `limit: 5`) to understand line structure, then use Grep to find key fields (`correct answer`, `plan`, `reasoning`, `=====`), then use targeted Read calls with `offset`/`limit` to read specific sections around each sample. Files may be very large (>25k tokens) and contain multiple concatenated runs — identify the final/canonical run by finding the last occurrence of `Running eval on N samples...`.

The answer sheet and scene graphs are **not** embedded — read them from the run being judged (see **Reference: answer sheet & graphs** below).

## Eval log format

The eval log JSON has a `samples` array. Each sample has:
- `idx`: sample index
- `task`: the natural-language task given to the model
- `answer_key`: regex pattern used by the automated evaluator
- `response`: the model's JSON response dict (keys: `primary_goal`, `relevant_graph`, `reasoning`, `plan`), or null on error
- `interaction_trace`: list of multi-turn interaction steps (each has `request`, `response`, `explored_nodes`)
- `formatted`: bool — did the response have the correct JSON keys?
- `plan_keyword`: bool — did the `answer_key` regex match in the `plan` field?
- `correct`: bool — `formatted AND plan_keyword` (the callback's verdict)
- `error`: error string if the sample crashed, else null

## Your job

For each sample:

1. Read the `task` and the full `response` (especially the `plan` and `reasoning` fields). Review `interaction_trace` too — it shows the multi-turn exploration the model did.
2. Use the answer sheet below to understand intent. The `answer` field is a regex — understand what it's testing for.
3. **Make your own judgment**: Did the model actually accomplish the task? Consider:
   - Does the plan make logical sense for the task?
   - Did the model identify the right target object/location?
   - Is the reasoning sound?
   - Would this plan actually work in the simulated environment?
4. Compare the model's behavior against the **Acceptance Criterion** for each task (see answer sheet below).
5. **For navigation / path tasks** (any task whose correct answer must give a route), do NOT eyeball the graph — verify the planner's stated route programmatically with the Python check in **Path Verification** below.

## Output format

### Per-sample comparison table

One row per task. The Notes column must be brief enough to fit on one line.

| # | Task | Answer Key | Callback | Judge | Notes |
|---|------|-----------|----------|-------|-------|

Where:
- **#**: sample index
- **Task**: first ~50 chars of the task
- **Answer Key**: the regex pattern (use `OR` instead of `|` to avoid breaking markdown tables)
- **Callback**: PASS or FAIL
- **Judge**: PASS, PASS*, or FAIL
  - **PASS**: correct answer, no unnecessary traversals
  - **PASS***: correct answer but with unnecessary traversal(s) — add a footnote `*correct answer but with unnecessary traversal(s)` below the table
  - **FAIL**: wrong or missing answer
- **Notes**: one short phrase (≤10 words) — e.g. "regex too narrow", "correct reasoning bad format", "genuine fail"

### Summary table

| Metric | Callback | Judge |
|--------|----------|-------|
| Correct | X/N | X/N |
| Accuracy | X.X% | X.X% |
| Correct w/ unnecessary trav | — | X/N |

Do NOT add a separate disagreements section — all reasoning belongs in the Notes column of the table.

## Judgment guidelines

- Be strict but fair. A plan that reaches the right target is correct even if wording is imperfect.
- `formatted=false` means the model didn't produce valid JSON structure — that's a genuine failure, agree with FAIL.
- `formatted=true` but `plan_keyword=false` means the regex didn't match — but the model may have produced a semantically correct answer. Check carefully.
- If `response` is null and `error` is set, that's a crash — mark FAIL.
- If the interaction_trace shows the model explored correctly but the final response format was wrong, note "correct reasoning, bad format".
- Plans using `goto()`, `explore_region()`, `inspect()`, `answer()` actions are the expected format for this environment.

### Unnecessary traversal detection

Review the `interaction_trace` for each sample and flag any of the following as unnecessary traversals (mark Judge as PASS*):
- `map_region` called on a field node (e.g. `map_region(field_6)`) — field nodes return empty descriptions and yield no useful information
- `goto` to the wrong location (e.g. navigating to a field that doesn't contain the target object)
- `inspect` called on an irrelevant object not related to the task
- Any action that triggers a navigation warning about repeating the same call type

These are planning inefficiencies that waste interaction turns. They do not make a correct answer incorrect, but should be flagged.

## Path Verification (navigation tasks)

For any task whose correct answer must contain a route, do not judge the path by
eye — large scene graphs make eyeballing unreliable. Verify it with a short
read-only Python REPL. The planner's reported route is correct only if every
consecutive pair of regions is a real edge in the graph.

Steps:

1. Locate the scene graph for the sample — the `graph` block of the populated
   `data_gen_*.json` (or `graph_gen_*.json`) the sample came from, or the scene
   graph embedded in the sample's `interaction_trace`.
2. Transcribe, in order, the list of regions the planner names as its route.
3. Run this check in Bash via `python3`:

   ```python
   import json
   g = json.load(open("<path to the graph JSON>"))
   g = g.get("graph", g)   # data_gen_*.json nests the graph under "graph"
   adj = {}
   for a, b in g["region_connections"] + g["object_connections"]:
       adj.setdefault(a, set()).add(b)
       adj.setdefault(b, set()).add(a)

   path = ["region_a", "region_b", "region_c"]   # the planner's stated route

   broken = [(path[i], path[i + 1]) for i in range(len(path) - 1)
             if path[i + 1] not in adj.get(path[i], set())]
   print("all hops are real edges :", not broken)
   print("broken hops             :", broken)
   print("no repeated regions     :", len(path) == len(set(path)))
   print("start -> end            :", path[0], "->", path[-1])
   ```

4. Judge the path from the output:
   - **FAIL** — any hop is not a real edge (`broken` non-empty), or the route
     does not start at the task's `init_node` / end at the intended destination.
   - **PASS\*** — all hops are real edges and endpoints are correct, but a region
     repeats (the planner padded or looped the route).
   - **PASS** — all hops are real edges, endpoints correct, no region repeats.

This is informal verification: it confirms the *output* route is a genuine path.
It does not prove the task is optimally solvable — rigorous solvability checking
is handled separately.

## IMPORTANT RESTRICTIONS

When using this skill, you are NOT ALLOWED to make changes to any code or files (except writing the output if requested). You MAY run a read-only Python REPL (via `python3` in Bash) for the single purpose of parsing scene graphs and verifying planner-reported paths, as described in **Path Verification**. Do not run any other scripts, and do not modify anything. This is a read-only analysis task.

---

## Reference: answer sheet & graphs

The answer sheet and scene graphs are **not embedded** — they are specific to
whichever eval run you are judging. Read them from that run.

Every eval-log sample was generated from a task in a populated `data_gen_*.json`
file, under the run's `populated_graphs/` directory — e.g.
`data/gen/<run>/populated_graphs/data_gen_GGG.json`. Each task in that file
carries everything you need:

- `task` — the natural-language task (matches the sample's `task`)
- `answer` — the answer regex (matches the sample's `answer_key`)
- `init_node` — the region the robot starts in
- `acceptance_criterion` — the rubric to judge against (Step 4)
- the file's `graph` block — the scene graph for **Path Verification** (Step 5)

To build the answer sheet for a run: match each eval-log sample to its source
task by `task` text (or by `idx` order), then read that task's
`acceptance_criterion` and `graph`.

If the user has not said which run the eval log came from, ask for the run
directory (or its `populated_graphs/` path) before judging.
