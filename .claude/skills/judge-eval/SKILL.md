---
name: judge-eval
description: Use this whenever the user wants to judge, review, or grade an evaluation log from PRISM planning eval. Trigger on phrases like "judge eval", "grade the eval", "review eval results", "check eval accuracy", or when the user provides an eval log JSON path and wants assessment. Also trigger when the user says /judge-eval.
---

# LLM-as-Judge for PRISM Planning Eval Logs

You are an LLM-as-judge evaluator for PRISM planning evaluation logs.

## Input

The user provides a path to an eval log file (e.g. `eval_logs/step_001782_epoch_2.000.json`, `results.json`, or a `.txt` log file). Read that file.

**MANDATORY: Always read files using the Read tool with `offset` and `limit` parameters in multiple passes.** Never use the Agent tool to read files. Start with a small chunk (e.g., `limit: 5`) to understand line structure, then use Grep to find key fields (`correct answer`, `plan`, `reasoning`, `=====`), then use targeted Read calls with `offset`/`limit` to read specific sections around each sample. Files may be very large (>25k tokens) and contain multiple concatenated runs — identify the final/canonical run by finding the last occurrence of `Running eval on N samples...`.

You do NOT need to re-read the answer sheet or the graph — both are embedded below as reference.

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
4. Note whether you **agree or disagree** with the callback's `correct` verdict.

## Output format

### Per-sample comparison table

One row per task. The Notes column must be brief enough to fit on one line.

| # | Task | Answer Key | Callback | Judge | Agree? | Notes |
|---|------|-----------|----------|-------|--------|-------|

Where:
- **#**: sample index
- **Task**: first ~50 chars of the task
- **Answer Key**: the regex pattern
- **Callback**: PASS or FAIL
- **Judge**: PASS or FAIL
- **Agree?**: Y or N
- **Notes**: one short phrase (≤10 words) — e.g. "regex too narrow", "correct reasoning bad format", "genuine fail"

### Summary table

| Metric | Callback | Judge |
|--------|----------|-------|
| Correct | X/N | X/N |
| Accuracy | X.X% | X.X% |
| Agreement | — | X/N (X.X%) |

Do NOT add a separate disagreements section — all reasoning belongs in the Notes column of the table.

## Judgment guidelines

- Be strict but fair. A plan that reaches the right target is correct even if wording is imperfect.
- `formatted=false` means the model didn't produce valid JSON structure — that's a genuine failure, agree with FAIL.
- `formatted=true` but `plan_keyword=false` means the regex didn't match — but the model may have produced a semantically correct answer. Check carefully.
- If `response` is null and `error` is set, that's a crash — mark FAIL.
- If the interaction_trace shows the model explored correctly but the final response format was wrong, note "correct reasoning, bad format".
- Plans using `goto()`, `explore_region()`, `inspect()`, `answer()` actions are the expected format for this environment.

## IMPORTANT RESTRICTIONS

When using this skill, you are NOT ALLOWED to: make changes to any code or files (except writing the output if requested), or run any scripts. This is a read-only analysis task.

---

## Reference: Answer Sheet (`data/eval/eval_1_multi_step.json`)

All tasks start at `init_node: field_1` (robot start location).

| # | Task | Answer regex |
|---|------|-------------|
| 0 | is there a boat in the scene? | `yes\|boat\|sailboat` |
| 1 | Find me something I can drive in | `truck` |
| 2 | I lost my keys. I last saw them when I parked my truck. | `truck` |
| 3 | communications are down. Why? | `internet\|damaged` |
| 4 | Is the light poles damaged? | `no\|i did not find\|not\|unable` |
| 5 | Is the truck damaged? | `no\|i did not find\|not\|unable` |
| 6 | Is the boat damaged? | `no` |
| 7 | Find me a place to stay for the night? | `cabin` |
| 8 | Where are tools likely to be? | `shed` |
| 9 | Can I cross the bridge? | `yes\|i did not find\|not` |

---

## Reference: Ground Truth Graph

### Topology

```
         [field_2]
        /    |    \
  [field_1]  |  [field_3]
   🤖        |    /
        \    |   /
         [field_5]──[field_4]
              │
          [field_6] ← 🚛 pickup_truck_1 (has keys), 🏠 cabin_1
              │
          [field_7] ← ⛵ sail_boat_1 (not damaged)
              │
          [field_8] ← 🏠 cabin_2
              │
          [field_9] ← 📡 internet_tower_2 (damaged), 💡 light_pole_1
              │
         [field_10]
              │
         [field_11]
              │
         [field_12]
              │
          [bridge_1] (not damaged)
              │
         [field_13] ← 🏚️ shed_1
              │
         [field_14]
              │
         [field_15]
              │
          [road_1]
           /     \
     [field_17]  [road_2]
```

### Objects

| Object | Location | Description |
|--------|----------|-------------|
| pickup_truck_1 | field_6 | has keys |
| cabin_1 | field_6 | — |
| sail_boat_1 | field_7 | not damaged |
| cabin_2 | field_8 | — |
| internet_tower_2 | field_9 | **damaged** |
| light_pole_1 | field_9 | — |
| shed_1 | field_13 | — |

Key structure: fields 1–5 form a loop with shortcut (field_2↔field_5); beyond field_6 is a strict linear chain ending in a fork at road_1.
