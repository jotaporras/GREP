# e20 — path-only prediction (design + runbook)

**Date:** 2026-08-24 · **Tag:** `e20_path_only_pred` · **Status:** implemented, generation jobs pending

## Hypothesis

Adviser's suspicion: the reasoning text in the distillation targets (the
`<think>Relevant graph: … Reasoning: …</think>` scaffold) is *confusing* the
graph-augmented model — the graph channel should carry the structure, and the
LLM's job should reduce to emitting the route. e20 strips the targets down to
**strictly path prediction**: the training answer is the bare node sequence
`a -> b -> c`, nothing else, and the eval prompt asks for exactly that.

## Parse-vs-regenerate (the opening question)

Answer: **both, because they answer different questions — and parsing is nearly
free.** The compact translator already unwraps every stored plan to a bare
route at train time; the think block is the only extra part. So "convert the
existing corpus" is not a new dataset at all — it is a **training-time flag**
(`data.response_format=route_only`) that renders any existing corpus's targets
as bare routes (samples whose final plan has no extractable route are dropped,
loudly). That conversion, however, cannot answer the *teacher* question the
task also asks (does route-only no-think prompting degrade Gemma-31B?) and it
inherits routes that were produced *with* reasoning. Hence:

- `route_only` conversion of the original v3 corpus = a free control arm
  (same routes, formatting-only change), available whenever we want it;
- **e20a** regenerates the *responses* path-only on the SAME populated graphs
  (`--skip-populate`), which costs no populate pass, keeps task coverage
  identical, and yields the paired teacher-accuracy comparison;
- **e20b** drops the teacher entirely (oracle routes) at 3× scale.

## Datasets

| corpus | source | teacher | size |
|---|---|---|---|
| `n_60_pathonly_v1` (e20a) | v3 populated graphs (52 train + 7 frozen test), 12 tasks/graph | Gemma-4-31B, ONE route-only **no-think** query per task (`--path-only`) | = v3 × teacher pass rate |
| `n_60_oracle_v1` (e20b) | 156 FRESH skeletons (seeds 401-556), vLLM populate, 12 tasks/graph | **none** — NetworkX ground-truth routes (`--oracle-paths`) | ~1.9k samples ≈ 3× v3 |

Both keep the SPINE rollout file shape (`sample_GGG_TTT.json` message lists,
`task: …Scene graph:{…}` user turn, 4-key JSON answer with
`plan="[answer(a -> b -> c)]"`), so `strip_icl` / `aggregate` /
`split_train_val` / `compact_prompt` are unchanged. Wrong or unparseable routes
are quarantined as `*_failed.json` (same protocol as SPINE no-answer rollouts);
the grader is the eval scorer itself (`path_validator.validate_structured`,
`full_response=None` — deterministic, judge-free). Oracle routes are shortest
paths through each task's waypoints with avoided nodes removed
(`derive_targets` supplies goal/waypoints/avoid — the same resolver the grader
uses), self-verified through the scorer before commit.

## Teacher-accuracy monitor (explicit ask)

`generated_plans/rollout_stats.json` (written by any `--path-only` /
`--oracle-paths` run): per-graph pass/fail + failure reasons
(`no_route` / `wrong_route` / `no_goal_resolved` / `generation_error`).
The paired baseline: `scripts/e20_grade_rollouts.py` runs the SAME grader over
the original v3 reasoning rollouts (SPINE `*_failed.json` counted as
`spine_no_answer`), writing `reasoning_grade_stats.json`. Compare the two pass
rates; a large drop on the path-only side = route-only no-think prompting
significantly reduces the base model's accuracy (report per-graph, since task
mix is identical per graph in e20a). `PATH_ONLY_THINKING=1` reruns the
distillation with thinking ON (stripped from the target) to separate
"no-think" from "route-only contract" if the drop is large.

## New axis: `data.response_format` (think_route | route_only)

- `think_route` (default): byte-identical no-op — locked by tests.
- `route_only`: system prompt asks for the bare arrow route and nothing else;
  the SFT target is the route alone. Requires `spine_tools=none`,
  `icl_examples=0`, and eval tools OFF (`PRISM_DISABLE_SPINE_TOOLS=1`) —
  validated fail-loud at config, preprocess, and client-build time. Recorded in
  `train_config.json` (both trainers) and **load-bearing at eval rebuild**:
  `checkpoint.resolve_response_format` (missing key = `think_route`, the exact
  historical value) → threaded through `EvalCallback` /
  `evaluate.eval_model_*` / `evaluate_model` / both HF and vLLM client stacks /
  `scalability_evaluation`. Grading is unchanged: the model's bare route is
  wrapped back to `[answer(route)]` by `compact_output_to_spine_json`
  (pre-existing behavior), so `eval/accuracy` remains the same objective
  RegEx/NetworkX metric.

Generator flags (`generate_data_spine.py`): `--path-only`,
`--path-only-thinking`, `--oracle-paths`; teacher clients
(`VLLMSpineClient`/`GemmaSpineClient`) grew `enable_thinking` (default True —
the original datasets and spine datasets generate exactly as before;
`EXTRA_GEN_ARGS` in the driver is empty by default).

## Fleet (betty dgx-b200; e19 protocol: lr 2.5e-4, 3 epochs, per-epoch eval)

| run | arm | data | submit |
|---|---|---|---|
| `e20_pathonly_hop_depth`  | hop_depth (best e19: 73.8%) | e20a | `ARM=hop_depth  DATASET=pathonly sbatch scripts/e20_path_only_sft.sbatch` |
| `e20_pathonly_text_edges` | LLM baseline | e20a | `ARM=text_edges DATASET=pathonly …` |
| `e20_oracle_hop_depth`    | hop_depth | e20b | `ARM=hop_depth  DATASET=oracle …` |
| `e20_oracle_text_edges`   | LLM baseline | e20b | `ARM=text_edges DATASET=oracle …` |

All four: `data.response_format=route_only`, `PRISM_DISABLE_SPINE_TOOLS=1`,
eval **always on the frozen n60_v3 test graphs** (84 questions) → directly
comparable to hop_depth 73.8 / cn7ub88q 72.6 / mask_a 56.0. Generation first:
`sbatch scripts/e20_n60_pathonly_generate.sbatch` and
`sbatch scripts/e20_n60_oracle_generate.sbatch`.

## Decision rules

- Primary: `eval/accuracy` on the frozen test graphs vs the think_route
  hop_depth/cn7ub88q line, remembering the ±10 pt single-run noise band —
  plus hallucinated-edge rate (route_only shrinks the completion to the part
  that can hallucinate edges).
- Teacher monitor is a gate for e20a: if the path-only teacher pass rate
  collapses (e.g. < ~50% vs the reasoning regrade), the e20a corpus is both
  small and biased toward easy tasks — lean on e20b (oracle) for the
  architecture question and report the teacher finding on its own.
- e20b vs e20a separates "no reasoning in targets" from "teacher-filtered
  data": oracle routes are complete and uniform; if oracle >> pathonly at
  matched arms, the filter (not the format) was the binding constraint.
- hop_depth vs text_edges at matched data answers the adviser's question:
  if stripping reasoning helps the graph arm more than the baseline, the
  reasoning text was indeed interfering with the graph channel.

## Ops notes

- vast storage is at **4.91/5.00 TB (98%)** — e20 adds 2 corpora + 4
  checkpoint dirs (`save_total_limit=1` kept). May need a cleanup pass first.
- e19 replication wave (7825460-65) still running; e20 does not depend on it,
  but if a replica lands far from 73.8%, note the band when reading e20 deltas.
- Tests: `tests/test_e20_route_only.py` (contract + wiring + generator modes;
  the generator tests need numpy/spine/networkx — betty login node has them).
