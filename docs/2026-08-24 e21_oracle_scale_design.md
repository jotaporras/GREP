---

## experiment: e21 oracle scale-up — design
date: 2026-08-24
status: design; corpus generation + size-control cell launching
sources: e20 results in [2026-08-24 e20_path_only_results.md](2026-08-24%20e20_path_only_results.md)

# e21 — scaling the purest signal

e20 showed that route-only oracle targets took the hop_depth graph arch from
73.8 (e19 best, reasoning targets) to **92.9/92.9** on the frozen n60_v3 test
set — with zero edges in the prompt. The working model: non-graph tokens in the
target are noise to the graph channel; bare node sequences are the purest
training signal the navigators can get. e21 scales that signal and aims the
question distribution at the *measured* residual failure modes.

## 1. What the last 6 errors are (e20 oracle hop_depth, cross-eval)

All 6 predicted paths were fully valid (edge validity 1.0, zero hallucinated
nodes/edges; 2 were cost-optimal). The failures:

- **5/6 start-grounding**: task references the start as a paraphrase ("the
  freight lift"), an ordinal sibling ("the second seed vault"), or a plain name
  ("the scout pad") and the model routes — perfectly — from the wrong node.
  Goal grounding was correct in all 6.
- **1/6 avoid-constraint violation** ("avoiding the xeno lab" → route passes
  `xeno_lab_1`).

So the residual is language→node grounding + constraint following, NOT path
search. Long-path scaling alone would not buy these points; the corpus must
also cover start-reference diversity and constraints.

## 2. Corpus `n_60_oracle_v2` (`scripts/e21_n60_oracle_v2_generate.sbatch`)

- **500 fresh skeletons**, SEED_START=601 (disjoint from all prior corpora),
  same generator params as v1 (6 communities × 8, ~63 nodes). Expected ~5.9k
  train samples ≈ 3.3× oracle v1. Populate (Gemma-4-31B/vLLM) is the only LLM
  cost, ~2 GPU-hours; oracle routes are NetworkX, self-verified by the scorer.
- **Hop stratification (the long-path lever)**: `N_LONGHOP=10` of 12 tasks per
  graph get fixed endpoints from `sample_longhop_constraints` — uniform over
  hop buckets [3, diameter] with the **diameter bucket weight-boosted 3×**
  (`--longhop-max-boost 3`, new). In e20b only 2/12 endpoint pairs were
  hop-controlled; the other 10 were the populate-LLM's free, short-biased
  choice. `max_boost=1.0` keeps the legacy sampler bit-identically (rng stream
  included). Delivered mix is auditable in `longhop_manifest.json`.
- **Avoid-constraints on long-hop tasks** (`--longhop-allow-avoid`, new): the
  LLM may add "without using X" to constrained tasks, but
  `validate_longhop_tasks` now rejects (retry loop) any avoided region that is
  the start/goal or lies on ANY shortest init→goal path — so hop stratification
  is preserved exactly. Waypoints stay forbidden on long-hop tasks; the 2 free
  tasks per graph keep supplying waypoint coverage.
- **Start-reference diversity** (`--grounding-directives`, new): the task-gen
  prompt now requires varied start references — name paraphrase, ordinal
  sibling (matching the `_N` index), object-hosted region ("the area holding
  the crate") — and caps robot_location starts at ~1/3 of route tasks. Same
  variety applies to destinations. All three flags default OFF and are
  byte-identical no-ops (tests: `tests/test_e21_longhop_sampler.py`).

## 3. Eval

- **Frozen n60_v3 test graphs 052–058, unchanged** — every e21 number stays
  comparable to e18/e19/e20. The extended eval (same 7 frozen graphs, ~60
  oracle-generated questions each, stratified by hop × task type) is the next
  step after corpus launch: at 92.9% on 84 questions one question = 1.2 pts,
  which cannot resolve progress toward 100. The original 84 stay canonical.
- Grader tightening before any ~100% claim: simple-path check (a current
  "correct" answer revisits `command_deck_1`), hop-optimality tolerance policy.

## 4. Training cells (`e20_path_only_sft.sbatch`, tag `e20_path_only_pred`… tag TBD `e21_oracle_scale`)

| cell | purpose |
|---|---|
| `DATASET=oracle_v2 ARM=hop_depth` | the main bet |
| `DATASET=oracle579 ARM=hop_depth` | oracle v1 subsampled to the pathonly size (579; `scripts/e21_subsample_split.py`) — separates data-size from target-purity in e20's 92.9 vs 84.5 |
| `DATASET=oracle_v2 ARM=text_edges` | baseline control; tests the "NetworkX routes are off-policy for text priors" hypothesis at scale |

Protocol identical to e20 (lr 2.5e-4, 3 epochs, binary edge weights,
route_only, eval on frozen v3 test graphs). With 3× data expect saturation by
epoch 2 — per-epoch evals will show it.

## 5. Risks

- N_LONGHOP=10/12 raises populate-validation strictness → more retries/graph
  skips than v1's 153/156 yield. Acceptable; monitor yield in the slurm log.
- Diameter of these skeletons is 4–5, so the boosted "long" bucket is still
  ≤5 hops. True ≥6-hop training needs sparser/larger skeletons — deferred
  (would change the graph distribution vs the frozen eval).
- vast quota at 4.93/5.00 TB before launch — prune before big runs.
