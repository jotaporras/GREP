---
experiment: e14 v3 — n60 widened corpus (2x graphs + long-hop tasks)
date: 2026-08-08
status: training complete + round-trip checks passed; train-graph probe 7451959 pending
wandb_tag: e14_scaling
---

# e14 v3 — does wider n60 training data close the GT transfer gap?

Follow-up to [2026-08-07 e14_scaling_v2_results.md](2026-08-07%20e14_scaling_v2_results.md).
The v2 same-path probe matrix showed n60 GT at **0.429 on its own train
graphs / 0.200 on test** (47% retention vs n30's 85%) — a fitting deficit
plus a transfer gap. This experiment tests the data lever: double the
training graphs and add explicitly long-hop tasks, retrain both n60 arms,
and see which half moves. Approved scope: no topology diversity, no name
permutation; the graphs-x-longhop confound is accepted (ablate later by
subsampling if the result warrants).

## Corpus: `data/n_60_vllm_v3` (built by jobs 7439536 + resume 7439857)

- **52 new train graphs**, seeds **201–252** — deliberately disjoint from
  v2's 101–133 so no train skeleton shares topology with the frozen test set.
- **7 frozen v2 test graphs** (v2 ids 005/011/012/016/020/026/027 →
  renumbered **data_gen_052–058**; mapping in
  `populated_graphs/appended_test_manifest.json`). Same skeletons, same
  names, original 10 tasks intact.
- **12 tasks/graph** = 10 per the v2 recipe + **2 long-hop Navigability
  tasks** whose (init, goal) region pair is sampled in Python, uniform over
  shortest-path lengths [3, diameter]. The LLM only phrases the task; a
  validator rejects any response that doesn't honor the fixed endpoints, so
  the hop mix cannot drift (commit `e337e33`; endpoints recorded in
  `populated_graphs/longhop_manifest.json`).
- 708 clean rollouts; split pinned via `--val-ids 052…058` → 52 train
  graphs (~570 train conversations vs v2's 260), 84 val conversations.

### Gates (both CLEAN before training submitted)

- **Hop mix** (`verify_longhop_corpus.py`): all 118 long-hop tasks verified
  on disk — endpoints match the manifest, BFS distances exact,
  criterion/answer name the goal. Delivered hops **{3: 40, 4: 59, 5: 19}**;
  region-graph diameters {4: 31 graphs, 5: 27, 6: 1} (per-graph uniform over
  [3, diam], so the aggregate peaks at 4). Note: the populate-path manifest
  records skeleton-space names (`region_K`); translate positionally
  (`regions[K-1]`) — rename preserves region order.
- **Hallucination scan**: 118 graph files / 1,416 tasks, zero dangling
  edges, placeholder survivals, unknown task refs, or non-region inits.

### Pipeline bug found and fixed on the way (would have contaminated training)

Rollout sample files were named by **list position** over the sorted
data_gen files, not the graph's own id. Two populate-rejected graphs
(022/028 — both single-attempt validator rejections, fine on retry) shifted
every later position by 1–2, so in pass 1: the explicit val-id match
silently missed 057/058, and rollouts of frozen test graphs 052/053 landed
in the **train** split — test contamination invisible to the answer-leak
check (which only greps for leaked answers, not file identity). The tell was
arithmetic: 60 val conversations ÷ 12 tasks = 5 graphs, not 7.

Fix `a8924e6`: sample ids now parse from the `data_gen_NNN` filename
(gap-proof, resume-stable). The 420 misnamed pass-1 samples were losslessly
renamed (contents were always per-graph correct), and the resume run
rebuilt the split: 59 graphs / 708 rollouts, no val-id WARN, 52/52 train
ids/files, CLEAN twice. v2 was unaffected (no gaps → positions == ids).

## Training (submitted 08-08, TAG=e14_scaling)

| arm | job | run name | script | key overrides |
|---|---|---|---|---|
| edges-in-prompt | 7439955 | `e14v3_n60_edges` | e14_baseline_llm.sbatch | TEXT_EDGE_LIST=present |
| graph-channel GT | 7439956 | `e14v3_n60_gt` | e14_stage1to3_binary.sbatch | NAV_GT=suite9_p152/path_navigator_gt.pt |

Both: DATA_ROOT=v3 split, 3 epochs, `eval.num_graphs=-1` (84 samples/eval),
and — first time — **`eval.post_train_graphs` enabled** (commit `e2a169a`):
after save, the job reloads the checkpoint from disk and re-evals, so a
reload-path regression fails at job end instead of in a later
transferability eval.

## Readout plan (fill in when jobs land)

1. Headline: in-training final accuracy + post-train round-trip accuracy
   per arm (12-task set, 84 samples — NOT comparable to v2's 70-sample
   numbers; reload path reads higher than in-training, v2 n60 0.114→0.200,
   so never mix paths in one table).
2. **Legacy-10 subset** from per-sample eval logs (tasks 0–9 per graph) —
   the direct v2 comparison. v2 numbers to beat: edges 0.943, GT 0.114
   (in-training path).
3. **Long-hop subset** (tasks 10–11 per graph) reported separately.
4. Fixed-path train/test probe for the new GT checkpoint
   (e14_transferability.sbatch on train-graph subset + test graphs) — the
   transfer-gap movement vs v2's 0.429→0.200 is the actual hypothesis test.
   Interpretation guide: big v3 improvement → data quantity was the
   bottleneck; barely moves → deficit is architectural.

## Results

**TL;DR: doubling the corpus lifted the GT arm ~3.4× (0.114 → 0.393) on a
harder eval set — and 4× on the identical legacy-10 subset (0.114 → 0.457) —
but the gap to edges-in-prompt (0.952) persists at ~2.4×. The data lever
works; it does not close the gap. The long-hop tasks split the arms
completely: edges ~13–14/14, GT 1–5/14.**

### Headline (84 samples = 7 frozen graphs × 12 tasks)

| arm | job / run | epochs 1→2→3 (in-training) | round-trip reload | Δ | halluc |
|---|---|---|---|---|---|
| edges (`present`) | 7439955 [zqvzaab6](https://wandb.ai/alelab/GREP-PRISM/runs/zqvzaab6) | 0.726 → 0.881 → **0.952** | 0.929 | −0.024 | 0.011 |
| GT (graph channel) | 7439956 [kma1nipe](https://wandb.ai/alelab/GREP-PRISM/runs/kma1nipe) | 0.286 → 0.357 → **0.393** | 0.440 | +0.048 | 0.084 |

- **Round-trip save→load check passed on both** (first time it ever ran;
  enabled by `e2a169a`). Deltas are small and *bidirectional* (−0.024 /
  +0.048), so the reload path is faithful, not systematically flattering —
  this retires the "reload reads higher" prior from the v2 probes as
  sampling noise on 84-sample evals.
- Checkpoints verified: `gnn_weights.pt` present on GT (2.2G), absent on
  the `arch=llm` edges arm (1.9G), as required.
- GT kept improving every epoch (v2 was flat from epoch 1) — the doubled
  corpus gave the graph channel something to learn from.

### Legacy-10 subset (identical graphs + tasks as v2 → direct comparison)

| arm | v2 final (in-training) | v3 in-training | v3 reload |
|---|---|---|---|
| edges | 0.943 | **0.957** (67/70) | 0.914 |
| GT | 0.114 | **0.457** (32/70) | 0.457 |

GT's legacy-10 score is identical on both eval paths (32/70 exactly), so
the 4× improvement over v2 is solid, not path-dependent. Edges was already
at ceiling. Per-graph GT (in-training): 052 2/10, 053 7/10, 054 3/10,
055 4/10, 056 6/10, 057 4/10, 058 6/10 — more diffuse than v2's
collapse-everywhere 0.0–0.2.

### Long-hop subset (14 samples: 2 per graph, hops 3–5)

| arm | in-training | reload |
|---|---|---|
| edges | 13/14 | 14/14 |
| GT | 1/14 | 5/14 |

With edges in the prompt the long-hop tasks are nearly free; the graph
channel largely fails them. Small n — treat as directional. The GT
in-training vs reload disagreement concentrates here (the legacy-10 rows
agree exactly), which is what moved the 84-sample totals 0.393 → 0.440.

### Transfer gap (fixed-path 2×2, v3 GT)

| | train graphs | test graphs | retention |
|---|---|---|---|
| v2 GT | 0.429 | 0.200 | 47% |
| v3 GT | _7451959 pending_ | 0.440 (round-trip, same fixed path) | _pending_ |

The v3 test cell (0.440) already exceeds v2's *train* cell (0.429). The
pending probe on 7 v3 train graphs (data_gen_000–006 subset) decides the
story: retention well above 47% → the widening closed transfer; train cell
also jumping with retention ~flat → the gain was fit, not transfer.

### Bookkeeping

- Training: 5h27m (edges) / 3h50m (GT), both COMPLETED 0:0, TAG
  `e14_scaling`, 936 steps (3 × 312 — ~2.4× v2's step count, matching the
  corpus growth 260 → ~570 conversations).
- Edges runs slower per step than GT (longer prompts carry the edge list) —
  it started 19 min earlier and finished ~1.5h later.
- Campaign totals across datagen + training: 0 debugger escalations; 2
  substantive bugs found and fixed before they could distort results
  (merge-into-nf4 reload, positional sample naming).
