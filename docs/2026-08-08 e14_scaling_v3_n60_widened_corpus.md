---

## experiment: e14 v3 — n60 widened corpus (2x graphs + long-hop tasks)
date: 2026-08-08
status: complete (2 training runs + round-trip checks + train/test probe 2x2 + 6-epoch GT extension + SPINE tools-enabled eval done)
wandb_tag: e14_scaling

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


| arm              | job     | run name          | script                      | key overrides                           |
| ---------------- | ------- | ----------------- | --------------------------- | --------------------------------------- |
| edges-in-prompt  | 7439955 | `e14v3_n60_edges` | e14_baseline_llm.sbatch     | TEXT_EDGE_LIST=present                  |
| graph-channel GT | 7439956 | `e14v3_n60_gt`    | e14_stage1to3_binary.sbatch | NAV_GT=suite9_p152/path_navigator_gt.pt |


Both: DATA_ROOT=v3 split, 3 epochs, `eval.num_graphs=-1` (84 samples/eval),
and — first time — `eval.post_train_graphs` **enabled** (commit `e2a169a`):
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
with retention rising 47% → 70% (fixed-path 2×2), so the gain is genuine
transfer, not memorization. But the gap to edges-in-prompt (0.952) persists
at ~2.4×: the data lever works and does not close the gap. The long-hop
tasks split the arms completely: edges ~13–14/14, GT 1–5/14.
Post-hoc 6-epoch extension: GT keeps climbing past the 3-epoch cutoff and
plateaus at ~0.56 (epochs 5 = 6 exactly), narrowing the edges gap to
~1.7× — see the follow-up section. Post-hoc SPINE (tools-enabled) eval: the
ordering and the widening survive tool access (edges 96.4% vs GT 47.6% here,
against 98.6% vs 71.4% at n30), and it exposes a GT-specific keyword-formatting
defect plus a graph-size-independent OOM signature.**

### Headline (84 samples = 7 frozen graphs × 12 tasks)


| arm                | job / run                                                            | epochs 1→2→3 (in-training) | round-trip reload | Δ      | halluc |
| ------------------ | -------------------------------------------------------------------- | -------------------------- | ----------------- | ------ | ------ |
| edges (`present`)  | 7439955 [zqvzaab6](https://wandb.ai/alelab/GREP-PRISM/runs/zqvzaab6) | 0.726 → 0.881 → **0.952**  | 0.929             | −0.024 | 0.011  |
| GT (graph channel) | 7439956 [kma1nipe](https://wandb.ai/alelab/GREP-PRISM/runs/kma1nipe) | 0.286 → 0.357 → **0.393**  | 0.440             | +0.048 | 0.084  |


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


| arm   | v2 final (in-training) | v3 in-training    | v3 reload |
| ----- | ---------------------- | ----------------- | --------- |
| edges | 0.943                  | **0.957** (67/70) | 0.914     |
| GT    | 0.114                  | **0.457** (32/70) | 0.457     |


GT's legacy-10 score is identical on both eval paths (32/70 exactly), so
the 4× improvement over v2 is solid, not path-dependent. Edges was already
at ceiling. Per-graph GT (in-training): 052 2/10, 053 7/10, 054 3/10,
055 4/10, 056 6/10, 057 4/10, 058 6/10 — more diffuse than v2's
collapse-everywhere 0.0–0.2.

### Long-hop subset (14 samples: 2 per graph, hops 3–5)


| arm   | in-training | reload |
| ----- | ----------- | ------ |
| edges | 13/14       | 14/14  |
| GT    | 1/14        | 5/14   |


With edges in the prompt the long-hop tasks are nearly free; the graph
channel largely fails them. Small n — treat as directional. The GT
in-training vs reload disagreement concentrates here (the legacy-10 rows
agree exactly), which is what moved the 84-sample totals 0.393 → 0.440.

### Transfer gap (fixed-path 2×2, v3 GT) — ANSWERED


|       | train graphs               | test graphs                         | retention |
| ----- | -------------------------- | ----------------------------------- | --------- |
| v2 GT | 0.429                      | 0.200                               | 47%       |
| v3 GT | **0.631** (7451959, 53/84) | 0.440 (round-trip, same fixed path) | **70%**   |


**Both halves moved, and transfer moved more.** Train +47% relative
(0.429→0.631), test +120% relative (0.200→0.441), retention 47%→70%. The
corpus widening bought genuine generalization, not just memorization — the
"transfer" outcome of the two pre-registered readings. Caveats: retention
is still below n30's 85% (gap narrowed, not closed), and 0.631 on the
model's *own training graphs* is still a real fitting deficit vs the edges
arm's 0.952 on *test*. The channel improves on both axes with more data
but catches the text bracket on neither.

Per-graph train accuracy spreads 0.417–0.917 (data_gen_000 0.917, 001
0.417, 002 0.667, 003 0.583, 004 0.417, 005 0.833, 006 0.583) — some
graphs remain much harder for the channel; a failure-mode pass over the
weak ones (001, 004) is the natural next diagnostic.

### Follow-up in flight: 6-epoch GT rerun (submitted 08-09)

GT was still climbing at the 3-epoch cutoff (0.286 → 0.357 → 0.393, no
plateau). Rerun of the identical recipe with `EPOCHS=6` (env override
added in `668586a`) — same v3 split, same suite9_p152 navigator,
round-trip check on. Question: does GT keep climbing past 0.393 in epochs
4–6, or was 3 epochs near the asymptote? Results to be appended here.

- **7459421** (run 84bouc2f): killed ~2h in. Its epoch-1 eval hit 2
CUDA-OOM crashes on long-sequence samples (49/58 GiB allocation attempts;
"reserved but unallocated" grew 1.7 → 45.7 GiB between them = allocator
**fragmentation**, not a leak) and ran ~2h vs the usual ~30–40 min.
Crashed samples count as *incorrect* (evaluate.py:389), and both
comparison runs had zero crashes, so a full run would have carried a
growing handicap in exactly the late epochs under test — and projected
~15h instead of ~8h.
- **7462298**: resubmit with `PYTORCH_ALLOC_CONF=expandable_segments:True`
(+ legacy `PYTORCH_CUDA_ALLOC_CONF` alias), otherwise identical. Died in
90s on dgx024 ("CUDA device busy/unavailable" — that node's second such
failure this campaign; excluded from later hops).
- **7462389** (run ksbxgm4e): the allocator flag *worked* — stranded
memory 45.7 GiB → 331 MiB — but OOMs persisted, and with fragmentation
gone the trend became legible: **genuinely-allocated memory grew ~20 GiB
per caught OOM** (free 23.1 → 3.4 GiB within one eval). Root cause found
in `evaluate.py`'s per-sample crash handler: the exception's
`__traceback__` chain pins the failed forward's KV cache/attention
buffers in reference cycles, so each caught OOM permanently leaked its
own attempt and cascaded later samples into OOM. Killed before it could
produce six garbage eval points (crashes count as *incorrect*).
- **Fix** `5b95231` (3rd substantive bug of the campaign): the handler now
clears `e.__traceback__`/locals, `gc.collect()`s, and empties the CUDA
cache after recording the crash. Note the leak was latent in *every*
prior eval — harmless there only because zero samples crashed.
- **7464551**: resubmit with the fix + allocator flags + `--exclude=dgx024`.
Expectation: ≤1–2 isolated OOM-scored samples per eval at worst (the
60–72 GiB single-allocation samples may legitimately fail), no cascade,
eval durations back near ~35 min once memory pressure is gone.



### 6-epoch results (7464551, run [n6dz4zlq](https://wandb.ai/alelab/GREP-PRISM/runs/n6dz4zlq)) — COMPLETED 0:0, 10h45m

**Answer: GT climbs well past the 3-epoch 0.393 and plateaus at ~~0.56 —
epochs 5 and 6 are identical (47/84), the plateau signature this run was
built to detect. Best estimate of this arm's ceiling on the v3 corpus:
~0.56, vs edges 0.952 (~~1.7× gap, down from ~2.4× at 3 epochs).**


| epoch | acc (84)  | crashes | crash-adj | legacy-10         | long-hop |
| ----- | --------- | ------- | --------- | ----------------- | -------- |
| 1     | 0.321     | 3       | 0.333     | 22/70             | 5/14     |
| 2     | 0.512     | 2       | 0.524     | 36/70             | 7/14     |
| 3     | 0.464     | 0       | 0.464     | 35/70             | 4/14     |
| 4     | 0.369     | 1       | 0.374     | 29/70             | 2/14     |
| 5     | 0.560     | 2       | 0.573     | 40/70             | 7/14     |
| 6     | **0.560** | 0       | 0.560     | **40/70 = 0.571** | **7/14** |


Post-train round-trip: **0.500** (42/84, 2 crashes; Δ −0.060 vs
in-training). The delta is *negative* where all earlier round-trips were
positive (+0.048/+0.029/+0.086) — confirming reload deltas are
bidirectional sampling noise, not a systematic reload bias.

Subset movement vs the 3-epoch run: legacy-10 0.457 → **0.571** (v2 was
0.114), and long-hop finally moved, 1/14 → **7/14** — the graph channel
learned some multi-hop routing given enough passes, though edges' 13–14/14
remains far ahead.

**How to read it (caveats that matter):**

- **Quote the plateau, never a single epoch.** Adjacent swings reach 0.19
(0.46 → 0.37 → 0.56), beyond the ±0.11 that 84-sample noise explains;
the epoch-4 dip is inside the run's own volatility, not a regression.
- **The trajectory is not reproducible epoch-by-epoch.** This run's
epoch 2 (0.512) reads +0.155 above kma1nipe's (0.357) — beyond 2 SE, and
the crash handicap biases *downward*, so it can't explain it. Genuine
run-to-run training variance; some of the 6-epoch gain over 0.393 is
that variance, but the epoch-5/6 plateau sits ~3 SE above it.
- **Crashes are OOM-scored samples counted as incorrect** (~0.02/point
understatement, crash-adj column corrects the denominator). 10 total:
3/2/0/1/2/0 across epochs + 2 in the round-trip — flat, not compounding,
confirming the `5b95231` leak fix held for the whole run.

Checkpoint: `outputs/e14_stage1to3/e14v3_n60_gt_6ep_n6dz4zlq` (2.2 GB,
`gnn_weights.pt` present, `checkpoint-1872` = 6 × 312 steps).

### SPINE tools-enabled eval (added 2026-08-10)

Everything above was measured with SPINE tool-calling **disabled**
(`PRISM_DISABLE_SPINE_TOOLS=1`) — the model writes a route directly, nothing
executes. These runs are the tools-**enabled** counterpart
(`scripts/e14_transferability_spine.sbatch`): the model may call the SPINE API
actions against a live simulator while planning. Same checkpoints, same 7
frozen v3 test graphs, same scoring. The n30 v2 arms were run in the same
batch — see section 6 of the
[v2 doc](2026-08-07%20e14_scaling_v2_results.md).


| job     | arm        | run                                                                  | Acc(obj)  | correct | eval errors | formatted | keyword |
| ------- | ---------- | -------------------------------------------------------------------- | --------- | ------- | ----------- | --------- | ------- |
| 7475037 | edges      | 7439955 [zqvzaab6](https://wandb.ai/alelab/GREP-PRISM/runs/zqvzaab6) | **96.4%** | 81/84   | 0           | 84/84     | 84/84   |
| 7475039 | GT 6-epoch | 7464551 [n6dz4zlq](https://wandb.ai/alelab/GREP-PRISM/runs/n6dz4zlq) | **47.6%** | 40/84   | 2           | 82/84     | 61/84   |


Both COMPLETED 0:0; outputs under
`$PROJ/outputs/e14_transferability/<run>_spine/`.

> [!warning] These runs are effectively SINGLE-SHOT — see the v2 doc's section 6
> callout for the full rollout audit. Summary: across the whole 4-job batch,
> **307 of 308 samples used exactly one planner call**. The models emit
> `[goto(X), answer(route)]` in a single response (the answer is written before
> any tool result exists), `goto` reveals nothing by design, so no feedback
> ever returns and the loop terminates immediately. Treat this section as a
> **prompt-format ablation**, not evidence about interactive planning.

**Reading (as a prompt-format ablation).**

- **The tool-documented prompt costs the n60 GT arm something.** The edges arm
holds its ceiling (0.952 → 96.4%, +1.2 points) while GT drops from its 0.560
plateau to 47.6% (−8.4 points). Since no tool feedback is involved, this is a
*prompt-length / prompt-format sensitivity*: the extra SPINE API
documentation degrades the graph-channel arm at n60 while leaving the
text-edges arm untouched. At n30 the same comparison was flat-to-positive for
both arms, so the sensitivity appears only at the larger graph size.
- **The gap ordering and widening are unchanged**: 27 points at n30
(98.6 vs 71.4), **49 points at n60** (96.4 vs 47.6) — matching the tools-off
picture, as expected once you know both conditions elicit the same
single-shot behaviour.
- **The arms differ in how often they even emit a tool call**, which is a
genuine behavioural signal even though the calls are inert: both edges arms
emit `goto` on 100% of samples (70/70, 84/84), while the GT arms emit it on
42/68 at n30 and only **17/82 at n60**. The graph-channel arm drifts away
from the documented output contract as graphs grow — the same direction as
its keyword-rate collapse below, and plausibly the same underlying failure.
- **The keyword defect is worse at n60**: 61/84 for GT (27% of samples fail to
emit the expected keyword) against a perfect 84/84 for edges. Combined with
n30's 62/70, this is a consistent, size-scaling failure of the graph-channel
arm to produce well-formed tool-mode output — a defect separate from
routing accuracy, and plausibly a bigger lever on the score than the crashes.
- **Crash handicap**: 2 OOM-crashed samples scored incorrect ⇒ adjusted
40/82 = **48.8%**; honest range 47.6–48.8%. Neither adjustment changes the
picture.

**The eval OOMs are localised to the graph channel, and they are not a
sequence-length problem.** Across the four-job batch the split is perfect —
0 crashes on both edges arms, 2 on each GT arm — and the confound runs
*backwards*: the edges arms carry the **longer** prompts (+800–930 tokens at
n30, +1650–1990 at n60, the edge list itself) yet never fail. The failing
allocations sit in a **62–70 GiB band regardless of graph size** (n30 GT:
64.20 / 68.04 GiB; n60 GT: 62.33 / 70.37 GiB), with fragmentation flat at
133–182 MiB. A tensor whose size barely moves when the graph doubles is a
fixed-shape *padded* buffer in the injection/attention path, not something
scaling with the input — so the lever is that padding/max dimension (or
chunking that one attention computation), **not** the generic eval-side
sequence-length cap considered earlier. Both failures were near-misses (short
by 5.6 GiB and 0.81 GiB), so a modest reduction would likely eliminate them.

### Bookkeeping

- Training: 5h27m (edges) / 3h50m (GT), both COMPLETED 0:0, TAG
`e14_scaling`, 936 steps (3 × 312 — ~2.4× v2's step count, matching the
corpus growth 260 → ~570 conversations).
- Edges runs slower per step than GT (longer prompts carry the edge list) —
it started 19 min earlier and finished ~1.5h later.
- Campaign totals across datagen + training: 0 debugger escalations; 2
substantive bugs found and fixed before they could distort results
(merge-into-nf4 reload, positional sample naming).

