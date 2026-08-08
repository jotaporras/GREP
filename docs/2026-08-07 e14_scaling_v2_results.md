---
tags: [experiment, e14, scaling, graph-injection, vllm-data]
date: 2026-08-07
status: complete (6/6 runs final; 7438021 checkpoint verification + train-graph probes 7438024/7438025 pending)
related: ["2026-07-21 e13_nav_pe_setup"]
wandb: alelab/GREP-PRISM (tag e14_scaling)
---

# e14 scaling on the v2 corpora — floor / text-edges / graph-channel brackets at n30 and n60

> [!abstract] TL;DR
> On the freshly regenerated v2 corpora (rename-map populate contract, zero
> hallucinated node refs), the learned graph channel is a large real gain over
> the no-connectivity floor at both sizes (n30: 0.157 → **0.700**; n60:
> 0.029 → **0.114**). But the trivial bracket — writing the edge list into the
> prompt text — beats it at **both** sizes and the gap widens sharply with
> scale: 0.986 vs 0.700 at n30 (1.4×), **0.943 vs 0.114 at n60 (8.2×)**. The
> graph channel collapses with graph size while text edges barely move
> (0.986 → 0.943). As it stands, the scaling result points against the learned
> channel: its gain over the floor is mostly "has connectivity at all," and it
> stops being a usable vehicle for connectivity by n60.

## 1. Setup

Six runs, three arms per graph size, all on `google/gemma-4-31B-it`, 3 epochs,
lr 2.5e-4, icl_examples=2, full test-set generation eval (`eval.num_graphs=-1`),
one B200 each:

- **floor** — `scripts/e14_baseline_llm.sbatch`, `TEXT_EDGE_LIST=none`: node
  names in the prompt, no connectivity anywhere. arch=llm, fresh LoRA.
- **edges-in-prompt** — same script, `TEXT_EDGE_LIST=present`: the informed
  bracket; eval side follows automatically (`include_edge_list`).
- **graph channel (GT)** — `scripts/e14_stage1to3_binary.sbatch`: mask_alpha=0.0
  binary adjacency, Psi from the navigator GT
  (`NAV_GT=outputs/e9_multistage_training/suite9_p152/path_navigator_gt.pt`;
  the sbatch default suite8 path does not exist on betty — every e14 run has
  actually used suite9_p152), LoRA continued from
  `e9_ms_stage1_sqgk4o3j/checkpoint-100`, fresh LoRA epochs on top.

Corpora: `$ALELAB_DRIVE/GREP-PRISM/data/n_{30,60}_vllm_v2/gen/nav_n{30,60}_gemma_data/split`
(260 train / 70 val rollouts, 26 train + 7 test graphs each), generated
2026-08-07 under the v2 rename-map populate contract — audited clean: 99 graphs
/ 990 tasks, zero dangling edges, placeholder survivals, or unknown task refs.
Both baseline arms cannot leak connectivity from the corpus; the floor arm's
only possible source would be the (absent) edge list.

## 2. Results (final, 3 epochs each)

| size | arm | job | run | gen acc | halluc | train loss | tok acc |
|---|---|---|---|---|---|---|---|
| n30 | floor | 7437500 | `e14v2_n30_baseline` [e07f6180](https://wandb.ai/alelab/GREP-PRISM/runs/e07f6180) | 0.157 | 0.537 | 0.079 | 0.970 |
| n30 | GT | 7437502 | `e14v2_n30_gt` [8g2o6yzd](https://wandb.ai/alelab/GREP-PRISM/runs/8g2o6yzd) | **0.700** | 0.089 | 0.068 | 0.973 |
| n30 | edges | 7437559 | `e14v2_n30_baseline_edges` [krktnvkr](https://wandb.ai/alelab/GREP-PRISM/runs/krktnvkr) | **0.986** | 0.005 | 0.067 | 0.975 |
| n60 | floor | 7437501 | `e14v2_n60_baseline` [32na8lx2](https://wandb.ai/alelab/GREP-PRISM/runs/32na8lx2) | 0.029 | 0.570 | 0.092 | 0.966 |
| n60 | GT | 7437503 | `e14v2_n60_gt` [q91esr6a](https://wandb.ai/alelab/GREP-PRISM/runs/q91esr6a) | **0.114** | 0.222 | 0.082 | 0.970 |
| n60 | edges | 7438021 | `e14v2_n60_baseline_edges` [iq8bo4sa](https://wandb.ai/alelab/GREP-PRISM/runs/iq8bo4sa) | **0.943** | 0.015 | 0.079 | 0.972 |

(7437560/[4ybimax8](https://wandb.ai/alelab/GREP-PRISM/runs/4ybimax8) was the
first attempt at the n60 edges arm — died at 46s on a dgx024 CUDA flake;
7438021 is its identical resubmit.)

Gen acc = `eval/accuracy` over the full 70-rollout test split (7 held-out graphs).

## 3. Reading

- **The graph channel works.** 4.5x over the floor at n30, 4x at n60, with
  hallucination cut 0.54 → 0.09 (n30) and 0.57 → 0.22 (n60). SFT alone
  (floor arms hit 0.97 token accuracy) does not solve routing; connectivity
  does.
- **But text edges work better at both sizes, and the gap explodes with
  scale.** n30: 0.986 (69/70) vs 0.700, hallucination 0.005 vs 0.089 — and
  the edges arm was already at 0.986 at epoch 0.92, so it converges there
  rather than getting lucky. n60: **0.943 (66/70) vs 0.114**, hallucination
  0.015 vs 0.222. Going n30 → n60 the channel loses 84% of its accuracy while
  the text bracket loses 4%. A ~48-region edge list in the prompt is
  evidently still fully usable by the LLM; the learned channel is the thing
  that doesn't scale. The honest decomposition of the GT arm's gain over the
  floor is mostly "has connectivity at all," not "the learned channel is the
  right vehicle."
- **Everything is much harder at n60 without usable connectivity** (floor
  0.029 ≈ chance, GT 0.114) — but 0.943 shows the tasks themselves are not
  the bottleneck; the connectivity representation is.
- Note the GT arms carry Stage-1 SFT epochs the baselines don't (sbatch
  header caveat); that asymmetry favors GT and it still trails the text
  bracket.

## 4. Failure-mode analysis (added 2026-08-08, from the epoch-3 `eval_logs` JSONs)

Per-sample classification of every wrong answer, validated against the true
test graphs (scratchpad `analyze_failures.py` / `analyze_invalid_edges.py` /
`analyze_sibling_confusion.py`).

**Where the errors are.** Failures are almost entirely *invalid edges* — the
plan is well-formatted, every node name exists (node hallucination is not the
problem on v2 data), but 1–2 transitions in the path don't exist in the graph:

| run | invalid edges | wrong endpoints | other |
|---|---|---|---|
| n30 GT (21 fails) | 16 | 5 | 0 |
| n60 GT (62 fails) | 46 | 15 | 1 eval crash |
| n60 floor (68 fails) | 57 | 11 | 0 |

**The phantom edges are local, and they repeat.** Nearly all invalid predicted
edges connect nodes at true distance 2–3 (n30 GT: 19/24 at d=2; never random
long-range jumps), and the same phantom edge recurs across tasks —
`atrium_1→lab_2` appears in 8 of the n30 GT failures, `command_deck_2→command_deck_1`
in 5 of the n60 GT ones. These are consistent wrong beliefs about adjacency,
not sampling noise: the channel conveys coarse proximity but the model commits
to specific nonexistent shortcuts.

**n30 GT's 0.700 is concentrated, not diffuse.** Per-graph accuracy:
`data_gen_011` **0.1**, `data_gen_020` 0.5, the other five graphs 0.8–0.9.
Without graph 011 the run scores ~0.86. In 011, 13/24 phantom-edge instances
are *sibling swaps*: predicted `a→lab_2` where `lab_1` (same base name,
different suffix) really is adjacent to `a`. So the common n30 GT error is
suffix confusion between duplicate-base-name nodes plus d=2 hop-skipping,
concentrated in whichever test graph has confusable duplicates near the routes.

**n60 GT's 0.114 is diffuse.** Every test graph scores 0.0–0.2. Success by
true optimal hop count (BFS start→goal): 1-hop 4/8, 2-hop 2/29, 3-hop 2/30.
At n30 the same curve is 0.92 / 0.58 / 0.70. So at n60 the channel's per-edge
adjacency fidelity drops to roughly a coin flip even for single-hop tasks, and
multi-hop compounding does the rest. Sibling swaps are only ~1/3 of n60
phantom edges — the signal is broadly blurred, not just suffix-confused.
(Context: on the old v1 data, GT scored 0.267 at n60 and 0.057 at n100 — the
size collapse predates the datagen fix and is monotone in graph size.)

**n60 floor fails differently**: it invents a generic hub topology
(`bridge→comm_hub` etc., the same made-up edges across every task) — pure
prior, as expected with no connectivity given.

**First-guess hypotheses.** n60 collapse: the graph-channel injection has a
fixed representational budget per node, and at 60+ nodes the binary-adjacency
signal blurs below the threshold the LLM needs to commit to exact edges —
consistent with errors staying local (d=2–3) and with the monotone v1 size
trend. n30 residual: not a corpus problem (v2 is clean) but channel
resolution on confusable node pairs — duplicate base names one hop apart get
near-identical injected representations and the model swaps suffixes.

**Overnight causal probe — result invalidated the probe, not the model.**
Jobs 7438024/7438025 re-evaluated the two GT checkpoints on 7 *train* graphs
each via `e14_transferability.sbatch` (the standalone reload path). Both came
back at floor: n30 GT **0.071**, n60 GT 0.043, hallucination ~0.46–0.54. A
checkpoint scoring 0.700 on held-out test graphs in-training cannot honestly
score 0.071 on its own training graphs — no capacity or generalization story
produces that direction. The suspect is the reload eval path itself
(`scalability_evaluation` → `graph_augmented_llm_from_pretrained`):

- Prior evidence it's been broken a while: `results/e14_transferability_p152`
  scored **0.0** on the same distribution where that checkpoint's in-training
  eval scored ~0.65+, and every e13f transferability result is ~0.0 too.
- Ruled out so far: resolved policies are correct (binary /
  decode_consistent / edge list none); Psi producer load is fail-loud and
  passed; the LoRA adapter's 820 remapped keys all exist in the rebuilt PEFT
  model (verified against an empty-weights rebuild), so the adapter is not
  silently dropped.
- Remaining suspects: LoRA `merge_and_unload` into the 4-bit-quantized base
  (dequantize→merge→requantize is lossy; training keeps the adapter
  unmerged), or a train/eval mismatch in how the mask/injection is rebuilt.
- train_v3 has a designed save→load round-trip check
  (`eval.post_train_graphs`) but it was not enabled in the e14 runs, so the
  reload boundary was never exercised.
- **Control job 7439074 verdict: reload path BROKEN.** Same reload protocol,
  same n30 GT checkpoint, its own TEST graphs: **0.057** (4/70) where the
  in-training eval scored 0.700. Every conclusion drawn through the reload
  path (both probes, the e14_p152 and e13f transferability ~0.0s) is void.
- **Prime suspect confirmed active in the logs**: PEFT warns
  `Merge lora module to 4-bit linear may get different generations due to
  rounding errors` (`peft/tuners/lora/bnb.py:397`) during every reload —
  `loaders.py` merges the LoRA into the nf4-quantized base
  (dequantize→merge→requantize). LoRA deltas are small relative to the nf4
  quantization step, so requantization can round the fine-tune away, leaving
  ~base Gemma + mask — exactly the floor-level behavior measured. Training
  never merges (adapter rides bf16 on the 4-bit base), so in-training eval
  is unaffected.
- **Discriminator job 7439306 confirmed it**: identical eval without
  `--four-bit` (lossless bf16 merge) scored **0.757** (53/70) vs 0.057 —
  full recovery on every graph, slightly above the in-training 0.700. Also
  26:40 vs 44:25 elapsed: the 4-bit reload wasn't even faster on a B200.

  | graph | in-training | reload+4bit | reload+bf16 |
  |---|---|---|---|
  | 005 | 0.9 | 0.0 | 0.9 |
  | 011 | 0.1 | 0.0 | 0.3 |
  | 012 | 0.9 | 0.1 | 0.9 |
  | 016 | 0.9 | 0.1 | 0.8 |
  | 020 | 0.5 | 0.0 | 0.7 |
  | 026 | 0.8 | 0.0 | 0.9 |
  | 027 | 0.8 | 0.2 | 0.8 |

- **Fix applied** (commit `b378e5a`): `loaders.py` keeps the LoRA adapter
  attached on 4-bit reload instead of `merge_and_unload` — matching both
  training (which never merges) and the plain-LLM loader path (which never
  merged either, which is why baseline reloads were unaffected).
- **Rerunning with the fix**: 7439396 (validation: n30 GT on test graphs,
  fixed 4-bit path, expect ~0.70) gates 7439398/7439399 (the n30/n60
  train-graph probes, outputs `*_on_train_fixed/`). Those finally answer the
  capacity-vs-generalization question the original probes were built for.

## 5. Metric caveats (eval definition, not run faults)

- `eval/accuracy` == `eval/valid_path_rate` bit-identically in 4/5 finished
  runs — accuracy appears to be counted as "produced a valid path." In
  7437503 alone the denominators differ (accuracy 8/70 vs valid_path 8/69):
  one test item dropped from the valid-path denominator. Don't treat the two
  metrics as interchangeable.
- `eval/path_optimality_rate` > 1.0 everywhere (1.09–1.52), i.e. produced
  paths run longer than optimal; the name suggests a ratio ≤ 1.

## 6. Ops log

- Submitted 2026-08-07 ~18:46 (original four) and ~19:53 (edges brackets),
  betty dgx-b200. Monitored by the Opus monitor-job pipeline; **zero
  fable-debugger escalations across all six lineages**.
- 7437560 died at 46s: `CUDA error: device busy or unavailable` during device
  setup on dgx024 (which was already hosting 7437501) — infra flake, not
  code; the identical script ran fine as 7437559. Resubmit with
  `TAG=e14_scaling TEXT_EDGE_LIST=present DATA_ROOT=.../n_60_vllm_v2/gen/nav_n60_gemma_data/split`;
  if it lands on dgx024 and repeats, add `--exclude=dgx024`.
- The original four were submitted with a wrong wandb tag (`e14_vllm_v2`) and
  patched server-side to group `e14_scaling` mid-run; the patch survived
  `finish()` on all four (verified). The edges runs were tagged correctly at
  submission. Standing rule: every e14 run groups under `e14_scaling`.
- 2026-08-08: 7437560 resubmitted as **7438021** (dgx006, verified
  `tag=e14_scaling` / `text_edge_list: present` in the log header); completed
  in 2h50m, metrics above from wandb — on-disk checkpoint check pending the
  next SSH window (the laptop's KCM credential breaks recurrently; outage
  timeline in the monitor's `ssh-outage-log.md`). All five
  finished runs verified on disk: sacct COMPLETED, checkpoints under
  `$PROJ/outputs/e14_stage1to3/<RUN_NAME>_<wandb_id>` (note the wandb-id
  suffix — the bare `<RUN_NAME>` path in the sbatch does not exist), each with
  `checkpoint-390`, `adapter_model.safetensors`, `eval_logs/`; GT runs also
  carry `gnn_weights.pt`.
- A macOS KCM quirk to remember: the TGT can show valid in `klist` while
  GSSAPI fails with `no credential for <UUID>`; a fresh `kinit` clears it.
