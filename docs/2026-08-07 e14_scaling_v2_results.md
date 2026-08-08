---
tags: [experiment, e14, scaling, graph-injection, vllm-data]
date: 2026-08-07
status: partial (5/6 runs final; n60 edges-in-prompt bracket died on a GPU infra flake 46s in, resubmit pending)
related: ["2026-07-21 e13_nav_pe_setup"]
wandb: alelab/GREP-PRISM (tag e14_scaling)
---

# e14 scaling on the v2 corpora — floor / text-edges / graph-channel brackets at n30 and n60

> [!abstract] TL;DR
> On the freshly regenerated v2 corpora (rename-map populate contract, zero
> hallucinated node refs), the learned graph channel is a large real gain over
> the no-connectivity floor at both sizes (n30: 0.157 → **0.700**; n60:
> 0.029 → **0.114**). But at n30 the trivial bracket — writing the edge list
> into the prompt text — scores **0.986** with 0.005 hallucination, decisively
> above the graph channel. Any claim that the learned channel is the effective
> way to supply connectivity has to contend with that run. Whether the text
> bracket degrades at n60 is exactly the run that flaked; rerun before drawing
> the scaling conclusion.

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
| n60 | edges | 7437560 | `e14v2_n60_baseline_edges` [4ybimax8](https://wandb.ai/alelab/GREP-PRISM/runs/4ybimax8) | — FAILED 46s | — | — | — |

Gen acc = `eval/accuracy` over the full 70-rollout test split (7 held-out graphs).

## 3. Reading

- **The graph channel works.** 4.5x over the floor at n30, 4x at n60, with
  hallucination cut 0.54 → 0.09 (n30) and 0.57 → 0.22 (n60). SFT alone
  (floor arms hit 0.97 token accuracy) does not solve routing; connectivity
  does.
- **But text edges work better, at least at n30.** 0.986 (69/70) vs 0.700,
  hallucination 0.005 vs 0.089 — and the edges arm was already at 0.986 at
  epoch 0.92, so it converges there rather than getting lucky. The honest
  decomposition of the GT arm's gain over the floor is mostly "has
  connectivity at all," not "the learned channel is the right vehicle."
- **Everything is much harder at n60** (floor 0.029 ≈ chance, GT 0.114).
  The open question is whether prompt-text edges also collapse with graph
  size — 48 regions of edge list is a long prompt — or stay near ceiling.
  That is the flaked run; do not write the scaling story without it.
- Note the GT arms carry Stage-1 SFT epochs the baselines don't (sbatch
  header caveat); that asymmetry favors GT and it still trails the text
  bracket.

## 4. Metric caveats (eval definition, not run faults)

- `eval/accuracy` == `eval/valid_path_rate` bit-identically in 4/5 finished
  runs — accuracy appears to be counted as "produced a valid path." In
  7437503 alone the denominators differ (accuracy 8/70 vs valid_path 8/69):
  one test item dropped from the valid-path denominator. Don't treat the two
  metrics as interchangeable.
- `eval/path_optimality_rate` > 1.0 everywhere (1.09–1.52), i.e. produced
  paths run longer than optimal; the name suggests a ratio ≤ 1.

## 5. Ops log

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
- Checkpoints expected under `$PROJ/outputs/e14_stage1to3/<RUN_NAME>` —
  **not yet verified on disk** (a Kerberos outage on the submitting laptop
  blocked ssh for the back half of the campaign; wandb `finished` is the only
  completion evidence so far).
- A macOS KCM quirk to remember: the TGT can show valid in `klist` while
  GSSAPI fails with `no credential for <UUID>`; a fresh `kinit` clears it.
