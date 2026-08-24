# e19 — hop-separated post-fusion (design + runbook)

**Date:** 2026-08-23 · **Tag:** `e19_hopfusion` · **Status:** implemented, n60_v3 fleet submitted

## Hypothesis

The e17 per-hop analysis and the e18 probe traced the mask arch's failures to an
*identification* problem: Ψ aggregates a node's whole multi-hop neighbourhood into one
vector, and the LLM cannot un-mix "who is 1 hop away" from "who is 2 hops away" out of
that aggregate — nothing in the channel says which hop a piece of information came
from. e19 tests whether giving the LLM **separate per-hop channels** — a stack of
node representations, one per hop distance, each fused through its **own projection
matrix and gain** — lets it condition its next-node choice on exactly the hop depth
the current reasoning step needs.

Two variants (user spec, 2026-08-23):

- **V1 `shift` — no R-PEARL.** A standalone graph transformer over non-R-PEARL node
  features produces H; the channel stack is H under repeated applications of the graph
  shift operator: Ψ_k = Ŝᵏ·H for k = 0..K (Ŝ = row-normalised symmetrised adjacency,
  no self-loops). Node features: a learned **codebook** — `Embedding(codebook_size,
  d_model)` indexed by node index. This is the "random features" option (random at
  init, learnable after) and is deterministic at eval; it also gives every node a
  distinct ID, which attacks the e18 sibling-confusion failure directly. The
  "LLM-processed hidden states" feature option is deferred: node-name embeddings only
  exist prompt-side (soft_edges-style), and the pf decode path rebuilds Ψ per graph
  without prompt context.
- **V2 `depth` — keep R-PEARL.** No new tower: the channel stack is **taps of the
  existing Ψ producer** (`GraphTransformer.forward_taps`) — channel 0 = the R-PEARL
  PE (pre-blocks), channel ℓ = the output of sparse-attention block ℓ. Receptive
  field grows k_gt hops per block, so channels are ordered by neighbourhood depth.
  Sharing the pe_model means the taps ride on the navigator-PE warm start
  (`pe_gt_from`) and the tower keeps its damped structural LR; only the fusion
  modules are fresh.

## Fusion (both variants)

Extends the e17 post-fusion residual write (same layer scope, same QUERY-role
injection-map positions, same decode injectors):

    h[p] += tanh(pf_gain_l) · Σ_k tanh(pf_ch_gain_k) · RMSNorm_k(W_k · Ψ_k[node(p)])

- `W_k`/`RMSNorm_k` are **separate per channel** (the point of the experiment);
  `pf_gain` stays per-layer.
- **e17/e18 finding applied — gains start OPEN.** Zero-init gates never open
  (pf_gain ≤ 5e-3 in every e17 run; decision_gain pinned at init in e18). These are
  fresh SFT runs with no warm-start no-op to preserve, so
  `post_fusion_gain_init=1.0` (tanh ≈ 0.76) on both the layer and channel gains; the
  pathway carries signal from step 0 and SFT learns around it — the same mechanism
  that made e18's decision gating work (fixed gain 3, LoRA adapts).
- Implementation seam: `pf_psi(g)` (returns [N, d] classic / [N, K, d] hop modes) and
  `_pf_project` (per-channel sum) — every existing pf call site (build_pf_signal,
  MaskDecodeInjector, BatchedMaskDecodeInjector, RL snapshot) goes through these two,
  so decode/parity machinery is unchanged. `pf_hop_mode="none"` is bitwise the
  pre-e19 pathway. Hop modes + pointer_fusion are rejected (shared ψ snapshot slot).

## e18 findings applied

- Base arm = **mask_a** (decision gating, gain 3): best e18 graph arm — 4× fewer
  hallucinated edges at n10, halved at n60, probe first_ok .98. B (struct keys) is
  dropped everywhere; D not used.
- `binding_head` rides as one arm per variant (bind was the other above-control arm).
- Metrics discipline: single-run n60 accuracy differences < ~10 pts are noise —
  decision metrics are hallucinated-edge rate + the paired neighbour probe
  (`results/e18_probe`), per e18. Per-epoch built-in eval (`EVAL_EPOCH_INTERVAL=1`)
  + the standard post-train eval keep epoch-3 numbers comparable to cn7ub88q
  (72.6 built-in) / mask_a 9a6lwgfj (55/84 ep2) / control rlkmq6hj.
- e17 lesson upstream of all of this: **data dominates** — everything trains on
  n60_v3 (the cn7ub88q recipe), not old-n30.

## Fleet (up to 3 runs per variant, betty dgx-b200, ~4–8 h each)

| arm | base | pf config | job (2026-08-23) |
|---|---|---|---|
| `hop_shift`       | mask_a        | shift, K=3 (4 ch), gain_init 1.0 | 7817653 |
| `hop_shift_k5`    | mask_a        | shift, K=5 (6 ch), gain_init 1.0 | 7817655 |
| `hop_shift_bind`  | mask_a_bind   | shift, K=3, gain_init 1.0 | 7817656 |
| `hop_depth`       | mask_a        | depth (4 ch: PE + 3 blocks), gain_init 1.0 | 7817657 |
| `hop_depth_bind`  | mask_a_bind   | depth, gain_init 1.0 | 7817658 |
| `hop_depth_gain3` | mask_a        | depth, gain_init 3.0 (gain-sensitivity arm) | 7817659 |

Submitted at commit `bad6f16` (tests: 21 hop/pf + 47 mask/e18 + 14 pointer/RoPE
all green on the betty login node). NOTE: the first submission (7817490-95) died
at t=0 with ARM unset — a remote-shell quoting bug in the submit loop
(`ARM=$a` expanded by the OUTER ssh shell where `a` is empty; escape as
`ARM=\$a` inside `ssh betty 'bash -lc "..."'`), not a code failure.

Submit: `ARM=hop_shift sbatch scripts/e19_n60_sft.sbatch` (defaults: n60_v3 split,
3 epochs, per-epoch built-in eval, probe on 3 test graphs, tag `e19_hopfusion`,
outputs `outputs/e19_hopfusion/`). Controls: the existing e18 n60 pair
(mask_a 9a6lwgfj / mask rlkmq6hj) + cn7ub88q — same recipe, same code path when the
e19 flags are off (locked by the no-op tests).

## New config keys (gnn.*, base_config.yaml)

`post_fusion_hop_mode` (none|shift|depth), `post_fusion_hop_k` (shift channels − 1),
`post_fusion_gain_init` (layer + channel gains), `post_fusion_codebook_size`,
`post_fusion_hop_gt_layers/heads/k` (shift-tower topology; k=1 keeps one hop per
block so the shift stack carries the hop separation). All recorded in
train_config.json (train_v3 passthrough) and load-bearing at eval rebuild
(loaders). Weights: `pf_proj`/`pf_norm` (now ModuleLists, same keys), plus
`pf_ch_gain`, and `pf_hop_gt` for shift mode (run_dir ↔ loaders, fail-loud).

## Decision rules

- Primary: hallucinated-edge rate and paired probe first_ok/exact vs mask_a
  (9a6lwgfj) at matched steps; accuracy secondary (n=84 noise).
- A hop-channel telemetry check rides in wandb (`e19/pf_ch_gain_k`, pf gain
  mean/absmax): if SFT drives all channel gains toward 0, the pathway is being
  rejected — that, with flat probe deltas, kills the hypothesis.
- If shift ≈ depth ≈ mask_a everywhere → hop separation is not the binding
  constraint; fold e19 into the e18 conclusion and proceed to the closed-loop
  SPINE stage on the best existing arm.

## 2026-08-24 — v1 → v2, sweep, results, replication wave

**v1 fleet (7817653-59) killed ~step 300** (user request): open gates + unit
RMSNorm pinned the write at unit RMS → loss stalled ~0.49 EMA vs mask_a 0.16.
**v2 = ControlNet-style init** (commit 9d52836): gates stay OPEN, `pf_norm`
scales ZERO-init → bitwise no-op at step 0 (verified: step-0 loss 0.274-0.285 =
mask_a 0.27), full-strength grad to the scale. v2 fleet **7822640-45**
(`e19_n60_<arm>_v2`): hop_shift 42o7anmc · hop_shift_k5 · hop_shift_bind
xwwqep8a · hop_depth · hop_depth_bind · hop_depth_gain3.

**mask_a hparam sweep** (9 runs, tag `e19_mask_a_sweep`, commit f8c9fb2/ac71ac0):
7822668/69 seed replicas · 70 lr_low · 79 lr_high · 80 slr_005 · 81 slr_02 ·
82 alpha_03 · 84 dg_6 · 7823159 scope_top_half. **All 8 six-hour jobs TIMEOUT
mid-epoch-3** (6h wall clock + node contention; solo runs take ~4h15) — ep1/ep2
evals + last checkpoints exist, no ep3 evals/probes except scope_top_half
(completed, final 50.0%). Ep2 ranking: slr_005 66.7% > slr_02 64.3% >
replicas ~54.8%; lr_high 60.7% at ep1.

**Built-in eval results (84 questions, 7 test graphs):** hop_depth **73.8%
final** (ties the cn7ub88q built-in record 72.6%); hop_shift 73.8% ep2 →
**60.7% final**, halluc 0.00-0.04/file (record low); baseline replicas ~54.8%,
9a6lwgfj final 56.0%. Probe (hop_depth vs mask_a): sibling_err/query
0.375 vs 0.507, halluc/query 1.326 vs 1.715, first_ok 0.979 =. Bind arms dead
again (binding_loss ~0.89 plateau, eval_loss 0.467). pf_norm_scale ~0.003 —
channels open barely yet the deltas are real; attribution needs the post-load
harness (pending).

**Replication wave (the one authorized overnight launch, jobs 7825460-65,
commit fdd9a24):** hop_depth s17 / s1031, hop_shift s17 / s1031,
hop_depth×slr0.05, hop_shift×slr0.05 — SEED/SLR knobs added to
`e19_n60_sft.sbatch`, wall clock 10h. Purpose: is hop_depth 73.8% real or the
top of the ±10 pt band, and does the sweep's best knob (slr_005) stack with
hop channels? Monitor bkkan13g9.

**TODO (morning):** post-load scalability evals on hop_depth/hop_shift/mask_a
checkpoints (canonical harness); rescue ep3 evals+probes for the 8 timed-out
sweep arms from their last checkpoints if the knob ranking matters.
