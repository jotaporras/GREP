# e18 — n10 identity fleet, first readout (2026-08-21)

Plan / runbook: `docs/2026-08-21 e18_n10_identity_plan.md`. Corpus
`data/n_10_vllm/gen/nav_n10_gemma_data/split` (40 train / 5 test graphs, 50
held-out tasks). Every arm: e17 mask recipe, 31B-it bf16, stage-1 LoRA warm
start, **MAX_STEPS=400** (2 epochs, 1.42 s/step), commit c78f838 for training;
readouts at d33e350+. wandb tag `e18_identity`.

## Results

Held-out navigation (50 tasks), path metrics over the parsed routes (~3.5 edges
per task ⇒ ~175 edges per arm), and the neighbour-naming probe (test graphs,
n=40 queries; train graphs n=24).

| arm | job | nav | per graph 040‥044 | edge valid | **halluc. edge rate** | full-path valid | probe test first_ok / exact / P / R / hall/q | probe train |
|---|---|---|---|---|---|---|---|---|
| mask (control) | 7731157 → readout 7731237 | 32/50 = 64 % | 2 6 8 8 8 | 0.834 | 0.166 | 0.72 | 0.975 / 0.025 / 0.60 / 0.70 / 1.45 | 0.92 / 0.12 / 0.59 / 0.83 / 1.92 |
| **mask_a** (A) | 7731158 → readout 7731232 | **38/50 = 76 %** | 6 4 10 9 9 | **0.924** | **0.038** | **0.82** | 0.975 / 0.10 / **0.70** / 0.79 / **1.18** | 0.92 / 0.12 / 0.61 / 0.90 / 1.96 |
| mask_b (B) | 7731159 | 25/50 = 50 % | 2 3 5 6 9 | 0.749 | 0.251 | 0.60 | 0.85 / 0.025 / 0.58 / 0.78 / 1.73 | 0.71 / 0.04 / 0.56 / 0.85 / 2.17 |
| mask_ab | 7731160 → readout 7731233 | 34/50 = 68 % | 1 6 9 8 10 | 0.848 | 0.152 | 0.74 | 0.90 / 0.10 / 0.64 / 0.76 / 1.38 | 0.92 / 0.08 / 0.56 / 0.80 / 2.08 |
| mask_bind | 7731227 | 36/50 = 72 % | 6 7 6 8 9 | 0.912 | 0.088 | 0.82 | 0.975 / 0.025 / 0.66 / 0.66 / **1.05** | 0.88 / 0.04 / 0.52 / 0.71 / 1.96 |
| mask_b_bind | 7731228 | 27/50 = 54 % | 1 4 5 8 9 | 0.783 | 0.217 | 0.62 | 0.825 / 0.025 / 0.64 / 0.72 / 1.30 | 0.67 / 0 / 0.52 / 0.77 / 2.33 |
| mask_d (D) | 7731163 | 30/50 = 60 % | 3 4 7 8 8 | 0.837 | 0.144 | 0.68 | 0.95 / 0.025 / 0.61 / 0.69 / 1.25 | 0.92 / 0.17 / 0.58 / 0.85 / 1.96 |
| text_edges | 7731164 | 50/50 = 100 % | 10 ×5 | 1.000 | 0.000 | 1.00 | 1.0 / 1.0 / 1.0 / 1.0 / 0 | 1.0 / 1.0 / 1.0 / 1.0 / 0 |
| calib (mask, 40 steps) | 7731051 | 16/50 = 32 % | 4 0 1 5 6 | 0.612 | 0.388 | 0.38 | 0.575 / 0.025 / 0.56 / 0.67 / 1.58 | — |

Paired vs `mask` (readout) on the same 50 tasks — wins / losses / exact binomial p:
mask_a **9 / 3** (0.15) · mask_bind 6 / 2 (0.29) · mask_ab 5 / 3 · mask_d 3 / 5 ·
mask_b_bind 4 / 9 · mask_b 3 / **10** (0.09) · text_edges 18 / 0 (<0.001).

Telemetry: `decision_gain` 3.0 → 2.986 (mask_a, mask_ab), `sk_gain` 1.0 → 0.988.
**This is by construction, not a finding**: the schedule is linear decay to 0
over `max_steps` (5 warmup steps) and Adam moves a scalar ≈ LR per step, so the
total drift available to any gain is ∫LR ≈ ½·2.5e-4·400 = 0.05 (0.12 even at
936 steps). The gains are effectively fixed hyperparameters at base LR: sweep
the inits, or give the scalars their own LR group, rather than reading their
stillness as "the model didn't want them". A's effect is therefore the LoRA/tower
training *under* a fixed-gain soft decision row. `binding_loss` (chance
ln 10 = 2.30): 2.40 → **0.88** at step 400 (mask_bind), 2.46 → **0.69**
(mask_b_bind) — binding is learned well within the budget. Caveat: mask_bind's
composite eval loss rose 0.375 → 0.391 between steps 200 and 400 while train
fell; the logged eval does not split LM vs binding terms.

Budget (wandb train/loss, window means): every n10 arm was still descending
when the LR hit zero (mask 0.110 at steps 201–300 → 0.102 at 301–400), and the
n60 control at 936 steps reached train 0.079 / eval 0.114 vs n10's ~0.10 / 0.13
while still descending in its last window. 400 steps was short; the 30-min rule
allows ~1200 at 1.42 s/step. mask_a's loss is within 0.002 of the control's —
A's gain is a decode-time effect invisible to teacher-forced loss. B arms are
worse on loss too (0.108 / 0.111 vs 0.102).

## Noise floor — read before the table

* **Same checkpoint, two evals**: the `mask` checkpoint scored 28/50 (2 3 7 8 8)
  in-process and 32/50 (2 6 8 8 8) in a fresh process. Decoding is greedy
  (`DECODE_KWARGS do_sample=False`); the 3 → 6 swing on graph 041 is bf16
  nondeterminism amplified by greedy decode. Budget ±4 tasks (8 pts) per arm
  before believing an accuracy difference; the probe re-run on one checkpoint
  moved P by 0.01.
* Tasks are bimodal: over the 7 graph arms, 19/50 tasks are solved by ≥ 6 arms
  and 11/50 by ≤ 1 — graphs 040/041 carry the hard tasks.
* The path metrics pool ~175 edges per arm and are the more stable readout:
  the control ↔ A gap (0.166 → 0.038 hallucinated edges, 0.83 → 0.92 valid) is
  ~30 vs ~7 bad edges.

## Decision rules applied (plan §3)

1. `text_edges` exact = 1.0 → the probe prompt/parser is sound. ✔
2. `mask` first_ok = 0.975 → rule 2 fires: on n10 every graph arm names *a*
   true neighbour first. But **exact ≈ 0.03** and 1.0–1.7 hallucinated names per
   query for every graph arm vs 1.0 / 0 for `text_edges`: the *set*-level
   identity question is wide open — the model over-names rather than mis-names.
   ⇒ n15 next (rule 2), with P / hall/q and the path hallucination rate as the
   primary metrics, not first_ok. **Not launched — report first, per the rule.**
3. B: exact 0.025 = control, nav 3 W / 10 L, worst path metrics of the fleet.
   B **hurts**, alone and on top of A (mask_ab < mask_a) and bind
   (mask_b_bind < mask_bind). Rule 3 not met.
4. `mask_a ≥ mask_b` (and ≥ everything graph-side) ⇒ keep A as the cheap
   mechanism. The e13b `decode_trail` contrast holds: the zero-shot *hard* trail
   did nothing; the *soft* decision row (gain 3, non-negative boost on the
   current node's neighbours, no hard block) trained under SFT is the best arm.
   Note the gain itself did not train — this is a fixed structural prior that
   the LoRA learned to use.
5. (c): `mask_d` ≈ `mask` (30 vs 32, 3 W / 5 L, halluc 0.144 vs 0.166) while
   `text_edges` = 1.0. The soft-edge upper bound is *not* an upper bound at 400
   steps — a fresh per-edge token language is not learned in 2 epochs, whereas
   text edges are read zero-shot. Don't read D as "the graph side can't carry
   edge information"; read it as "D needs more steps than this loop allows."
6. `mask_bind` alone helps (36/50, halluc 0.088, lowest probe hall/q 1.05, at
   the cost of probe recall 0.66) ⇒ combine with the best pathway:
   **`mask_a_bind`** added to `scripts/e18_arms.sh` (1823d43), not yet run.

## Proposals for the morning (nothing launched)

1. **`ARM=mask_a_bind sbatch scripts/e18_n10_sft.sbatch`** on n10 — the rule-6
   combination; ~45 min wall.
2. **n15** (`DATA_SPLIT=$PROJ/data/n_15_vllm/gen/nav_n15_gemma_data/split`):
   `mask`, `mask_a`, `mask_bind`, `mask_a_bind`, `text_edges`. Sibling error
   becomes measurable (duplicate type prefixes) and `text_edges` may stop
   saturating. Calibrate s/step first (the n15 prompts are longer).
3. **Shrink the eval noise** before any "X beats Y" claim: the readout driver
   accepts `--permutation-seed 32 42 58` (3 relabellings × 50 tasks, paired
   across arms) — ~8 GPU-min per arm per seed via `e18_n10_readout.sbatch`
   (add the flag). Alternatively evaluate on more test graphs.
4. Drop B from the next fleets unless someone wants to debug why a learned
   structural query at every position hurts (hypothesis: `sk_gain` 1.0 at init
   adds a Ψ-similarity logit at *all* positions, not just decision steps — the
   opposite of A's selectivity; a `struct_keys_gain_init=0` arm would test that).

## Bugs found and fixed tonight (all pushed; betty at 1823d43)

* `decision_gain` built on CPU → first wrapper parameter → eval loader /
  inference took `cpu` as the model device after reload; A arms scored 0/10
  with per-sample crashes and the probe died (d33e350 + device-placement test).
  Readout re-run on the saved checkpoints via the new
  `scripts/e18_n10_readout.sbatch` (eval + probes on an existing run dir).
* `RUN_DIR=$(ls -dt ${RUN_NAME}_*)` matched sibling arms: 7731157 probed the
  mask_d checkpoint as `mask` (9de06d0; the mask probe was re-run, 7731237).
* `inference.py` passed `decision_maps=` to the non-learnable
  `GraphMaskLLM.build_structural_mask`, which did not accept it → TypeError at
  decode for `graph_mask_llm` (364e644).
* Collator `Batch` iterated as `(key, value)` pairs in the bind path
  (bc8a303, earlier in the evening).

## What I'm unsure about

* Whether 8 pts of same-checkpoint eval noise is typical or this corpus is
  unusually knife-edge; the permutation-seed readout would tell.
* `decision_gain_init=3.0` is untuned and *cannot* move under this schedule
  (see Telemetry); the A effect may be sensitive to it (a 1.0 / 6.0 sweep is cheap).
* B's failure is confounded by `sk_gain` pinned at 1.0 for the same reason —
  `struct_keys_gain_init=0.0` is the fair test.
* D's failure to beat the control may be the 400-step budget, the scale rule
  (soft tokens at mean name-token norm), or the splice-after-BOS position — not
  separated.
* The probe truncates some lists at `max_new_tokens=256` (small recall loss,
  same for every arm).
* Readout evals overwrite `eval_logs/cross_eval/` — the in-process per-task
  JSONs for `mask` are gone (only the per-graph counts survive in the log).
