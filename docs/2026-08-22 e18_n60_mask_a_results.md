# e18 — mask_a (decision gating) at full scale on n60_v3, paired with a fresh control (2026-08-22)

Follow-up to `docs/2026-08-21 e18_n10_fleet_results.md`, where `mask_a` was the
best graph-side arm on n10 (38/50 vs 32/50, hallucinated-edge rate .038 vs .166).
The question here was whether that carries to the full-scale corpus the e17
numbers live on. Two SFT jobs, same commit (cfbb3cf), same data, same seed
handling, launched together:

| job | arm | wandb | run dir | wall |
|---|---|---|---|---|
| 7771434 | `mask_a` (`decision_gating=true`, `decision_gain_init=3.0`) | 9a6lwgfj | `outputs/e18_identity/e18_n60_mask_a_9a6lwgfj` | 4 h 16 |
| 7771441 | `mask` (e17 recipe control, nothing new) | rlkmq6hj | `outputs/e18_identity/e18_n60_mask_rlkmq6hj` | 5 h 16 |

Recipe: `scripts/e18_n10_sft.sbatch` with `MAX_STEPS=-1 EVAL_EPOCH_INTERVAL=1
PROBE_TEST_GRAPHS=3` on `data/n_60_vllm_v3/gen/nav_n60_gemma_data/split` —
31B-it bf16, stage-1 LoRA warm start, navigator-GT tower, binary edges,
`decode_consistent`, LR 2.5e-4 linear to 0 over 3 epochs = 936 steps, 1.7 s/step.
Held-out set: 7 test graphs × 12 tasks = 84 tasks (graphs data_gen_052–058).
Each job scored the held-out set four times: the built-in generation eval at the
end of every epoch (the harness cn7ub88q's 72.6 % came from) and once more
post-training from the saved checkpoint (the reload harness the e17 suite uses).
Then the neighbour-naming probe on 3 test graphs and 3 train graphs (48 rooms
each, 144 queries per set).

## Results

### Navigation accuracy, per epoch (built-in eval)

| epoch | A | fresh control | A − ctrl | e17 control cn7ub88q (built-in) |
|---|---|---|---|---|
| 1 | 36/84 (42.9 %) | 32/84 (38.1 %) | +4 | 37/84 |
| 2 | **55/84 (65.5 %)** | 47/84 (56.0 %) | +8 | 59/84 |
| 3 | 47/84 (56.0 %) | 47/84 (56.0 %) | 0 | 61/84 (72.6 %) |

Post-training re-score of the epoch-3 weights (reload harness): A **49/84
(58.3 %)**, control **45/84 (53.6 %)**. Paired on the 84 tasks: A wins 14, loses
10, both solve 35, neither 25 — exact binomial p = 0.54. Per graph 052…058,
A 6 6 6 8 6 8 9 vs control 6 5 8 7 6 7 6.

Same-checkpoint noise on n60: A 47 → 49 and control 47 → 45 between the two
scorings of identical weights, i.e. ±2 tasks on the total, up to ±3 on one
graph. (On n10 it was ±4 of 50.)

Validation loss (teacher-forced, epoch 3): A 0.121, control 0.115 — A is not
better on loss, as on n10.

### Path metrics (the stable readout)

Built-in eval, per epoch — hallucinated-edge rate / valid-path rate:

| epoch | A | control |
|---|---|---|
| 1 | .073 / .43 | .071 / .39 |
| 2 | **.027** / .65 | .060 / .57 |
| 3 | **.039** / .56 | .085 / .57 |

Post-training (reload harness), per test graph 052…058:

| | A | control |
|---|---|---|
| hallucinated-edge rate | .04 .05 .00 .02 .05 .00 .05 (mean ≈ .03) | .08 .14 .03 .05 .11 .02 .10 (mean ≈ .076) |
| edge validity | .96 .95 1.00 .98 .95 1.00 .95 (≈ .97) | .92 .86 .97 .95 .89 .98 .90 (≈ .92) |
| valid-path rate | .50 .50 .50 .67 .50 .67 .75 (≈ .58) | .50 .42 .67 .58 .50 .58 .50 (≈ .54) |

A's hallucinated-edge rate is strictly lower on **7 of 7** graphs (sign test
p = .016) and roughly half the control's, the same ratio the n10 fleet showed.

### Neighbour-naming probe (144 queries per set, identical queries for both arms)

| set | arm | first_ok | exact | P | R | sib/q | hall/q |
|---|---|---|---|---|---|---|---|
| test (052–054) | **A** | **.979** | .007 | **.658** | **.561** | .507 | **1.72** |
| test | control | .819 | .007 | .491 | .484 | .562 | 2.42 |
| train (3 graphs) | **A** | **.993** | .014 | **.658** | .547 | .382 | **1.84** |
| train | control | .924 | .000 | .493 | .510 | .424 | 3.28 |

Paired per query on the test set: first_ok A wins **26 / loses 3** (p < .001);
precision higher on 100 queries, lower on 33; hallucinated names fewer on 75,
more on 34; sibling errors lower on 30, higher on 19 (≈ tie). Recall is the weak
axis for both: lists are truncated at `max_new_tokens=256` and the model stops
early on big neighbourhoods (n_named 5.6 vs 6.2 true).

Two things this probe says that n10 could not:

* **There is no train/test gap in neighbour identity.** A scores the same on the
  graphs it trained on as on held-out ones (P .658 both, hall/q 1.84 vs 1.72);
  the control is *worse* on train graphs for hallucinated names (3.28 vs 2.42).
  The model does not hold the neighbour sets cleanly even where it has seen the
  graph 3 times; over-naming is a property of how the graph reaches the LLM, not
  of generalisation.
* **Sibling confusion exists at n60 (~0.5 per query) and A does not fix it.**
  n10 has unique type prefixes so sib/q was structurally 0 there; at n60
  (`crew_dorm_1/2/3`, `cell_1/2/3`, `turbine_1/2/3`) half of all queries contain a
  wrong-index sibling, equally for both arms. That is the e17 "wrong endpoint"
  failure mode showing up at the identity level, and it is the open problem.

### Gain telemetry

`e18/decision_gain` 3.000 → 2.964 over 936 steps, gradient norm 0.002–0.009
throughout (always pushing down, slightly). As predicted by the drift bound
(∫LR ≈ 0.12) the scalar is pinned by the schedule; A at n60 is, as at n10, "the
LoRA/tower trained under a fixed gain-3 soft decision row". `decision_gain_init`
remains an untuned hyperparameter.

## What this means for the hypotheses

**H-A (decision gating lets the LLM pick the right next node):** half-confirmed.
The mechanism does exactly what it was built for at the level it acts on — at
n60 the model invents about half as many edges (.03 vs .076 on 7/7 graphs) and
names a true neighbour first 98 % of the time vs 82 % — and this holds on
training graphs as much as on held-out ones. But at n60 that does **not** turn
into more solved navigation tasks: 49 vs 45 on the reload harness, 47 vs 47 on
the built-in one, p = .54. On n10 the accuracy gain was 6/50 with the same
mechanism; on n60 it is ≤ 4/84 and inside the noise.

**Why the edge gain stops paying at n60** — two observations that fit together:
(1) the remaining failures at n60 are not mostly invented edges any more (the
control's hallucinated-edge rate is already .076, vs .166 on n10); (2) the probe
shows ~0.5 sibling confusions per query for *both* arms and recall ≈ .5. So the
n60 failure mass has moved to *wrong-but-real* endpoints and *missing*
neighbours, which a non-negative boost on the current node's true neighbours
does not address — it never suppresses a sibling that is itself a neighbour,
and it does not make the model enumerate more of the set.

**H-budget / H-schedule:** A's epoch-2 → epoch-3 drop (55 → 47, built-in) while
the control stayed at 47 is beyond the ±2 same-checkpoint noise, but with one run
per arm it cannot be separated from run-to-run variance. The last epoch is the
low-LR tail of the linear schedule; a constant-LR or longer-plateau run is the
cheap test.

## On the 14-point gap to cn7ub88q

Both fresh runs sit ~14 points under cn7ub88q's built-in 72.6 % (61/84). I
checked for a harness or config cause:

* wandb config diff cn7ub88q → rlkmq6hj: only `gnn.post_fusion` True → False
  (cn7ub88q carried the pf modules with the tanh gate at ≤ 0.005 — inert, see
  `zero-init-gates-never-open`) plus the e18 flags, all off. Same LR, schedule,
  epochs, data, eval settings, warm start.
* Commits 190075b → cfbb3cf: no change in the eval harness (`callbacks.py` only
  adds e18 telemetry, `train_v3.py` only records flags at save, the collator
  adds a `decision_map` the plain mask ignores). `gnn_llm.py` had the mask
  forward and the decode bias-row machinery refactored into helpers (+1180
  lines for the e18 pathways); the regression tests pass but I have not proved
  bit-equivalence against the 190075b mask.
* The 08-22 e17 suite re-scored cn7ub88q's own checkpoint at **63.1 % (53/84)**
  on the reload harness (`e17-eval-suite-jobs`; 18/84 discordant vs its built-in
  72.6, McNemar p = .096). A's reload 58.3 % is within 4 tasks of that.

So the honest reading is that 72.6 was the top of a same-config band whose width
we have now seen three times (cn7ub88q 72.6 / vhrn7jce 61.9 / rlkmq6hj 56.0 on
the built-in harness), and that single n60 runs cannot resolve differences under
~10 points. Anything claimed at n60 needs either paired multi-seed runs or the
path/probe metrics, which pool hundreds of events per arm.

## Proposals (nothing launched)

1. **Stop using single-run n60 accuracy as the decision metric.** Use the
   hallucinated-edge rate (300 edges/arm) and the paired probe (144 queries,
   sign tests) as primaries; report accuracy with its ±2–4 band.
2. **Target sibling confusion directly** — it is the failure A leaves untouched
   and the one n60 makes measurable. Candidates already in the arm table:
   `mask_a_bind` (binding loss separates `crew_dorm_2` from `crew_dorm_3` in the
   residual at mention time) and, if the LR-pinned `sk_gain` was B's problem,
   `mask_b` with `struct_keys_gain_init=0`. Run them on n10 at 1200 steps first
   (the rule-6 combination was never run), then the winner on n60 paired with
   a control, two seeds each.
3. **Recall**: raise the probe's `max_new_tokens` (lists of 6–8 rooms truncate at
   256 tokens with reasoning), and split recall into "stopped early" vs "missed
   while still listing" — only the second is an identity failure.
4. **Schedule check** for the epoch-3 drop: one `mask_a` n60 run with a constant
   LR (the RL trainer's setting) or 4 epochs, read at every epoch.
5. **Bit-equivalence of the refactored mask**: score cn7ub88q's checkpoint with
   the built-in harness at cfbb3cf (one `e18_n10_readout.sbatch`-style job, ~25
   GPU-min). If it reproduces ~63–72 the 14-point gap is variance; if it lands
   at ~56 the refactor changed the mask and e17 ↔ e18 numbers are not comparable.

## Artifacts

* Run dirs (checkpoint, `eval_logs/cross_eval/data_gen_05x.json` per-task,
  `path_metrics_*.png`): see the table at the top.
* Probes: `results/e18_probe/e18_n60_{mask_a,mask}_{test,train}.json`
  (`per_graph.<graph>.records[]` with `truth/named/first_ok/precision/recall/
  sibling_err/hallucinated`).
* SLURM logs: `slurm-e18-n60-7771434.out`, `slurm-e18-n60-7771441.out` (use
  `grep -a`; progress bars make them binary). Per-epoch accuracy and path
  metrics are in wandb only (`eval/accuracy`, `eval/acc/data_gen_05x`,
  `eval/hallucination_rate`, `eval/valid_path_rate`, `eval/path_optimality_rate`).
* The control hit 3 caught per-sample OOMs (runaway 20k-token re-query prompts
  after an unparseable first plan, 47–57 GiB prefill); A hit none. These are
  scored as failures for the sample, same as the e17 control's 2. Pre-existing;
  not fixed here.

## What I'm unsure about

* Whether A's 55 → 47 in the last epoch is a regression under the low-LR tail
  or a draw from the ±4 band — one run per arm cannot tell.
* Whether the 14-point gap to cn7ub88q is pure variance: the mask-forward
  refactor between 190075b and cfbb3cf is untested for bit-equivalence
  (proposal 5 settles it for ~25 GPU-min).
* `decision_gain_init=3.0` is untuned and cannot move under this schedule; the
  n60 result may look different at 1.0 or 6.0.
* The probe's recall numbers conflate list truncation with genuine misses.
* 3 of 7 test graphs for the probe was a wall-time cap; the other 4 are unprobed.
* The reload harness and the built-in harness disagree by up to 9 points on the
  same weights (cn7ub88q); I used both here and report which is which, but the
  field still lacks a single canonical n60 number per checkpoint.
