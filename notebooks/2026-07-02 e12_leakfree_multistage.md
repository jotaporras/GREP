---
tags: [experiment, e12, graph-injection, multistage]
date: 2026-07-02
status: complete (betty job 6917943; 6917889 = identical first attempt, cancelled at epoch 1 for an eval-breadth fix)
related: ["2026-07-01 e11_injection_asymmetry"]
wandb: alelab/GREP-PRISM (tag e12_leakfree_multistage)
---

# e12 — Leak-free multistage (additive family)

> [!abstract] TL;DR
> e11 fixed the answer-side leak (`prompt_only`) and tripled no-edge generation
> accuracy for the **mask** family. The **additive** multistage pipeline has a second,
> different leak that `prompt_only` cannot reach: stage 2 supervises the edge-bullet
> tokens *in the prompt*, and under `full_sequence` those target tokens carry their
> own node's ψ — the label in q-space. e12 generalizes the fix to the principle
> **injection ∩ loss-target = ∅** (`injection_scope=exclude_supervised`) and reruns
> the e9 multistage chain with both scopes corrected, changing nothing else.

## 1. The stage-2 leak, precisely

Stage 2 trains the PE as an edge-list reconstructor: LLM+LoRA frozen,
`loss_target=edge_list`, edges present. Predicting the target `region_7` in
`• region_5 <=> region_7`, teacher forcing places ψ_{region_7} on the query
positions inside region_7's own span — the assignment of ψ to that span is *indexed
by the label* (the teacher-forced text already says region_7 there). The honest
circuit ("from ψ_{region_5} + Gram geometry infer the neighbor, copy its name from
the scene block") competes with the shortcut ("my own query carries who I am") and
SGD picks the shortcut. Prompt-side scene-block mentions are the legitimate channel
(pairing known from the prompt itself) — the exclusion preserves exactly those.

Evidence this matters: `e9_ms_stage2_3bitwckz` reached edge-reconstruction
convergence yet its stage 3 (`kfuu2djo`) showed only a −0.2-nat generation-compatible
channel and gen acc 0.10 — the stage-2 gate opened (0.27) but onto a circuit trained
against inputs generation (and the stage-3 no-edge prompt) never provides.

## 2. The change

`data.injection_scope=exclude_supervised` (commit `732d480`, default `full_sequence`
unchanged): the collator subtracts the loss-target positions
(`_LOSS_TARGET_COLUMN[trainer.loss_target]`, here `edge_list_idx`) from every
injection span, splitting spans that straddle the block
(`exclude_positions_from_injection_map`). Verified on 8 real corpus examples:
~608 node-mention positions stripped from the edge block per example; all 33 nodes
retain scene-block spans; resulting coverage exactly `full − edge_block`
(`tests/test_injection_scope.py`, 18/18 local + betty).

## 3. Design (betty job 6917889, `scripts/e12_multistage_leakfree.sbatch`)

Single sequential job, paired to the historical chain — ONLY the injection scopes
change:

| stage | paired run | config | e12 delta | e12 run |
|---|---|---|---|---|
| 2: PE-only edge-list reconstruction | `e9_ms_stage2_3bitwckz` | rpearl_gt_llm d2048 L5, init_lora `e9_ms_stage1_sqgk4o3j` frozen, lr 5e-3, 4 ep, edges present | `injection_scope=exclude_supervised` | `0dq55cex` |
| 3: joint PE+LoRA, no edges | `kfuu2djo` (gen acc **0.10**) | lr 2.5e-4, 3 ep, `text_edge_list=none`, full 10-graph eval | `injection_scope=prompt_only` | `wh0537au` |
| 3': joint, with edges (parity) | `6lefhd76` | same, `text_edge_list=present` | `injection_scope=prompt_only` | `gvzylvay` |
| diag | — | `diag_injection_ablation.py` on the no-edge stage-3 checkpoint | — | `injection_diag.json` in `wh0537au` run dir |

## 4. Falsifiable predictions → outcomes (2026-07-02, job 6917943)

1. **Stage 2 trains slower / to a worse edge-list loss** than 3bitwckz. →
   **HELD.** Eval loss by epoch: 0.534 / 0.427 / 0.397 / **0.3885** vs leaky
   0.4375 / 0.3819 / 0.3588 / 0.3525 — slower, converging to near-parity
   (residual +0.036 mean-nats; diluted over the block, so plausibly ~0.3–0.5
   nats on the actual target-node decisions). The descent is attributable
   entirely to the PE pathway (LLM+LoRA frozen, gate cold-started at 0; final
   pe_gain **0.238** ≈ historical 0.27). The honest reconstruction readout is
   learnable; the shortcut bought speed, not capability.
   *(Caveat: stage-2 `eval/accuracy` 0.98 is NOT evidence — edges are in the
   prompt at stage 2, the leaky run scored 0.98 too.)*
2. **Stage 3 no-edge gen acc lands above the 0.10–0.16 floor** (paired: 0.10).
   → **FAILED.** `wh0537au` = **0.09** on all 10 graphs (per-graph 0–0.2).
   Same floor as the leaky chain. Under the identical protocol the mask family
   reached 0.39–0.48 (e11) — including the *parameter-free* mask at 0.39.
3. **Diagnostic**: gap grows well past ms_stage3's −0.16 nats; train_style ≈
   prompt_only. → **PARTIAL.** Decision tokens (n=519):
   train_style 0.599/2.244 · prompt_only 0.595/**2.074** · no_injection
   0.603/2.414. Generation-compatible channel = **−0.34 nats** (2× the leaky
   chain's −0.16, but far from mask's −0.92) with **flat accuracy** — correct-node
   prob 0.089→0.126, no argmax flips. train_style is now *worse* than
   prompt_only: answer-side ψ is OOD noise to a prompt_only-trained model —
   no leak signature, the scopes verifiably worked.
4. **Parity**: with-edges arm stays ≈0.87–0.93. → **HELD (exceeded).**
   `gvzylvay` = **0.99**.

> [!important] Verdict
> The "1 holds, 2 fails" branch fired, with the diagnostic pinning the failure
> mode: the additive channel after leak-free pretraining is **real but ~an order
> of magnitude too weak** (−0.34 nats delivered where ~2 nats/decision are
> needed). Every named implementation failure is now individually eliminated —
> cold-start gate (opened, 0.23), stage-2 label leak (removed; reconstruction
> still learns), answer-side leak (removed; parity 0.99) — and the floor
> persists, while the same fixes moved the mask family to 0.39–0.48. The
> remaining differentiator is the **interface**: ψ consumed as attention
> structure (Gram-mask) works; ψ added as content into q/k/v at prompt mentions
> does not yield a usable free-generation readout. The encoder survives (the
> project-best 0.48 consumes pretrained ψ via the mask); the residual-stream
> injection route is what's dead-ish. Single seed; 0.1-grained metric; but the
> 30-pt cross-family separation is not noise.

Next candidates, in order of expected information: decode-time injection for the
mask family ([[e11_decode_time_injection_design]]; e11-diagnostic train_style
0.865/0.52 is the existence proof for ~0.87-level headroom = the with-edges
baseline); scaled leak-free alignment graded by *generation-time* reconstruction
(teacher-forced reconstruction NLL demonstrably does not transfer); additive-as-
content only if some new interface idea changes the delivery mechanism.
