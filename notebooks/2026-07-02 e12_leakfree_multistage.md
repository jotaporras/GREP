---
tags: [experiment, e12, graph-injection, multistage]
date: 2026-07-02
status: running (betty job 6917889)
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

| stage | paired run | config | e12 delta |
|---|---|---|---|
| 2: PE-only edge-list reconstruction | `e9_ms_stage2_3bitwckz` | rpearl_gt_llm d2048 L5, init_lora `e9_ms_stage1_sqgk4o3j` frozen, lr 5e-3, 4 ep, edges present | `injection_scope=exclude_supervised` |
| 3: joint PE+LoRA, no edges | `kfuu2djo` (gen acc **0.10**) | lr 2.5e-4, 3 ep, `text_edge_list=none`, full 10-graph eval | `injection_scope=prompt_only` |
| 3': joint, with edges (parity) | `6lefhd76` | same, `text_edge_list=present` | `injection_scope=prompt_only` |
| diag | — | `diag_injection_ablation.py` on the no-edge stage-3 checkpoint | — |

## 4. Falsifiable predictions

1. **Stage 2 trains slower / to a worse edge-list loss** than 3bitwckz — the
   shortcut is gone, so reconstruction must route through the anchor node's ψ.
   A *matching* loss with the exclusion would itself be informative (shortcut was
   never load-bearing).
2. **Stage 3 no-edge gen acc lands above the 0.10–0.16 floor** (paired: 0.10).
   Reference points from e11: mask family reached 0.39–0.48.
3. **Diagnostic**: prompt_only − no_injection NLL gap grows well past ms_stage3's
   −0.20 nats, and train_style ≈ prompt_only (no residual leak signature).
4. **Parity**: with-edges arm stays ≈0.87–0.93 (prompt_only must not regress the
   easy regime).

If 1 holds but 2 fails: the additive read-out (ψ added into q/k/v at scene mentions)
may be too weak an interface even leak-free → escalate to decode-time injection
([[e11_decode_time_injection_design]]) or scaled leak-free alignment pretraining.
