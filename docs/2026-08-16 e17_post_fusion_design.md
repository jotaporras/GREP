# e17 — Post-fusion graph conditioning (design sketch, v0)

**Status: draft for discussion — Claude's best guess, to be iterated.**

## 1. Why change the architecture

The e16 RL campaign produced a clean negative result across every cheap lever:


| run      | init           | data                             | LR           | outcome                                                                    |
| -------- | -------------- | -------------------------------- | ------------ | -------------------------------------------------------------------------- |
| [0ai15jpo](https://wandb.ai/alelab/GREP-PRISM/runs/0ai15jpo) | [nyvi1tww](https://wandb.ai/alelab/GREP-PRISM/runs/nyvi1tww) (66%) | n30                              | 7e-6         | +5pt to ~70%, saturates by step 100                                        |
| [3rb8hbwk](https://wandb.ai/alelab/GREP-PRISM/runs/3rb8hbwk) | [ai8c2bm0](https://wandb.ai/alelab/GREP-PRISM/runs/ai8c2bm0) (84%) | n30                              | 7e-6         | flat 83–85%; train reward saturates                                        |
| [sjahr1l7](https://wandb.ai/alelab/GREP-PRISM/runs/sjahr1l7) | [ai8c2bm0](https://wandb.ai/alelab/GREP-PRISM/runs/ai8c2bm0)       | n60_v3 (42% init, huge headroom) | 7e-6         | flat on n60 (39–45%) **and** flat train success (0.60/0.49/0.62 by thirds) |
| [uvfe426y](https://wandb.ai/alelab/GREP-PRISM/runs/uvfe426y) | [ai8c2bm0](https://wandb.ai/alelab/GREP-PRISM/runs/ai8c2bm0)       | n60_v3                           | 3.5e-5 const | flat train success                                                         |
| [pwcgk2aa](https://wandb.ai/alelab/GREP-PRISM/runs/pwcgk2aa) | [ai8c2bm0](https://wandb.ai/alelab/GREP-PRISM/runs/ai8c2bm0)       | n60_v3                           | 7e-5 const   | flat train success; no instability                                         |


Key diagnostic facts from [sjahr1l7](https://wandb.ai/alelab/GREP-PRISM/runs/sjahr1l7)'s group structure: only ~~13% of GRPO groups were all-fail and ~28% all-success — **~~60% of groups carried genuine advantage signal for 300 steps**, and the policy did not improve *on its own training prompts*. Raising LR 10× changed nothing (and nothing blew up, i.e. the gradient direction is small *and* unproductive, not merely under-scaled).

Conclusion we're acting on: the current fusion point cannot express the behavioral changes the reward asks for. Every existing arch injects the graph signal **inside self-attention** (q/k/v additive, attention bias, or rotation) — it can change *where the LLM looks*, but the reward asks the model to change *what it does with what it sees* (path selection, backtracking, cost comparison). The mask arch in particular only modulates node-token↔node-token attention scores; by mid-network the LLM has already committed its representation of the graph, and a log-gate on attention among node tokens is a very low-bandwidth actuator for "choose the other branch".

## 2. Design principle

Move (part of) the graph signal **after** the LLM's fusion of the prompt — inject into the residual stream of late layers and/or the pre-logits representation, where gradients from the reward have a short, direct path into the graph tower, and where the signal can steer *token selection* rather than attention routing.

Two consequences we want by construction:

1. **Short gradient path for RL.** In the mask arch, reward → logits → 40+ layers → attention-score bias → log-gate → ψψᵀ → tower. Post-fusion: reward → logits → (0–4 layers) → tower. If RL still can't move the policy with a 2-hop gradient path, the hypothesis "RL + this tower can't learn this task" is falsified much more decisively.
2. **Ψ=0 ⇒ no-op preserved.** Same invariant the vLLM plugin locked: gated residual write with `tanh(gain)` initialized at 0 means the warm-started model is bit-identical at init, so we can warm-start from [ai8c2bm0](https://wandb.ai/alelab/GREP-PRISM/runs/ai8c2bm0)'s LLM weights (tower rides along) without an SFT re-run *for the v1 probe*.



## 3. Candidate designs



### A. Late-layer gated residual injection (recommended v1)

At each of the last K decoder layers (K≈2–4, config `post_fusion_layer_scope`, reusing the `dense_top_half`-style scope machinery at `gnn_llm.py:426`), add a per-position gated write into the hidden states:

```
h[b, p, :] += tanh(pf_gain_l) * pf_norm_l( pf_proj_l( ψ[node_of(p)] ) )   for node positions p
```

- `pf_proj: Linear(d_gt → hidden)` — new module; note `learnable_graph_mask` has **no** projection to LLM hidden today (only the additive arch's `pe_proj`, `gnn_llm.py:1096`), so this is new capacity, per-layer or shared.
- Position→node mapping reuses the existing injection maps (`build_injection_map` `gnn_llm.py:2434`, `decode_consistent` semantics) — decode-side, the same pre-hook pattern as `BatchedMaskDecodeInjector` (`trainers_rl.py:605`) arms the per-step row; the injector only needs to supply ψ for the *current* token's node (or nothing for non-node tokens).
- Implementation is a residual-stream analog of the existing `_struct_bias` contract: arm a `_pf_signal` tensor on the core, read it in a decoder-layer forward wrapper. No trainer changes beyond adding `pf_*` modules to `structural_parameters()`.

Why v1: smallest delta from the current codebase; keeps the mask arch untouched (can even run mask + post-fusion together, config-gated); preserves the warm-start no-op; and it is the cleanest test of the short-gradient-path hypothesis.

### B. Logit-bias head ("graph steers the sampler")

A small head maps (last hidden state at position t, ψ of candidate next-node tokens) → additive logit bias over the vocabulary tokens that name graph nodes. Zero-init final layer ⇒ no-op at init.

- Most direct actuator possible for path choice; gradient path is 1 hop.
- Cons: needs the node↔vocab-token mapping (exists implicitly in the injectors' node-position machinery, but vocab-side is new work: multi-token node names need span scoring, not single-token bias); risks degenerating into a learned lookup that ignores the LLM. Good **v2 / ablation**, not v1.



### D. Graph-generated LoRA (Javier's idea 1)

A LoRA adapter on late-layer projections where **one of the two factors is produced by the GT** rather than learned as a free parameter: e.g. for a target weight `W`, the update is `ΔW = B · A(ψ)` with `A(ψ)` emitted per-graph by the tower (hypernetwork-style), `B` a trained free matrix (zero-init ⇒ no-op at init, same warm-start story as A).

- The graph then modulates the *computation* (the weights), not the activations — per-graph task conditioning rather than per-token signal injection. Strictly more expressive than A for "this graph changes how you plan" and composes with the existing LoRA (separate adapter name).
- Design choices to pin down: which factor is graph-generated (generating the down-projection `A` from pooled ψ is cheapest: `d_gt → r·hidden` head); per-graph (pooled ψ) vs per-node (needs a routing story for which tokens see which node's ΔW — per-graph is the sane v1); which modules (late-layer `down_proj`/`o_proj` first).
- Gradient path: reward → logits → few layers → `B`/head → tower. Comparable to A. Main risk: rank-r bottleneck × pooled-ψ bottleneck may be *too* low-bandwidth; and peft won't express input-dependent factors, so it's a manual wrapper on the target modules (moderate implementation cost, more than A, far less than C).



### E. GT node distribution ⊕ LLM vocab distribution (Javier's idea 2; supersedes B)

The GT emits its own probability distribution over **nodes** (e.g. score each node against the current decode state: `p_gt(node) ∝ exp(score(h_t, ψ_node))`), the LLM emits its usual vocab distribution, and the two are merged into a reweighed vocab distribution — a pointer/copy mechanism in the RAG/CopyNet lineage:

```
p(tok) = (1 − g_t) · p_llm(tok) + g_t · Σ_{node} p_gt(node) · p_spell(tok | node)
```

with a learned (zero-init ⇒ no-op) gate `g_t = σ(w·h_t + b)`, and `p_spell` distributing node mass over the tokenization of each node's name.

- This is the most honest division of labor: the GT owns *which node comes next*, the LLM owns *fluency and everything else*. GRPO's advantage signal lands directly on `p_gt` — a 1-hop gradient path into the tower, the shortest of all candidates.
- The hard part is `p_spell` for multi-token node names: mass must be placed on the *next* token of a partially-spelled name (prefix-tracking over the trie of node-name tokenizations during decode). The injectors' node↔position machinery gives us node spans in the prompt; the vocab-side trie is new but self-contained work. Constrained-decoding literature has this exact machinery.
- Also the best *interpretability* payoff: `g_t` and `p_gt` are directly inspectable per step (when does the model consult the graph, and what does the graph want).



### C. Cross-attention adapter block (full post-fusion)

Insert 1–2 new cross-attention blocks after the last decoder layer: queries from the LLM hidden states, keys/values from Ψ (per-node embeddings), gated residual output. This is the "proper" post-fusion arch (Flamingo-style) and the strongest version of the idea — the LLM can *query* the graph per decoding step.

- Cons: new trainable block (~2·hidden·d_gt + hidden² per block ≈ 100M+ at Gemma-31B hidden size) almost certainly needs an SFT stage before RL (random cross-attn under RL-only training is a known dead end). This is the destination if A shows life; too heavy for the first probe.



## 4. v1 experiment plan (cheap, decision-oriented)

1. **Implement A** behind `arch=post_fusion_mask` (mask arch + late residual injection, both towers sharing `pe_model`). Ψ=0 no-op test + parity test vs `learnable_graph_mask` with gain=0, in the style of the existing mask tests.
2. **Sanity SFT-free RL probe (3 h, betty)**: warm-start [ai8c2bm0](https://wandb.ai/alelab/GREP-PRISM/runs/ai8c2bm0), n60_v3, t=1.15, LR 3.5e-5 const, `structural_lr_mult=1`, 120 steps — *identical protocol to the LR probes*, so [uvfe426y](https://wandb.ai/alelab/GREP-PRISM/runs/uvfe426y)/[pwcgk2aa](https://wandb.ai/alelab/GREP-PRISM/runs/pwcgk2aa) are the controls. Success signal: train `full_path_valid` thirds ascend (anything like 0.55→0.65+ is a hit given the flat controls).
3. If ascent appears → full 300-step run + the standard dual ladder (n30 + n60_v3 test splits, no-ICL) logged to wandb.
4. If flat again with a 2-hop gradient path → strong evidence the bottleneck is not the fusion point but the tower/task coupling (or RL-from-verifiable-reward on this task per se); pivot to C-with-SFT or rethink the reward.

Tag stays `e16_rl_training` for the probes unless we declare e17 — **decide before launching** (tag discipline: never drift mid-campaign).

## 5. Open questions for Javier

- Q1: Candidate ranking. Javier's two ideas are D (graph-generated LoRA) and E (node-dist ⊕ vocab-dist merge, superseding B). Claude's current ordering by decision-value-per-week: **E ≈ A first** (E has the shortest gradient path and the cleanest division of labor but needs the spell-trie; A is the smallest code delta), then D, then C. To discuss.
- Q2: OK to keep the mask bias active alongside the residual injection in v1 (isolates the *added* pathway), or do you want post-fusion *replacing* the mask (isolates the *fusion point*)?
- Q3: Warm-start question: A preserves init-no-op so [ai8c2bm0](https://wandb.ai/alelab/GREP-PRISM/runs/ai8c2bm0) works as-is; if you'd rather give the new pathway an SFT warm-up first (teacher-forced on n60_v3 train), that's ~1 day extra but de-risks C later.
- Q4: e17 tag now, or keep probing under e16?



## 6. Code touchpoints (from the fusion-point survey)

1. Injection read sites: `gnn_llm.py:58-125` (mask bias consumption) — post-fusion adds a parallel read of `_pf_signal` in a decoder-layer wrapper, not in attention.
2. Layer scoping: reuse `resolve_mask_active_flags` pattern (`gnn_llm.py:423-450`) with a new `post_fusion_layer_scope`.
3. Decode-side: extend `BatchedMaskDecodeInjector` (`gnn_llm.py:2689`) to also arm the per-step ψ row.
4. Trainer: `MaskGRPOTrainer` needs only `structural_parameters()` to include `pf_*` modules (`gnn_llm.py:848`); rollout/loss plumbing (`trainers_rl.py:585,712`) is unchanged in shape.
5. Checkpoint contract: `pf_*` weights ride in `gnn_weights.pt` next to the tower; `load_gnn_config` gets the new arch fields.

## 7. Results (campaign through 2026-08-19)

![e17 eval summary](e17_eval_summary.png)

*(plot: `e17_eval_summary.png`, alongside this note in `docs/`; left = RL held-out curves, right = SFT bars — the two panels use different eval harnesses and are NOT cross-comparable.)*

### 7.1 RL — post-fusion is the first arch that RL actually moves

Full-scale runs (candidate A, mask + post-fusion, warm start [ai8c2bm0](https://wandb.ai/alelab/GREP-PRISM/runs/ai8c2bm0), n60_v3 corpus, t=1.15, constant LR, reward v2, 300 steps, tag `e16_rl_training`). Held-out n60 test, `scalability_evaluation`, no-ICL, n=84:

| GRPO step | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|
| lr 3.5e-5 ([`rw7eg7b6`](https://wandb.ai/alelab/GREP-PRISM/runs/rw7eg7b6)) | 50.0 | 48.8 | 50.0 | 51.2 | 56.0 | **60.7** |
| lr 7e-5 ([`64k0icwg`](https://wandb.ai/alelab/GREP-PRISM/runs/64k0icwg)) | — | 46.4 | 39.3 | 52.4 | 50.0 | 45.2 |

References: SFT init 42.0%, best of the 120-step probe 51.2%. The lr 3.5e-5 curve is monotonic from step 100 and *accelerating* at 300 — 60.7% is a trend, not a lucky checkpoint (+18.7 over init, +9.5 over the probe). lr 7e-5 is jagged and regresses by 300: too hot. Contrast with §1's table: every pre-e17 arch was flat under identical RL protocols. The short-gradient-path hypothesis (§2) is supported — moving the fusion point after attention is what unlocked RL.

Extension `e17_pf_full_lr35_ext300` ([`ge4v7i3s`](https://wandb.ai/alelab/GREP-PRISM/runs/ge4v7i3s), init from [rw7eg7b6](https://wandb.ai/alelab/GREP-PRISM/runs/rw7eg7b6) final, steps 300→600): completed 7h28; held-out evals pending. Caveat from its log tail: train reward saturates by the end (reward 3.8 = max, reward_std 0, `frac_reward_zero_std` 1, entropy 0.13, tower_grad_norm 0) — zero within-group variance ⇒ zero GRPO advantage, so late extension steps coast. If the held-out curve plateaus, the lever is harder data or higher rollout temperature, not more steps.

### 7.2 SFT — post-fusion is inert under teacher forcing

Two SFT rounds (e13f_alpha00_binary recipe + `post_fusion=true`, tag `e17_post_fusion`). Round 1 accidentally trained the pf modules in the structural group (0.012 × 2.5e-4 ≈ 3e-6 effective LR → `pf_gain` absmax ~1e-4, gate never opens) — those runs are pf-dormant **controls**. Fix (commit 9d0b06c): pf modules moved to `base_lr_parameters()`, full base LR. Built-in train_v3 end-of-run eval:

| dataset | round 1 (pf dormant) | round 2 (pf full LR) | Δ |
|---|---|---|---|
| n30_old (n=100) | 66.0 | 66.0 | 0.0 |
| n30_v2 (n=70) | 68.6 | 74.3 | +5.7 |
| n60_v3 (n=84) | 72.6 | 61.9 | −10.7 |

Even at full LR the tanh gate opens only ~0.2–0.5% (`pf_gain` absmax 2–5e-3): the SFT loss simply doesn't ask for the pathway, and the net accuracy effect is noise-to-mildly-harmful. **Verdict: post-fusion's case rests entirely on RL** — consistent with §2's motivation (the pathway exists to give the *reward* a short gradient path; teacher forcing already has all the gradient path it needs).

Open item: round-1 n30_old's 66.0 vs the historical e13f 86% reference is likely a harness difference (train_v3 built-in eval vs `scalability_evaluation`) — verify before reading it as a regression.


## 8. Summary (2026-08-19, for the experiment log)

- Designed and implemented candidate A, post-fusion residual injection: at the deeper half of the global-attention decoder layers, node-token hidden states receive a gated residual write of the projected node embedding, `h += tanh(gain)·RMSNorm(proj(ψ))`, with zero-initialized gains so pre-e17 mask checkpoints warm-start bit-identically; the mask attention bias stays active alongside.
- The design goal is a short gradient path for RL: reward → logits → a few layers → tower, versus the 40+-layer path of the attention-injection architectures.
- Main result: the first architecture RL has ever moved in this project. GRPO at lr 3.5e-5 on n60_v3 (run [rw7eg7b6](https://wandb.ai/alelab/GREP-PRISM/runs/rw7eg7b6), reward v2, 300 steps) took held-out n60 accuracy from 42.0% at SFT init to 60.7% at step 300, monotonic from step 100 and still rising at the end; lr 7e-5 was unstable and regressed, and the flat e16 runs under the identical protocol serve as controls. Note: 60.7% measures generalization to unseen graphs at the trained size (n60), not to unseen sizes.
- Under supervised fine-tuning the pathway is inert: with the pf modules at full base LR, the gates open to only ~0.2–0.5% and accuracy changes are noise-to-mildly-harmful — the post-fusion pathway earns its keep only under reward-driven training, consistent with the design motivation.
- Status and next steps: the 300→600 extension run finished with train reward fully saturated (zero within-group variance, so late steps carry no GRPO advantage) and its held-out evals are pending; no RL checkpoint has been evaluated on other graph sizes yet (n30/n100 ladder pending); candidates E (GT node distribution merged into the vocab distribution) and D (graph-generated LoRA) remain unimplemented.
