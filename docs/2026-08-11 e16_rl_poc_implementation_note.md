# e16 RL PoC — implementation note and the frozen-Ψ-tower defect

**Status: development paused 2026-08-11. This note records what was built, what
is wrong with it, and the concrete fix, so work can resume under supervision.**

## 1. The defect, in one paragraph

The GRPO trainer (`src/prism/training/trainers_rl.py`) **freezes the Ψ tower**
— the GT/PE stack that produces the graph signal — at construction
(`GraphGRPOTrainer.__init__`, the block marked "v1: freeze the Ψ tower").
Only the LLM's LoRA adapter receives gradients. This inverts the purpose of
e16: the reason for adding RL was that SFT gives the PE weights no meaningful
learning signal, and RL rewards were supposed to be that signal. With the
tower frozen, RL trains the LLM on top of a fixed graph signal — and in the
PoC that signal was **randomly initialized** (see §3), i.e. noise. The 300-step
plaza run therefore could not, by construction, teach the graph channel
anything. The decision was flagged as open question Q6 in the plan but was
never surfaced for an explicit go/no-go, which it needed.

## 2. What the PoC run actually was

- Run: `outputs/e16_rl_training/e16_rl_gt_llm_gemma-4-31b-it_r16_4bit_n4tb8hpr`
  on plaza; wandb run `n4tb8hpr`, tag `e16_rl_training`.
- 300 GRPO steps, 22 h 47 m (~274 s/step), group size 8, max completion 1024,
  lr 1e-5, rank-16 LoRA, nf4 policy on GPU 1, nf4 vLLM rollout engine on GPU 0.
- From scratch: `gnn.arch=gt_llm`, `pe_node_features=word_embeddings`,
  **no navigator carry** (see §3), Ψ tower random and frozen.
- Result: no sustained reward ascent. Shaped reward 1.09 → 1.21 (mid-run) →
  1.10 (end); `full_path_valid` 0.08 → 0.18 → 0.09; keyword flat at 0.41;
  format pinned at max from step 1. Consistent with "the graph channel is
  noise and the LoRA can only memorize slowly."
- What the run DID validate (all still good): the vLLM graph engine at 31B,
  disk-light nf4 serving with a LoRA adapter on one 48 GB A6000, the
  two-GPU trainer/engine split, the GRPO loop end to end (rollouts → rewards
  → loss → optimizer → run-dir save), and a throughput baseline.

## 3. Two upstream discoveries that shaped (and undermined) the PoC

1. **The e14 "GT" checkpoints are the mask architecture.**
   `e14v3_n60_gt_6ep` records `architecture: learnable_graph_mask` — a dense
   attention-score bias (log-gated ΨΨᵀ, non-edges hard-blocked) on every
   layer. Paged flash-attention cannot apply a per-request dense bias, so
   these checkpoints are not vLLM-servable and cannot seed RL-with-vLLM. The
   "GT" in their name refers to the GT tower *producing* Ψ, not the additive
   `gt_llm` architecture.
2. **The e9 navigator weights do not fit the additive tower.**
   `path_navigator_gt.pt` is the mask arch's Ψ producer (NavigatorPE: a
   pe_gcn plus 2 GT blocks). `gt_llm` builds a `SemanticGraphTransformer`
   with a different module structure; the strict state-dict load refuses it.
   So the from-scratch run had no warm start for the tower — hence random.

## 4. Why the freeze existed (and why neither reason survives scrutiny)

The freeze was justified by two mechanisms:

- **Per-prompt Ψ cache**: transports are built once per prompt and reused
  every step — only valid if the tower never changes. *Fix: rebuild
  transports each optimizer step. Cost is negligible against 274 s/step.*
- **LoRA-only engine sync**: the rollout engine receives policy updates as a
  LoRA adapter hot-swap; the tower is not part of the adapter. *This turns
  out not to matter: the engine never holds tower weights at all. Ψ ships
  per-request as a tensor computed driver-side by the policy's own tower, so
  a learning tower reaches rollouts automatically once transports are rebuilt
  per step. No new sync mechanism is needed.*

## 5. The v2 fix (concrete change list)

All in `src/prism/training/trainers_rl.py` unless noted.

1. **Remove the freeze** — delete the `requires_grad = False` loop over
   `core.structural_parameters()` and `core.pe_norm` in `__init__`.
2. **Make the loss side differentiable through the tower.** This is the real
   work item. Today the loss forward arms *cached, detached* Ψ tensors
   (`_transport_cache` → `_psi_for_rows` → `_pe_signal`), so even unfrozen,
   no gradient reaches the tower. The loss path must rebuild Ψ per micro-batch
   through the live tower (`build_pe_signal` on the policy's core model) so
   the GRPO loss backpropagates into the PE weights. Standard GRPO covers the
   rest: the gradient flows through the loss-side log-probs, which now depend
   on the tower.
3. **Invalidate the rollout transport cache every optimizer step** (or key it
   by `(prompt, lora_version)`); rebuild driver-side transports so rollouts
   sample under the current tower.
4. **Two-group learning rate** — reuse the SFT trainer's
   `structural_lr_mult` optimizer grouping (`trainers.py`,
   `create_optimizer`) so the tower trains at its own rate.
5. **Saving already works** — `save_run_dir` writes `gnn_weights.pt` from the
   live tower; nothing to change.

## 6. Open decisions for the supervised session

- **Warm start vs. cold start for the tower.** Options: (a) RL-from-scratch
  with the tower learning from reward alone (sparse, slow, but the purest
  test of the hypothesis); (b) first pretrain a navigator-style Ψ producer
  *for the additive architecture* and load it via `gnn.pe_gt_from`; (c) first
  SFT-train an additive `gt_llm` checkpoint, then RL from it with everything
  unfrozen. Choice shapes the betty run (M6).
- **`completions/clipped_ratio = 1`** in trl's logs: trl sees no
  EOS-terminated completions from the custom rollout path. Likely a stop-token
  accounting artifact, but verify before trusting length-based metrics.
- **KL term**: `beta` is forced to 0 in v1 because trl's reference model would
  compute Ψ-free log-probs. With an unfrozen tower this stays true; revisit
  only if drift becomes a problem.
- Remaining M4 exit criteria: reward-ascent evidence (needs v2), checkpoint
  reload + dual-backend eval of an RL run dir.

## 7. What is done and trustworthy (unchanged by the defect)

- `src/prism/models/vllm_graph/` — Ψ-through-multimodal-channel vLLM plugin,
  Gemma-4 port, bnb/nf4 + LoRA serving, engine policy checks. Full test suite
  green on Mac CPU and plaza GPU (noop invariant, HF/vLLM parity, KV-shared
  refusal, eval backend, rewards, save-dir round-trip, GRPO smoke).
- `--backend vllm` eval path (M2), pending its real-checkpoint parity run —
  which now requires an additive checkpoint (same blocker as §6a).
- `requirements-rl.txt` — the working plaza environment recipe (torch
  2.11/cu128, vllm 0.26.0+cu129 release wheel, transformers <5.15), which is
  also the recipe for the betty `GREP-PRISM-rl` env (M6).
