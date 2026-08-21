# 2026-08-21 — Direction discussion: where the graph channel actually fails

**Status:** discussion record (no new runs). Inputs: every experiment note e10–e18,
the e17 n60 analysis, and a per-hop breakdown of the e17 RL checkpoints produced in a
parallel session. Purpose: fix the shared diagnosis before brainstorming architectures.

## 1. The bar the project has to clear

The intervention (a learned graph channel replacing the textual edge list) must be
(a) trainable, and (b) justified against the text-edges planner on at least one axis:
context efficiency, scalability with graph size, or transfer across sizes. No result to
date clears (b) on any axis; the text-edges arm is ~0.95 at both n30 and n60 and is
assumed size-robust.

A standing design constraint, restated by Javier: the channel must keep **semantic and
structural information merged** — a vague query ("go to where you can find tools")
should resolve to a node (`shed_1`) by combining what the node *is* with where it
*sits*. Any architecture that reduces nodes to opaque handles gives this up.

## 2. State of the evidence (agreed)

| family | verdict | reason on record |
|---|---|---|
| additive (Ψ into q/k/v at name tokens) | dead | channel real but ~10× too weak; self-limiting — amplitude corrupts the lexical identity the readout needs (e12 gate sweep, inverted-U) |
| attention mask (hard adjacency block + ΨΨᵀ log-gate) | partial | 0.48 → 0.86 ladder on old n30 (e13f); collapses with size (ai8c2bm0: 84% n30 → 41.7% n60_v3, same weights) |
| mask + post-fusion, RL | best graph-channel number | 42 → 60.7% n60 (McNemar 18/2), then saturates (ext300 plateau ~55%); mask-only RL inert |
| text edges | reference | 0.986 n30 / 0.952 n60 |

- 0.86 on old n30 is **not** parity with 0.95; the n30 gap is real and the n60 gap is large.
- The n60 GT model reaches only **0.63 on its own training graphs** — an in-sample
  fitting deficit, not (only) a generalization failure. Consequence: train-graph
  accuracy is a valid, faster signal for judging new candidates.
- Missing control: the text-edges model's *cross-size* transfer (n30-trained → n60)
  was never measured. Assumed robust; one eval job would settle it.

## 3. The per-hop finding (new, decisive)

Accuracy by gold shortest-path hops, e17 checkpoints (init → best RL):

| hops | n30 | n60 |
|---|---|---|
| 1 | 71% → 86% | 86% → 86% |
| 2 | 70% → 83% | 24% → 38% |
| 3 | 47% → 62% | 17% → 24% |
| 4 | 17% → 35% | 0% → 13% |
| ≥5 | 0% (6 tasks) | 0% (12 tasks) — partly a diameter artifact (n60_v3 diameters 4–5) |

Two things this rules in/out:

1. **Degradation with size at fixed path length.** Same hop count, larger graph, much
   lower accuracy. This is not path-reasoning difficulty; it is per-decision readout
   degrading with N.
2. **Decisions are not independent.** n60 1-hop at 86% would predict 2-hop ≈ 74% under
   independence; observed 24%. The first hop is *anchored* (current node is a prompt
   mention the model finds by name; goal named in the task). The second hop requires
   reading adjacency from a node reachable only through the channel — no text anchor.

Interpretation (agreed): the model has "blurry vision" — it cannot reliably see even
**one step ahead from the node it is standing on** (1-hop < 100% under the most
favourable conditions the architecture will ever have; text-edges ≈ 100%), and the
unanchored readout collapses with N.

## 4. Diagnosis (agreed)

- The topology is in the model **losslessly**: the mask is a hard −inf block on
  non-adjacent node-token pairs. The model still emits 2-hop-shortcut edges in ~67% of
  its failures and is sensitive to the mask (k_hops=2 collapses to 0.16). So the mask
  works as a **permission on what the LLM reads**, never as a **pointer to what it
  should write**.
- The failure is **node-identity binding**, not topology: the LLM cannot convert "the
  positions I was allowed to attend to" into "the name I should write," and that
  conversion degrades as the candidate set grows (mean degree ~4.4 → ~6 at n60).
- A second, related failure: **goal selection**. At the n60 init, 29/84 failures are
  wrong-endpoint (plus numbered-sibling confusion `sub_dock_1` ↔ `_2`); text-edges
  names the right endpoint 100% of the time. RL halved this (29 → 13) but left
  hallucinated edges at 19. Routing architecture alone does not address it.
- Every architecture to date modulates *where the LLM looks*; none gives the LLM a
  mechanism to *bind a node identity to an output token*.

## 5. Directions discussed

- **Pointer fusion (E)** — demoted to a **fallback**, not a starting point. Biasing
  the sampler toward what the GT believes overrides the LLM rather than teaching it;
  we would not learn whether the LLM integrated the graph.
- **Node identity slots (one soft `<node>` token per node next to its name)** —
  **rejected** as the centre: it fixes the anchoring problem but turns nodes into
  opaque handles, giving up the semantic⊕structural merge in §1.
- **Kept — binding auxiliary loss** during SFT: the hidden state at each name mention
  must predict its node's Ψ (or an equivalent identity target), so name↔node binding is
  supervised explicitly rather than hoped for from the LM loss.
- **Kept — neighbour-naming probe as the dev loop**: "you are at X, name its
  neighbours" on train graphs, starting from very small graphs and scaling N. It is the
  1-hop readout in isolation — the fastest discriminator for any candidate, and a
  simplified task to iterate on before full planning.

## 6. What the next architecture has to do

Keep semantic and structural information merged in one representation the LLM can
*write from*, and make the 1-hop anchored readout ~100% on small graphs before anything
else. Prediction to test against §3: if identity binding is the bottleneck, the 2-hop
cliff at n60 flattens first.

Next session: brainstorm architectures under these constraints.

## 7. Context loading guide for the next session

This session loaded all 14 docs + ~6k lines of code (~150k tokens) and most of it was
history already condensed above. For the architecture brainstorm, load in this order
and stop when the question is answered:

**Tier 1 — always (≈ 12k tokens)**
- this note
- `docs/2026-08-19 e17_architecture_implementations.md` — what exists today: the
  shared Ψ spine, candidates A/D/E/C as implemented, gradient-path table, plumbing
  touchpoints. The single best map of the current architecture surface.
- `src/prism/models/architectures.py` (218 lines) — the factory; every arch and its
  constructor knobs in one place.

**Tier 2 — when designing a concrete mechanism (≈ 25k tokens)**
- `src/prism/models/gnn_llm.py` lines ~750–1560 only: `LearnableGraphMaskLLM` —
  constructor, the four `enable_*` pathways and their hooks, `build_structural_mask`,
  `forward`. Skip lines 1–750 (attention patches, WIRE helpers) and 1850–2860 (WIRE
  class) unless the idea touches q/k directly.
- `src/prism/models/gnn_llm.py` lines ~3158–3360: `mask_node_values`,
  `_MaskDecodeRowState`, `BatchedMaskDecodeInjector` — only if the idea needs
  per-decode-step state.
- `src/prism/models/gt.py` lines 210–260 (`GraphTransformer` docstring) and 821–890
  (`build_psi_producer`) — the tower's interface; the rest is internals.
- `docs/2026-08-19 e17_analysis_n60.md` §4–§6 only — failure decomposition and
  per-path-length figures.

**Tier 3 — only for a specific question**
- `docs/2026-07-02 e12_leakfree_multistage.md` §4–§5 — the additive-family verdict and
  gate sweep; read before proposing anything that adds Ψ into the residual stream.
- `docs/2026-07-21 e13_nav_pe_setup.md` results tables — the 0.48 → 0.86 ladder and the
  k_hops=2 / layer-scope dead ends.
- `src/prism/training/trainers_rl.py` `MaskGRPOTrainer` (lines ~400–840) — only when
  wiring a candidate into RL.
- `src/prism/training/train_v3.py` — only for the `gnn_config` provenance contract
  (lines ~251–310) when adding config keys.

**Skip** (fully condensed here or ops-only): e10, e11 (both), e14 v2/v3, e16 note,
e17 post-fusion design, e18 spine draft, `train_rl.py`, `trainers.py`, `rewards.py`,
`r_pearl.py`, `gcn.py`, `vllm_graph/*`, all notebooks.
