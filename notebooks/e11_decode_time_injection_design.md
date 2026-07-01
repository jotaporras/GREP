# e11 — Decode-time graph injection (design proposal, NOT implemented)

Status: **proposal for review**. This is core-method territory; nothing here is wired
in. Companion code that IS implemented and runnable: the injection-ablation
diagnostic (`scripts/diag_injection_ablation.py`, helpers in
`src/prism/eval/injection_diag.py`) and the train-side alignment flag
(`data.injection_scope=prompt_only`, clamped in `SpineDataCollator._extract_graph`).

## 1. Problem statement

All current architectures deliver the graph channel asymmetrically:

- **Training (teacher-forced):** injection maps cover the full sequence
  (`SpineDataCollator._extract_graph`), so node mentions inside the assistant answer
  — the labels — carry Ψ (additive) or an adjacency-masked attention row (mask
  archs). The token that decides the next plan step can anchor on the *previous*
  step's mention, whose k/v carry the channel.
- **Generation:** `inference.py` builds prompt-only maps and both attention patches
  no-op on cached decode steps (`gnn_llm.py:30`, `gnn_llm.py:70`). Generated tokens
  — including the model's own previous plan steps — never carry the channel.

`injection_scope=prompt_only` fixes this by *removing* the answer-side channel from
training. Decode-time injection is the complementary fix: *adding* the channel to
generation, so the inductive bias ("attention masked/biased by adjacency at the
moment of next-step prediction") actually exists where the prediction is made. The
two compose: decode-time injection defines what generation can see; training must
then expose exactly the same thing (§4).

## 2. Mechanism

During `generate()`, maintain a growing token→node assignment over the *generated*
suffix, updated once per decode step:

1. **Mention tracking.** After each generated token, run an incremental matcher over
   the generated suffix against `node_token_variants` (same longest-first,
   disjoint-span semantics as `build_injection_map`). A node id is assigned to a
   span only once the span is COMPLETE and unambiguous (no other node's token
   sequence is a strict extension of what has been generated so far).
   Precedent: `InjectedCompositeGraphLLM.decode_setup / decode_extend`
   (`composite_graph_llm.py`), driven by a forward pre-hook in
   `inference.py:239-244`.
2. **Channel extension, mask family.** Keep the prefill `[1,1,S0,S0]` bias; at each
   decode step build the single query row `[1,1,1,K]` for the new token: if the
   token belongs to a completed mention of node `i`, the row is `M[i, π(s)]` over
   all key positions `s` (prompt AND generated) currently assigned to nodes,
   `-inf`-blocking non-adjacent node keys as in `build_structural_mask`; otherwise
   the row is 0. The attention patch's current shape-mismatch fall-through
   (`q_len == bias.shape[-2]`) becomes a lookup of this per-step row.
   Subtlety: the just-generated token enters the KV cache with the k/v computed on
   THIS step, so its own key-side assignment must be decided *before* its forward —
   i.e. assignment happens when the token that completes a mention is *consumed* as
   input, which is exactly the `decode_extend` pre-hook timing.
3. **Channel extension, additive family.** Same tracking; for a query token assigned
   to node `i`, arm a per-step `_pe_signal` of shape `[1,1,H]` = placed Ψ_i (gated,
   normed as in `build_pe_signal`); its k/v then carry `W_k Ψ_i` / `W_v Ψ_i` into
   the cache, restoring the "previous plan step anchors ψ-space" circuit at eval.

## 3. The partial-mention consistency rule

A mention's node id is only knowable once the mention is complete. Teacher-forced
training with full-sequence maps violates this: query positions *inside* an answer
mention already carry the ground-truth node id (that is the name-completion leak —
with up to 3 nodes sharing a type prefix, the discriminating `_N` suffix is exactly
where it cheats). Rule for train/decode consistency:

> Answer-side tokens act as **keys** with their node id (later queries may read
> them), but as **queries** they are node-assigned only from the first position
> AFTER their mention completes.

- Mask family: expressible directly — the bias is per (query, key) pair; zero the
  in-span query ROWS of answer mentions, keep their key COLUMNS.
- Additive family: needs separate q- vs k/v-injection masks in
  `_prism_pe_attention_forward` (inject Ψ into k/v at all span positions; into q
  only where the assignment is decode-knowable). Small patch, but it touches the
  method's core equation — author's call.

## 4. Implementation order (each step testable)

1. `injection_scope=prompt_only` training arms (running: `e11_injection_scope`
   sbatch) — no decode-time work needed; generation already matches.
2. Mask-family decode extension (§2.2) + training with the §3 rule
   (`injection_scope=decode_consistent`, a third mode). Parity test: teacher-forced
   forward of a full sequence under the §3 rule must equal the step-by-step decode
   biases token-for-token (extend `tests/test_graph_mask.py` style).
3. Additive family (§2.3) with the q/kv split.

## 5. Risks / open questions

- **Readout still has to be learned.** Decode-time injection restores the channel;
  whether LoRA + 400 conversations can learn to use it is exactly what the
  diagnostic + e11 arms measure. If decision-token accuracy is flat even
  train-style (leak included), the bottleneck is the objective, not the asymmetry —
  scale the LLM-side alignment pretraining (leak-free stage-2 edge-list
  reconstruction on synthetic graphs, graded by generation-time reconstruction F1)
  before building this.
- **Greedy commitment.** The mask constrains a mention only after it completes; the
  first token of a wrong node is not prevented, only its consequences. A
  reconstruct-then-plan think-block (already scaffolded by "Key connections:" in
  the targets) may matter more than hard masking at decode.
- **Latency.** Per-step matcher is O(nodes × max-name-len) on the suffix — noise
  compared to a 31B forward.
- **Batch decode** is out of scope (eval generates one sample at a time).
