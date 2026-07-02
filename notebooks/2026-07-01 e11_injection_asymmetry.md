---
tags: [experiment, e11, graph-injection, bug-analysis]
date: 2026-07-01
status: diagnostic partially complete; e11 retrains running
related: ["e10_relative_pe_notes", "e11_decode_time_injection_design"]
wandb: alelab/GREP-PRISM (tag e11_injection_scope)
---

# e11 — The train/inference injection asymmetry

> [!abstract] TL;DR
> Every no-edge-list architecture delivered the graph channel to positions that only
> exist under teacher forcing: node mentions **inside the assistant answer** (the
> labels). At generation, decode steps receive **no** injection and the map is
> prompt-only, so the learned mechanism has nothing to run on. A decision-token
> diagnostic confirms it causally for the mask family (+12 pts / −1.04 nats from the
> answer-side channel alone) and shows the additive family never engaged its channel
> at all (`pe_gain` gate never opened). This reframes *"the inductive bias doesn't
> help"* as *"the inductive bias was never present at the moment of prediction."*

## 1. Setup under test

- **Task:** SPINE/PRISM navigation planning; compact prompt lists node names + robot
  location; `data.text_edge_list=none` removes the edge bullets, so connectivity must
  reach the model through the graph channel. Answer = `<think>…</think>` +
  arrow-joined route; grading is a deterministic walk-validity check on the graph.
- **Data:** `nav100_n30_gemma_data` revised; 400 train / 100 val conversations;
  ~24 regions, ~53 edges (mean degree ≈ 4.4); held-out shortest paths median 2 hops.
- **Architectures:** additive `RoPE(x) + Ψ` into q/k/v (`rpearl_llm`, `gt_llm`,
  `rpearl_gt_llm`), structural attention masks (`graph_mask_llm`,
  `learnable_graph_mask` with $M = \alpha A + (1-\alpha)\,\mathrm{sim}(\Psi\Psi^\top)$),
  `postfusion` cross-attention.

## 2. The mistake, precisely

Let $\pi : \{1..T\} \to \{0, 1..N\}$ be the injection map (token → node, 0 = non-node),
and $B \in \mathbb{R}^{T\times T}$ the structural bias added to attention logits,
$B_{ts} = M_{\pi(t)\pi(s)}$ when $\pi(t), \pi(s) \neq 0$ (else 0), with
$M_{ij} = -\infty$ for non-adjacent $i,j$.

**Training** built $\pi$ over the FULL sequence — prompt ⊕ gold answer:

```python
# src/prism/data/data.py — SpineDataCollator._extract_graph (before e11)
scope_start = find_last_graph_scope(example["input_ids"], self.tokenizer)
injection_map = build_injection_map(
    example["input_ids"], node_token_seqs, scope_start=scope_start
)   # no upper bound: answer-side mentions are node-assigned
```

**Generation** built $\pi$ over the prompt only, and the attention patch skips decode
steps entirely:

```python
# src/prism/models/gnn_llm.py — _graph_mask_attention_forward
if bias is not None and q_len == bias.shape[-2] and k_len == bias.shape[-1]:
    ...  # prefill only: with KV cache q_len=1, shapes mismatch -> stock attention
```

So the attention matrix differs between the two regimes exactly on the answer block:

$$
\underbrace{B^{\text{train}}_{t\cdot},\; B^{\text{train}}_{\cdot t} \;=\; M_{\pi(t)\,\cdot}}_{t \in \text{answer mentions}}
\qquad\text{vs.}\qquad
B^{\text{gen}}_{t\cdot} = B^{\text{gen}}_{\cdot t} = 0 .
$$

Why this is label leakage: the token deciding plan step $k{+}1$ can, at training,
(a) read the step-$k$ mention a few tokens back, whose **key/value states were
computed under $M_{X_k\cdot}$** (its row encodes "my neighbors"), and (b) once the
target mention starts, its own **query row is $M_{X_{k+1}\cdot}$** — the ground-truth
node's identity, injected into the input. SGD takes the shortcut; the circuit it
builds reads inputs that generation never provides. Same argument for additive
$\Psi$: answer-side $q \mathrel{+}= W_q\Psi_{X}$ is the label in q-space (moot in
practice — see §4, the gate never opened).

> [!warning] Why no metric caught it
> `graph_acc/answer_nodes` grades every token inside answer name spans. Decision
> tokens (first token of a node's first answer mention) are ~4% of those positions;
> the rest are name-completions and repeat-mentions, copyable without any graph
> knowledge — the **no-graph baseline scores 0.979** on this metric. "PE Gain" is the
> raw gate scalar `tanh(pe_gain)`, not an ablation. `eval/loss` is teacher-forced
> *with* the leak. Only generation accuracy saw the failure — as a floor (~0.10–0.16).

## 3. The diagnostic

`scripts/diag_injection_ablation.py` (helpers `prism/eval/injection_diag.py`): one
teacher-forced forward per condition, same weights, same tokens, only $B$ varies:

| condition | injection map | equals |
|---|---|---|
| `train_style` | full sequence | what training optimized |
| `prompt_only` | clamped at `answer_start` | what generation conditions on (with a gold prefix — an upper bound) |
| `no_injection` | empty / gate zeroed | stock LLM |

Graded on **decision tokens** $D$ (first token of each node's first answer mention):

$$
\mathrm{NLL} = -\tfrac{1}{|D|}\textstyle\sum_{t\in D} \log p_\theta(x_t \mid x_{<t}, B),
\qquad
\mathrm{acc} = \tfrac{1}{|D|}\textstyle\sum_{t\in D} \mathbf 1[\arg\max = x_t].
$$

## 4. Results (100 val conversations, n=519 decision tokens)

| checkpoint (arch) | train_style | prompt_only | no_injection |
|---|---|---|---|
| `rv_gmask_noedges` (**mask, param-free**, 12B) | **0.705 / 0.92** | 0.595 / 1.93 | 0.584 / 1.96 |
| `rv_rpearl_noedges` (additive, 12B) | 0.574 / 1.72 | 0.578 / 1.72 | 0.578 / 1.72 |
| `rv_gt_l5d2048_noedges` (additive GT, 12B) | 0.578 / 1.70 | 0.574 / 1.70 | 0.574 / 1.70 |
| `e10_integ_rpe` (learnable mask + pretrained GT, 31B) | *pending* | *pending* | *pending* |
| `e9_ms_stage3` (additive, gate 0.27, 31B) | *pending* | *pending* | *pending* |
| `e9_llm_baseline_noedges` (floor, 31B) | — | — | *pending* |

Completion/repeat tokens: ≈0.99 in every condition, every model (the saturation).

> [!important] Reading
> - **Mask family — leak confirmed.** Answer-side masking triples the probability on
>   the correct next node ($e^{-0.92}\!\approx\!0.40$ vs $e^{-1.96}\!\approx\!0.14$);
>   in perplexity terms ~2.5 effective candidates (≈ knowing the neighbor set,
>   degree ≈ 4.4) vs ~7 (broad guess). Prompt-only — all generation gets — retains
>   ~1 pt. The model *learned to use adjacency*; it learned it on a channel that
>   doesn't exist at decode.
> - **Additive family — channel never engaged.** All conditions identical; causal
>   confirmation of the `pe_gain` cold-start (gates ended at ~0.001–0.005; only
>   stage-2 pretraining moved one to 0.27). Those "no-edge PE runs" were effectively
>   baseline runs.
> - The ~0.58 floor includes easy decisions (start node = `Robot location:` copy,
>   goal named in the task); the leak's effect concentrates on intermediate hops.

## 5. The fix(es)

**(a) Match training to generation** — `data.injection_scope=prompt_only` (default
`full_sequence` unchanged):

```python
# SpineDataCollator._extract_graph (e11)
if self.injection_scope == "prompt_only":
    injection_map = clamp_injection_map(injection_map, example["answer_start"])
```

Forces any useful circuit to route through prompt mentions (find current node's
prompt mention by name → read its mask-contextualized value / ψ → emit neighbor).
Running as `e11_injection_scope` (arms paired to `ror8gtet` and `tyvhwlmx`, only
this flag changed).

**(b) Match generation to training** — decode-time injection: assign node ids to
*completed* generated mentions and extend $B$ rows/columns per decode step. Needs
the post-completion consistency rule at training too (mid-mention query rows carry
the label — not reproducible at decode). Design: [[e11_decode_time_injection_design]].

**(c) If (a) trains to a null:** the bottleneck is learning the readout from 400
conversations — scale LLM-side alignment (leak-free edge-list reconstruction on
synthetic graphs, graded by *generation-time* reconstruction F1) before more
architecture work.

## 6. Falsifiable predictions

1. e11 arm (a) generation accuracy: any real gain over the 0.10–0.16 floor means the
   prompt-side readout is learnable; parity with floor + healthy training loss means
   sample-starved readout → go to (c).
2. `e10_integ_rpe` diagnostic (pending) shows the gmask-shaped leak signature.
3. `e9_ms_stage3` (gate 0.27) shows a small train_style > prompt_only gap — the only
   additive checkpoint where it's possible.

## 7. Provenance

- Diagnostic JSONs: `outputs/diag_ckpts/<run>/injection_diag.json` (local),
  `$ALELAB_DRIVE/GREP-PRISM/outputs/.../injection_diag.json` (betty).
- Code: commits `6452f93` (diagnostic + flag + tests `tests/test_injection_scope.py`),
  `c7b947d` (design note), branch `fable_experiments`.
- Historical anchors: refactor_verify no-edge gen acc 0.01–0.16 vs with-edges
  0.76–0.88; §5.1/§5.2 artifacts in [[e10_relative_pe_notes]] (same genus: train-time
  attention seeing what generation cannot).
