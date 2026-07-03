---
tags: [transfer, gegr, graph-injection, e11, e12]
date: 2026-07-02
source_project: GREP-PRISM (branch fable_experiments)
status: final — additive-PE line closed, mask line active
---

# GREP-PRISM → GEGR: findings transfer (injection-asymmetry investigation, e11–e12)

Written to be self-contained: everything needed to audit and fix a graph-augmented-LLM
pipeline of the GREP/GEGR family without access to GREP-PRISM history. Context: SPINE
navigation planning, Gemma-4 (12B/31B) + LoRA, graph channel delivered either
**additively** (ψ into q/k/v at node-mention token positions) or as a **structural
attention mask/bias** (adjacency-shaped logit bias). All findings below were
established causally (single-variable A/Bs or same-weights ablations), not by
correlation across runs.

## 0. TL;DR — what to do in GEGR

1. **Audit for the injection asymmetry (§1).** If training builds injection maps over
   the full sequence (prompt ⊕ gold answer) and generation doesn't inject at decode
   steps, your model is being trained on a channel that does not exist at inference.
   This single bug held every no-edge-text architecture at the no-information floor
   for months, and *no teacher-forced metric showed it* (§2).
2. **Port the two scope fixes (§3):** `injection_scope=prompt_only` for answer-side
   supervision, `exclude_supervised` for prompt-side supervision (reconstruction-style
   objectives). Principle: **injection positions ∩ loss-target positions = ∅.**
3. **Port the decision-token diagnostic (§4) before trusting any result.** It is the
   only instrument we had that measures the graph channel *at the moment of decision*,
   causally, for the cost of a few hundred forwards.
4. **Prefer mask/logit-bias interfaces over additive residual-stream injection (§6).**
   Under identical leak-free protocols: parameter-free adjacency mask 0.39, learnable
   mask + pretrained PE 0.48, fully pretrained additive PE **0.09 = floor**. The
   additive interface is self-limiting (measured, §5.4): amplitude corrupts the very
   token identities the readout needs, with the useful ceiling at ~0.25× embedding
   scale delivering ~6× too little information per decision.

## 1. The bug class: train/inference injection asymmetry

Let π map token positions to node ids (0 = non-node) and let the graph channel act at
positions where π ≠ 0 (additive: `h_t += tanh(g)·ψ_{π(t)}`; mask: attention bias
`B_ts = M_{π(t)π(s)}`).

**Leak A — answer-side (any `loss_target` on responses).** Training builds π over the
full teacher-forced sequence, so node mentions *inside the gold answer* — the labels —
carry their node's channel. The token deciding plan-step k+1 can read (a) the step-k
mention a few tokens back whose k/v were computed under that node's mask row /
ψ ("my neighbors"), and (b) once the target mention starts, its own query row is the
*ground-truth node's identity injected into the input*. At generation, maps are
prompt-only and the attention patch no-ops on cached decode steps (shape guard:
`q_len == bias.shape[-2]` fails when q_len=1), so the learned circuit's inputs don't
exist. SGD prefers this shortcut and it **actively suppresses** the honest circuit:
removing it didn't just stop hurting — accuracy *tripled* (§5.1) and hallucination
collapsed (0.65→0.22, 0.63→0.17).

**Leak B — supervised-prompt-side (reconstruction pretraining).** If the loss target
lives in the *prompt* (we supervised the edge-list bullets: `loss_target=edge_list`),
clamping maps at `answer_start` does nothing — predicting the target node in
`• region_5 <=> region_7`, the positions inside `region_7`'s own span carry
ψ_{region_7}: the label in q-space. The subtlety: the ψ *value* exists at generation
too; what's illegitimate is the *pairing* — the map assigns ψ_{region_7} to that span
only because the teacher-forced text already says region_7 there. Free generation
could only make that assignment after the mention completes, when the prediction is
over.

**Decode fall-through (check your model code).** Both our attention patches silently
fell back to stock attention whenever shapes didn't match the prefill case. That is
*correct* behavior given prompt-only maps, but it means: (i) generated tokens never
receive injection, and (ii) nothing crashes to tell you training assumed otherwise.
Grep for the shape guards; treat every silent fallback as a place train/inference can
diverge without error.

## 2. Why no metric caught it — audit yours

- **Token-level "graph accuracy" saturates on copyable tokens.** Our
  `graph_acc/answer_nodes` graded every token inside answer node-name spans; ~96% of
  those are name-completions and repeat-mentions predictable with zero graph
  knowledge. The **no-graph baseline scored 0.979** on it. Only the *first token of a
  node's first answer mention* (a "decision token", ~4% of node positions) requires
  adjacency.
- **A gate scalar is not an ablation.** We logged `tanh(pe_gain)` as "PE Gain" and
  read increases as the channel engaging. It measures amplitude, not usefulness.
- **Teacher-forced eval loss inherits the leak.** It conditions on the same
  full-sequence maps as training; it improves as the shortcut is learned.
- Only *free-generation* task accuracy saw the failure — as an uninformative floor.

## 3. Validated fixes to port

Collator-level scope control on the injection map (opt-in; default preserves history):

```python
# scope resolution inside the collator's per-example graph extraction
if self.injection_scope == "prompt_only":          # answer-side supervision
    injection_map = clamp_injection_map(injection_map, example["answer_start"])
elif self.injection_scope == "exclude_supervised": # prompt-side supervision
    injection_map = exclude_positions_from_injection_map(
        injection_map, set(example[self.supervised_positions_key]))
    # supervised_positions_key = the loss-target index column, e.g. "edge_list_idx";
    # set by the trainer from loss_target. Reject loss_target='all' (would empty map).
```

The two map utilities (self-contained, port verbatim; spans are `(start, end)`
half-open, maps are `{node_id: [spans]}`):

```python
def clamp_injection_map(injection_map, scope_end):
    """Drop spans starting at/after scope_end; truncate straddlers; drop empty nodes."""
    clamped = {}
    for nid, spans in injection_map.items():
        kept = [(s, min(e, scope_end)) for s, e in spans if s < scope_end]
        if kept:
            clamped[nid] = sorted(kept)
    return clamped

def exclude_positions_from_injection_map(injection_map, excluded):
    """Remove a position set from every span, splitting spans into maximal
    remaining sub-spans; drop empty nodes. Enforces injection ∩ loss-target = ∅."""
    if not excluded:
        return {nid: sorted(spans) for nid, spans in injection_map.items()}
    result = {}
    for nid, spans in injection_map.items():
        kept = []
        for start, end in spans:
            run = None
            for pos in range(start, end):
                if pos in excluded:
                    if run is not None:
                        kept.append((run, pos)); run = None
                elif run is None:
                    run = pos
            if run is not None:
                kept.append((run, end))
        if kept:
            result[nid] = sorted(kept)
    return result
```

Wiring notes (mirror `GREP-PRISM/src/prism/training/train_v3.py:74-100`):
validate the scope value at startup, print it (`[data] injection_scope=...` — grep the
job log to confirm the path was live), and record it in the checkpoint's model config
so diagnostics can recover it. Ensure every node keeps ≥1 prompt-side span after
exclusion (our prompt format lists nodes before edge bullets, so this held; assert it —
some feature modes, e.g. word-embedding node features, hard-require full coverage).
Tests to adapt: `GREP-PRISM/tests/test_injection_scope.py` (18 cases: clamp boundary
semantics, span-splitting, disjointness invariant `covered == original − excluded`,
collator wiring with a fake tokenizer).

## 4. The measurement instrument: decision-token diagnostic

Port `GREP-PRISM/scripts/diag_injection_ablation.py` + `src/prism/eval/injection_diag.py`.
Design: teacher-forced forwards, same weights, same tokens, seed reset before *every*
forward (identical stochastic PE probes across conditions) — only the injection map
varies. Conditions: `train_style` (full map), `prompt_only` (clamped), `no_injection`
(empty map for masks; gate temporarily zeroed for additive — an empty map can break
feature prep). Grade only **decision tokens** (first token of each node's first
answer mention); report acc and NLL (e^NLL ≈ effective candidate count; compare to
mean node degree).

Reading rules:
- `train_style ≫ prompt_only ≈ no_injection` → leak-trained channel; explains
  generation collapse.
- all three equal → channel never engaged (check the gate, §6.2).
- `prompt_only ≫ no_injection` → generation-compatible readout exists; the bottleneck
  is elsewhere (search/decoding).
- **Caveat:** the diagnostic measures the *current* circuit, not the reachable one.
  Our param-free mask showed a near-zero prompt-side channel on the leaky checkpoint
  (−0.03 nats) yet retrained to 0.39 under `prompt_only` — a small measured channel
  does not bound what leak-free retraining can build.

Gate-sweep mode (`--gate-sweep "0,0.1,0.25,0.5,0.75,1.0"`, additive archs): reruns
`prompt_only` with the gate pinned to each effective value (pe_gain := atanh(g)) plus
the trained value. Answers "is the trained gate amplitude-limiting?" in ~10 min of
forwards instead of a training run.

## 5. Evidence (all Gemma-4-31B unless noted; full 10-graph held-out generation eval)

### 5.1 Leak removal alone (e11) — single-flag A/B vs paired historical runs
| arch (interface) | full_sequence | **prompt_only** | halluc. rate |
|---|---|---|---|
| adjacency mask, **parameter-free** | 0.13 | **0.39** | 0.65 → 0.22 |
| learnable mask + pretrained PE | 0.14 | **0.48** | 0.63 → 0.17 |

### 5.2 Leak-free additive multistage (e12) — every excuse eliminated, floor persists
Chain: SFT → PE-only edge-list reconstruction (`exclude_supervised`) → joint
no-edge-text (`prompt_only`). Only the scope flags changed vs the paired leaky chain.
- Stage-2 reconstruction **learns without the shortcut**: eval loss 0.534→0.3885 over
  4 epochs (leaky: 0.4375→0.3525), descent attributable purely to the PE pathway
  (LLM frozen, gate cold-started); gate opens to 0.23.
- Stage-3 no-edge generation: **0.09** (leaky paired run: 0.10; floor ≈ 0.10–0.16).
- With-edges parity under prompt_only: **0.99** (historical band 0.87–0.93) — the fix
  costs nothing in the easy regime.
- Diagnostic: generation-compatible channel −0.34 nats (leaky chain: −0.16; mask
  family: −0.92), decision accuracy flat — mass moves, argmax never flips. ~2 nats
  per decision are needed.

### 5.3 The additive cold-start saddle (why from-scratch gates never open)
Gate g = tanh(pe_gain), init 0. The gate's own gradient is open (tanh′(0)=1) but is an
inner product with an *uninformative, per-forward-resampled* ψ → zero-mean noise; the
PE's gradient carries the tanh(g)≈0 factor → coupled saddle. From-scratch runs ended
with |gate| ≈ 1e-3 (Adam random-walk residue ~lr·√T). Escape requires pretraining the
PE with the LLM frozen at high lr (or warm-starting). Multistage is a *valid
instrument* — but its stage objective must be leak-free (§1 Leak B) and, ideally,
graded at generation: teacher-forced reconstruction NLL demonstrably does not transfer
to free-generation use (§5.2).

### 5.4 The additive interface is self-limiting (gate sweep, e12 checkpoint)
Decision NLL vs pinned effective gate: 2.41 (g=0) → **2.07 (g≈0.23–0.25, minimum;
= trained value)** → 2.30 (0.5) → 3.61 (0.75) → 4.92 (1.0 — equal to the no-graph
baseline 4.91). At high amplitude, name-copying itself degrades (repeat-mention NLL
×34). Mechanism: ψ shares the residual stream with text; amplitude buys signal only
until it erases the lexical identity the readout needs to bind mentions by name. The
optimizer had parked the gate at the measured optimum — the ceiling is the interface,
not the optimization. Mask/logit-bias interfaces pay no such tax (orthogonal to the
content pathway).

## 6. Architecture guidance for GEGR

1. **Interface ranking (empirical):** structural attention bias/mask ≫ additive
   residual injection. If GEGR's tasks (reachability, positionality, navigability)
   need adjacency at decision time, deliver it as attention structure. The learnable
   variant that won here: `M = α·A + (1−α)·sim(ΨΨᵀ)` with hard-blocked non-edges,
   α=0.7, applied on dense/global attention layers, ψ from a pretrained probe-based
   graph transformer.
2. **If you keep an additive channel:** expect the gate to equilibrate at the
   interference optimum (~0.2–0.3 of embedding scale here); do not chase amplitude.
   Run the gate sweep (§4) before concluding anything from the gate's value.
3. **Pretraining objectives:** grade them at *generation* (free-run reconstruction
   F1), not teacher-forced NLL; and apply `exclude_supervised` if the target is in
   the prompt.
4. **Open directions, in expected-value order** (designs, not results):
   - **Decode-time injection** (mask family): assign node ids to *completed*
     generated mentions and extend bias rows per decode step; requires the matching
     train-time rule (mentions act as keys with node id, as queries only from the
     span-final position — mid-mention query assignment is the label, Leak A again).
     Existence proof of headroom: the leak-trained circuit under train_style scores
     0.865 acc / 0.52 nats on decisions — with-edges-level performance.
     Design note: `GREP-PRISM/notebooks/e11_decode_time_injection_design.md`.
   - **Interleaved graph tokens** (additive revival, untested): give ψ its own
     *inserted* token positions adjacent to each mention instead of superposing onto
     name tokens — removes the §5.4 interference tax by construction; binding by
     adjacency. Cost: token-insertion plumbing (maps, labels, positions).
5. **Reporting hygiene:** never report token accuracy over node spans without the
   decision/completion/repeat split; never report a gate value as evidence of use;
   pair every architecture change with a single-flag A/B against a named historical
   run.

## 7. Ten-minute audit checklist for the GEGR codebase

- [ ] Where are injection maps built for **training**? Full sequence? → Leak A.
- [ ] Where for **generation**? Prompt-only / no decode injection? → asymmetry exists.
- [ ] Any prompt-side supervised objective (reconstruction, infilling)? → Leak B;
      needs `exclude_supervised`, `prompt_only` is not sufficient.
- [ ] Grep attention patches for shape-guard fallbacks (`q_len == …`); list every
      silent no-op path and decide, per path, what training should assume.
- [ ] Does eval include *free-generation* task accuracy (not only teacher-forced)?
- [ ] Are token-accuracy metrics split by decision/completion/repeat?
- [ ] Is the collator's scope printed in logs and recorded in checkpoints?
- [ ] Run the decision-token diagnostic on the current best no-text-fallback
      checkpoint before building anything new.

## 8. Provenance (GREP-PRISM, branch `fable_experiments`)

- Code: `6452f93` (diagnostic + `prompt_only` + tests), `732d480`
  (`exclude_supervised` + e12 chain), `c589b1e` (gate sweep).
- Lab notes: `notebooks/2026-07-01 e11_injection_asymmetry.md`,
  `notebooks/2026-07-02 e12_leakfree_multistage.md`.
- W&B (alelab/GREP-PRISM): e11 arms `bznw3x9p`/`22lq43i6` vs paired
  `ror8gtet`/`tyvhwlmx`; e12 chain `0dq55cex` → `wh0537au`/`gvzylvay` vs paired
  `3bitwckz` → `kfuu2djo`/`6lefhd76`; diagnostics JSONs in each run dir
  (`injection_diag*.json`).
- Caveats attached to all of the above: single seed per cell; generation metric
  0.1-grained per graph; 31B; 400-conversation corpus. The cross-family separations
  (0.09 vs 0.39–0.48) and the gate-sweep shape are far outside these grains; the
  finer deltas (parity 0.99 vs 0.93, channel −0.34 vs −0.16) are not.
