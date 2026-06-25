---
name: verifier
description: Use this when the user wants PyTorch unit tests written for specific code to prove it behaves as intended and to surface implementation, interface, or runtime breaks. Trigger on "/verifier", "verify this code", "write tests for", "unit test this", "test this module/function/class", "check this for breaks", "does this actually work", or when the user points at code (a file, function, class, or diff) and wants its behavior guaranteed by tests. Scope is strictly the referenced code plus its dependency closure — never the rest of the codebase. Takes an optional leading `DEEP-LEARNING` flag ("/verifier DEEP-LEARNING [prompt]") that unlocks testing of ML model inputs, outputs, and internal architectural flows; WITHOUT the flag the skill may only test deterministic computer-science code (datasets, evaluation, training orchestration) and is forbidden from exercising model architecture, model inputs/outputs, or any ML-based stochastic behavior.
---

# PyTorch Code Verifier

You design and implement PyTorch unit tests for **exactly the code the user referenced** (and the
dependencies that code actually calls), then run them to surface **implementation, interface, and
runtime** breaks. The deliverable is a runnable `test_*.py` plus a triage of every break found.

You are the project's fresh-pass verification oracle (see `AGENTS.md` → *Verification*). The whole
point is to catch errors the author's own reasoning would miss — so the tests must encode the
*contract*, derived independently from docstrings/spec/types, **not** a paraphrase of the
implementation. A test that re-implements the function and asserts they match proves nothing.

## Mode gate — parse this BEFORE anything else

The skill runs in one of two modes, selected by a single leading argument:

- **`/verifier DEEP-LEARNING [prompt]`** → **DL mode.** Full machine-learning verification is
  unlocked: you may test model architecture, layer/module internals, tensor inputs and outputs,
  forward/backward flows, gradient behavior, attention/masking, generation, parity/ablation against a
  reference module — the entire battery described below.
- **`/verifier [prompt]`** (flag absent) → **CS-only mode.** This is the default. **You are NOT
  ALLOWED to test any machine-learning model's inputs, outputs, or internal architectural flows.**

Parse the first whitespace-delimited token of the argument string. If it is exactly `DEEP-LEARNING`
(case-insensitive), strip it and run in DL mode; the remainder is the prompt. Otherwise run in CS-only
mode with the full argument as the prompt. **State which mode you are in as the first line of your
response**, so the user can see the gate was applied.

### CS-only mode — what you MAY and MUST NOT test

In CS-only mode the target is treated as ordinary deterministic computer-science code. You verify the
*plumbing*, never the *learning*.

**MAY test (deterministic CS surface):**
- Dataset / data-pipeline logic: parsing, tokenization bookkeeping, collation, batching/padding
  shapes as a pure function of inputs, masking/label construction, shuffling determinism under a
  fixed seed, train/val splits, indexing, caching, serialization round-trips.
- Evaluation logic: metric arithmetic, regex/answer matching, path/graph verification, accuracy
  aggregation, score reduction, log parsing — the deterministic scoring code, not the model that
  produced the predictions.
- Training **orchestration** (not the model): config/argument plumbing, LR-schedule math, optimizer
  parameter-grouping *structure*, checkpoint save/load round-trips, step/epoch counters, seeding,
  gradient-accumulation bookkeeping, callback/loop control flow — exercised with **stub tensors or
  fixed numeric inputs**, asserting the bookkeeping, not any learned quantity. This covers only the
  *mechanical* orchestration: that the plumbing moves the configured values where they belong. Any test
  that requires **interpreting the quality of the hyperparameters** — whether an LR/schedule/batch
  size/weight-decay/warmup choice is *good, stable, or appropriate*, whether a config will *train
  well* — is ML judgment and is **deferred to DL mode**; in CS-only mode assert that the value is
  plumbed correctly, never that it is a *sensible* value.
- Pure graph / tensor-shaped utilities that are deterministic functions of their inputs (adjacency
  construction, edge-list manipulation, coordinate math), **as long as** they are not part of a
  model's architectural forward path and you assert exact/closed-form values, not learned behavior.

**MUST NOT test (ML surface — DL mode only):**
- Any `nn.Module` / model **forward or backward** pass, layer internals, attention, masking-as-bias,
  positional encodings, or architectural data flow.
- Model **inputs or outputs** as such — logits, hidden states, embeddings, generated tokens,
  parity/ablation of a model against a reference, gradient flow through learned parameters. (You may
  still *call* the model as an opaque function to verify output **handling** — see the worst-case
  carve-out below — but never assert on the output's correctness.)
- Any property that depends on **learned weights** or on **ML stochasticity** (dropout, sampling,
  weight init variance, training convergence, loss going down).
- Any **interpretation of hyperparameter quality** — judging whether an LR, schedule, batch size,
  warmup, weight decay, or any training config is *good / stable / appropriate / likely to train well*.
  CS-only mode may verify such a value is *plumbed* correctly; deciding whether it is a *sensible*
  value is DL-mode judgment.

**Stochasticity rule (CS-only mode):** randomness is permitted **only** as deterministic empirical
calculation — e.g. seed a RNG and assert an exact shuffle order, or compute a closed-form statistic
over fixed data. No ML-based stochastic testing: no asserting over sampled model outputs, no
distributional/“runs N times and checks the mean” probes of a model, no init-variance checks.

**Worst-case carve-out — the model as a black-box function (CS-only mode):** you are *always* allowed
to **call** a machine-learning model as an opaque function/method — exactly as the CS infrastructure
itself calls it — for the **sole** purpose of checking that its output is **handled** correctly by the
surrounding harness (parsed, reshaped, indexed, written, dispatched, scored, serialized). This is
permitted even in CS-only mode because here the model is just another callable in the code path, not
the thing under test. Strict conditions:

- **Deterministic inputs only.** Feed the model the inputs already provided / fixed by the code or the
  prompt; change nothing about them. No varying, sampling, perturbing, or crafting new inputs to probe
  the model. Pin every decode/seed knob the harness exposes so the call is reproducible.
- **No accuracy interpretation.** You may NOT assert on, judge, or comment about whether the model's
  output is *correct, accurate, good, or sensible*. Treat the returned value as opaque bytes. The only
  legitimate assertions are about the **handling**: type/shape/dtype/keys of what comes back, that the
  harness consumes it without error, that downstream CS code routes/transforms/stores it as specified.
- **Harness under test, not the model.** A test in this carve-out fails when the *output-handling code*
  mishandles a real model return (wrong shape assumption, dropped field, bad parse, crash) — never
  because the model "got it wrong". If the model is expensive/external, prefer the smallest real call
  or an inline stub returning a representative deterministic value; use a real call only to confirm the
  return *shape/type contract* the harness depends on.

If the named target is *itself* a model/architecture and the flag is absent, **do not test it.** Say
so plainly: report that the target is ML-surface, that DL mode was not requested, and that the user
must re-invoke with `DEEP-LEARNING` to verify it. Then, if any genuinely deterministic CS helpers sit
in scope (e.g. a shape utility the model calls), offer to verify *only those*. Never quietly fall back
to architectural testing.

Everything below this section is written for the full battery; in CS-only mode, apply only the parts
that fall within the MAY list above and skip every model/architecture/IO item.

## Scope discipline — read this first

The single hard rule: **test only the referenced code and the code it depends on. Do not wander.**

- **In scope:** the named target (file / function / class / method / diff), plus the *dependency
  closure* — the project functions and classes the target actually invokes, transitively, until you
  hit a boundary.
- **Boundaries (stop here):** third-party libraries (`torch`, `transformers`, `torch_geometric`, …),
  unrelated sibling code the target does not call, and heavy/external effects (LLM API calls,
  network, GPU-only kernels, multi-GB data loads). You exercise the boundary, you do not test *into*
  it.
- **Do not** add coverage for, refactor, or "improve" code outside the closure. If a real bug clearly
  lives just outside scope, note it in the report; don't chase it with tests.

When unsure whether something is in the closure: if the target's correctness depends on it producing
a specific result, it's in scope — exercise it for real. If it's just a passthrough or an external
service, stub it with the smallest possible fake (mirror `tests/test_sim.py`'s inline `_DummyClient`).

## Procedure

### 1. Resolve the target
Pin down precisely what to verify. Read the user's prompt for a path, symbol, or diff. If they say
"this code" without a pointer, ask once for the file + symbol. Restate the target list before
proceeding (e.g. `prism.models.gnn_llm.GraphMaskLLM.build_structural_mask` + 2 helpers it calls).

### 2. Map the dependency closure
`Read` the target. List every project symbol it calls and every input it consumes. Follow those into
their definitions (Read/Grep), one hop at a time, stopping at the boundaries above. Produce a short
closure map: `target → {in-scope deps to exercise} | {boundaries to stub or feed minimally}`. This
map *is* your scope; do not test anything not on it.

### 3. Characterize intended behavior (the oracle)
For each in-scope symbol, write down the contract **from its spec, not its body**:
- **Signature & interface:** args (required/optional, types), return type/shape/keys, raised
  exceptions, mutation/side-effects, train vs eval mode expectations.
- **Tensor contract:** output **shape**, **dtype**, **device** as a function of inputs; broadcasting
  rules; batch handling.
- **Invariants:** the properties the docstring/PR claims (symmetry, conservation, parity, masking
  rules, monotonicity, idempotence). These become the strongest assertions.
- **Domain:** valid input ranges and what *should* happen on invalid input (per `AGENTS.md`, the
  project wants loud failure, not silent fallback — so "rejects bad input with a clear error" is a
  legitimate, testable contract).

If the contract is genuinely ambiguous, state your assumption in the test docstring and flag it in
the report rather than guessing silently.

### 4. Design the test matrix
Cover the three break classes the user cares about. Aim for the smallest set of tests that pins the
contract; every test must be able to *fail* for a real reason.

> **CS-only mode:** restrict the matrix to deterministic CS behavior (the MAY list in **Mode gate**).
> Drop every forward/backward, parity/ablation, gradient-flow, generation, and finite-output-of-a-model
> item below — those are DL-mode only. Keep the shape/value/interface/edge/raises items *only* where the
> symbol under test is deterministic plumbing rather than model architecture or model IO.

**Implementation breaks** — the logic is wrong even though it runs:
- Invariants from step 3 (the headline assertions).
- Numerical parity / ablation identity (e.g. `Ψ=0 ⇒ logits identical to stock`; `use_edges=False ⇒
  identity adjacency`). Compare against an *independent* reference (stock module, hand-computed small
  case, closed-form), never against the code under test.
- Known small cases computed by hand (a 2–3 node graph, a length-4 sequence) where you can assert
  exact values.
- Determinism: same seed ⇒ same output.

**Interface breaks** — the contract with callers/dependencies is violated:
- Output shape/dtype/device match the declared contract for representative + boundary inputs.
- Return structure (tuple arity, dict keys) is what callers in the closure assume.
- Required args enforced; optional args/defaults behave; keyword paths work.
- The target composes with its in-scope deps as wired (call it the way the real caller does).

**Runtime breaks** — it crashes or silently corrupts on inputs it should handle:
- Forward (and `backward()` where params exist) runs without error and produces **finite** output
  (`torch.isfinite(...).all()`).
- Gradient flow: `loss.backward()` populates grads on parameters that should learn; grads are finite
  and not all-zero where signal is expected.
- Edge inputs in the *documented* domain: batch=1 and batch>1, empty graph / single node / no edges,
  fully-masked-out rows (softmax safety), longest realistic sequence, repeated entities.
- `generate()` / decode-time path if the module is used autoregressively (KV-cache step, query
  len 1).
- Invalid input fails **loud** (assert it raises), matching the project's anti-silent-error stance.

Scale the matrix to the target: a pure helper needs a handful of value/shape tests; an `nn.Module`
warrants the full forward/backward/parity/edge battery.

### 5. Implement the tests — match repo conventions exactly
Write to `tests/test_<target>.py` (repo convention; these are durable behavior guarantees). For a
throwaway check the user doesn't want committed, use the scratchpad dir instead and say so.

Conventions, lifted from the existing suite (`tests/test_graph_mask.py`,
`tests/test_pe_injection_parity.py`, `tests/test_sim.py`):

- Start with `import sys; sys.path.insert(0, "src")` so `prism.*` imports resolve.
- **Tiny, random-init, CPU.** No GPU, no real checkpoints, no network. The project's base-LLM is
  **Gemma 4 12B** (`google/gemma-4-12B-it`), so build the fixture from a tiny random-init
  **`Gemma4UnifiedTextConfig`** mirroring that architecture — never load the real 12B weights. Shrink
  every dimension to CPU scale while keeping the Gemma 4 architecture/config class. Because
  `Gemma4Unified*` is post-cutoff and missing on older `transformers`, wrap the import in the `_skip`
  optional-dep pattern (exactly as `tests/test_pe_injection_parity.py` does):
  ```python
  try:
      from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig
  except Exception as e:  # noqa: BLE001 — any import failure ⇒ unsupported here
      return _skip(f"gemma4_unified unavailable: {e}")
  torch.manual_seed(0)
  cfg = Gemma4UnifiedTextConfig(
      vocab_size=64, hidden_size=32, intermediate_size=64,
      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
      head_dim=8, max_position_embeddings=64, attn_implementation="eager")
  llm = Gemma4UnifiedForCausalLM(cfg)   # random init, CPU — stands in for Gemma 4 12B
  ```
  Do **not** substitute `LlamaConfig` or the Gemma 1 `GemmaConfig`/`GemmaForCausalLM` classes — the
  base-LLM under test is Gemma 4 12B, whose architecture (q/k-norm, single-tensor RoPE, sliding
  windows, KV-shared layers) only `Gemma4Unified*` reproduces.
- `torch.manual_seed(0)` (or a fixed seed) anywhere randomness enters — determinism is mandatory.
- Plain `def test_*()` functions with `assert`; no test classes. A module-level docstring states the
  invariant under test; per-test docstrings state the specific case.
- Add the runner footer so the file works standalone *and* under pytest:
  ```python
  if __name__ == "__main__":
      for name, fn in list(globals().items()):
          if name.startswith("test_") and callable(fn):
              fn(); print(f"{name}: PASS")
      print("done")
  ```
- For genuinely optional deps (a model class missing on older `transformers`), use the `_skip` pattern
  from `test_pe_injection_parity.py` — skip under pytest, print `[SKIP]` as a script. Do **not** skip
  to dodge a failure.

Test-code style rules (consistent with `AGENTS.md`):
- The oracle is independent of the implementation. Hand-compute, use a stock reference module, or
  assert structural invariants — never copy the function's body into the test.
- `try/except` is allowed **only** to assert that something raises, or for the optional-dep skip.
  Otherwise write straight-line, fail-loud test code (no defensive guards, no swallowed errors).
- Keep fixtures small and inline (helper factories like `_tiny_llm`, `_graph`), mirroring the suite.

### 6. Run and triage
Run with the project env. Enter it with **`conda activate`** (not `conda run` — only `activate`
exports the env state `uv run --active` keys off), then invoke through **`uv run --active`** so it
uses the activated conda env instead of an ephemeral venv. **Always try `pytest` first:**
```bash
conda activate GREP-PRISM && uv run --active -m pytest tests/test_<target>.py -v
```
If that fails because `pytest` isn't installed in the env, fall back to running the file as a script
via its runner footer: `uv run --active python tests/test_<target>.py`.
If `conda` isn't on PATH in this shell, drop the `conda activate` prefix but keep `uv run --active`
(and note the env caveat). For each failure, determine whether it's a **test bug** (fix the test and
rerun) or a **real break** in the target (keep the test red, record it). Never weaken an assertion to
make a real failure pass — a red test that pins a true bug is the deliverable working.

### 7. Report
See **Output format**. Lead with the breaks found; the user's attention is the scarce resource.

## Error-class definitions (use these labels in the report)

| Class | Means | Example |
|-------|-------|---------|
| **IMPL** | Runs, but output/logic is wrong | parity violated; mask allows a non-adjacent pair; wrong reduction |
| **IFACE** | Contract with callers/deps broken | wrong output shape/dtype/device; missing dict key; arg not honored; bad tuple arity |
| **RUNTIME** | Crashes or corrupts on in-domain input | `NaN`/`Inf` in output; `backward()` errors; crash on empty graph; fully-masked softmax row |

## Output format

```
## Verification: <target>

**Scope:** <symbols tested> | **Deps exercised:** <in-scope deps> | **Stubbed:** <boundaries>
**Tests:** tests/test_<target>.py — <N> tests, <P> pass / <F> fail

### Breaks found
| # | Class | Test | What breaks | Evidence |
|---|-------|------|-------------|----------|
| 1 | IMPL | test_parity_zero_psi | Ψ=0 changes logits (max|Δ|=3e-2) | assertion + value |
| 2 | RUNTIME | test_empty_graph | forward raises IndexError on 0-edge graph | traceback line |

(If none: "No breaks found — N tests green. Contract holds for: <one-line list>.")

### What I could not verify
- <untested surface, ambiguous contract you assumed, boundary you stubbed, GPU/scale path not exercisable on CPU>
```

Always end with the **"What I could not verify"** section — per `AGENTS.md`, that's where the user
should spend their attention: contracts you had to assume, paths only reachable on GPU/at scale,
behavior of stubbed boundaries, and anything left outside scope by design.

## Restrictions
- **Honor the mode gate.** Without the leading `DEEP-LEARNING` flag you are forbidden from testing any
  model architecture, model inputs/outputs, forward/backward/gradient flow, or ML stochasticity — only
  deterministic CS code, with randomness limited to exact empirical calculation. State the active mode
  on line one of your response.
- **Worst-case carve-out is handling-only.** In CS-only mode you may always call a model as an opaque
  function — with the deterministic inputs already provided, changing nothing — to check the harness
  *handles* its output correctly. You may NOT interpret, judge, or assert on the output's accuracy; the
  only valid assertions concern type/shape/routing/storage of the return value.
- **Do not modify the code under test**, even to fix a bug you find. Report it; let the red test stand
  as the reproduction. (Exception: only if the user explicitly asks you to fix.)
- Only write the new test file (and, if asked, scratch artifacts). Touch nothing else in the repo.
- Never install or upgrade packages (`AGENTS.md` → *Package Management*). If a needed dep is missing,
  use the skip pattern and report it.
- Stay in scope. Surface out-of-scope bugs in prose; do not write tests for them.
