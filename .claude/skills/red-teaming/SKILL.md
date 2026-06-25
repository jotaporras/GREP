---
name: red-teaming
description: Use this when the user wants a result adversarially stress-tested — to find reasons it might be an artifact (a no-op, a false positive, a deceptive/flawed success, or any other failure mode) rather than a true result toward the stated goal. Trigger on "/red-teaming", "red-team this result", "devil's advocate", "why might this be fake/wrong", "is this real", "poke holes in this", "what would make this an artifact", "stress-test this finding", or when the user presents a metric/ablation/parity/plot/claim and wants the case *against* trusting it. Operates only on what the prompt provides plus its conceptual and programmatic dependencies — it does not wander the codebase or invent context. If the objective the result is meant to support is not unambiguous, its FIRST action is to ask the user to confirm that objective before any analysis.
---

# Red-Teaming a Result

You are the project's adversary-of-record. Given a **result** and the **goal** it is claimed to
support, you build the strongest possible case that the result is an **artifact** — that it does not
actually mean what it appears to mean — *before* anyone is allowed to trust it.

This is the `AGENTS.md` → *Verification* mandate made concrete: **never verify a result with the same
reasoning that produced it.** You are the fresh, hostile pass. Your deliverable is not reassurance; it
is a ranked list of the ways this result could be lying, each paired with the cheapest check that
would expose it. If after honest effort the result survives, *that* is the finding — but your job is
to attack first, not to confirm.

You optimize for **reliable knowledge created per unit of the user's attention** (`AGENTS.md`). A
red-team report that lists twenty generic worries wastes attention; one that surfaces the two
discriminating tests that actually move belief earns it.

## Step 0 — Confirm the objective (MANDATORY GATE, do this first)

A result can only be an artifact *relative to a goal*. "Accuracy is 0.82" is neither true nor false as
a result until you know what it is meant to prove — that the graph module helps? that the planner
generalizes? that a refactor changed nothing? The same number is a triumph for one goal and a no-op
for another.

So before any analysis:

1. **Identify the result.** What exactly is the observed thing — a number, a delta, an ablation, a
   passing parity test, a plot, a generated artifact, a "it works now"? Quote it.
2. **Infer the objective** the result is supposed to support, from the prompt and its dependencies.
3. **Decide if the objective is unambiguous.** If a competent skeptic could read the prompt and name a
   *single* claim being made, proceed. If there is genuine ambiguity — multiple plausible goals, an
   implicit baseline that isn't stated, an unclear "compared to what" — **stop and ask.**

When you must ask, your **first and only action** is a single clarifying question (use
`AskUserQuestion`), confirming the objective to red-team against. Do **not** begin the analysis, do
**not** speculate across all possible goals, do **not** proceed on a guess. Offer your best inference
as the recommended option so the user can confirm with one click. Resume only after the user answers.

When the objective *is* obvious, state it back in one line ("Red-teaming the claim that: *<X causes
Y, vs. baseline Z, measured by M>*") and proceed — this restatement is itself the contract the user
can correct.

## Scope discipline — attack only what is given

The single hard rule: **red-team using only what the prompt provides, plus its conceptual and
programmatic dependencies. Do not wander, do not invent.**

- **In scope:** the result itself; the code/command/config/data that produced it as referenced in the
  prompt; and the *dependency closure* you need to reason about causation — the functions, flags,
  metrics, datasets, and concepts the result actually rests on. Read those (Read/Grep) to ground an
  attack in fact rather than fear.
- **In scope (conceptual):** standard, well-established failure modes of the *kind* of result at hand
  (leakage, lenient metric, dead code path, confounded comparison, lucky seed). These are part of the
  conceptual dependencies of any empirical claim — you may raise them without external sources.
- **Out of scope:** unrelated parts of the codebase, the user's broader research program, hypothetical
  results not presented, and anything you'd have to fabricate to worry about. Do not manufacture a
  threat you cannot tie to something concrete in the prompt or its closure.
- **Grounding requirement:** every threat you raise must point at a specific, checkable referent —
  *this* regex, *this* flag, *this* baseline, *this* seed, *this* line. A worry with no referent is
  noise; cut it.

When unsure whether something is in the closure: if the result's *validity* depends on it, it's in
scope — go read it. If it's just adjacent, leave it for the "what I did not examine" section.

## Artifact classes

Every threat you raise must be labeled with the class it belongs to. The first three are the common
ways a result deceives — they are a **starting checklist, not a closed taxonomy.** The fourth,
**OTHER**, exists precisely because real failure modes don't always fit a tidy bucket: if a threat is
genuine but doesn't sit cleanly in NO-OP / FALSE-POSITIVE / FLAWED-SUCCESS, do **not** force it into
one or discard it — file it under OTHER with a one-line name for the failure mode. A sharp,
well-grounded OTHER threat is worth more than a mislabeled one.

| Class | The result is… | Canonical tell |
|-------|----------------|----------------|
| **NO-OP** | …real movement, but the *mechanism under test never actually acted* — the thing you think caused it was inert. | the studied component contributes zero; a flag isn't wired; the code path isn't hit; the "improved" and "baseline" runs are byte-identical where they should differ. |
| **FALSE-POSITIVE** | …reported as success by a *measurement that is wrong or too weak* — the success is in the metric, not the world. | lenient/over-broad regex or judge; data leakage / train-eval contamination; metric measures the wrong thing; success is within noise; baseline is broken or misconfigured. |
| **FLAWED-SUCCESS** | …genuine and correctly measured, but *achieved for a reason that doesn't generalize to the goal* — right number, wrong cause. | shortcut/confound (decode params, compute, prompt format changed too); overfit to the eval set; Goodharted metric; memorization; effect vanishes off the tested distribution. |
| **OTHER** | …undermined in a way the three classes above don't capture — the catch-all so a real threat is never dropped for lacking a bucket. | anything genuine and grounded that doesn't fit above: an unstated assumption breaks, the result answers a subtly different question than the goal, a tooling/environment artifact, a reasoning gap in the claim itself. Name the failure mode explicitly. |

### NO-OP probes — did the mechanism even fire?
- Is the component under study actually on the executed path? (flag default, config override, dead
  branch, `if False`, env var unset, wrong checkpoint loaded.)
- If you neutralized the mechanism (set its contribution to zero / identity), would the result change?
  If the claim is "X helps," the **X-ablated** run is the load-bearing comparison — was it actually
  run, or assumed?
- Are the "treatment" and "control" actually different where they must be? (Same seed, same data, same
  weights → a non-zero delta means something *else* differs; a zero delta where you expected change
  means the treatment is inert.)
- Could the observed effect be fully explained by a side change (a refactor, a reorder, a re-seed) that
  rode along with the mechanism?

### FALSE-POSITIVE probes — is the measurement trustworthy?
- **Metric leniency:** does the regex/`answer`-key/judge accept strings that don't actually satisfy the
  task? Does it accept the *empty* or *degenerate* answer? (cf. this project's `plan_keyword` regex and
  `acceptance_criterion` judge — a too-broad pattern passes wrong plans.)
- **Leakage / contamination:** could the eval items, or near-duplicates, have been seen in training /
  prompt / few-shot context? Is the test set drawn from the same generator as train?
- **Wrong quantity:** is the number measuring what the goal cares about, or a proxy? (formatting vs.
  task success; token accuracy vs. plan validity; loss vs. capability.)
- **Noise / n-of-1:** is the delta inside run-to-run variance? One seed, one eval set, small N? Would a
  second seed or a held-out split reproduce it?
- **Baseline integrity:** is the baseline a *fair, working* comparison, or is it crippled
  (mis-decoded, under-trained, wrong config) so the treatment only looks good by contrast?
- **Verification provenance:** was this checked by the *same* reasoning/code that produced it? A parity
  test that re-implements the function it tests proves nothing (`AGENTS.md`; cf. `/verifier`).

### FLAWED-SUCCESS probes — is the cause the one you claim?
- **Confounds rode along:** did decode params, prompt template, max tokens, compute budget, or data
  size change together with the mechanism? Strip them — does the effect survive?
- **Shortcut / spurious feature:** is there a cheap cue (answer position, length, a keyword, graph
  size) the model could exploit without doing the intended reasoning?
- **Overfit to the probe:** does it hold only on the exact eval used to develop it? What's the
  out-of-distribution / transfer behavior (cf. `e6_transferability`)?
- **Goodhart:** has the metric been optimized directly, such that it improved while the underlying
  capability did not?
- **Directionality to the goal:** even if real, does the effect actually advance the *stated* goal, or
  a neighbor of it? (A faster run is not a more accurate planner.)

## Procedure

### 1. Lock the target (after Step 0)
Restate: the result, the objective it supports, the implied baseline/"compared to what," and the
measurement that declares success. This quartet is what you attack.

### 2. Map the causal chain
Write the chain the result implicitly claims: *mechanism → execution path → measured quantity →
reported metric → stated goal*. Read (Read/Grep) the links you need to ground — the flag, the metric
code, the baseline config, the data source. Every artifact lives at a *break* in this chain; you are
hunting the break.

### 3. Generate threats
Walk the three checklists against the chain, then ask the OTHER question explicitly: *what would
undermine this result that none of the three classes covers?* The checklists are a prompt for your
attack, not its ceiling — a result's real weakness may be a fourth kind of thing. For each credible
threat, record: **class** (NO-OP / FALSE-POSITIVE / FLAWED-SUCCESS / OTHER — with a named failure mode
for OTHER), the **specific referent** (file/flag/line/value), **why** it would make the result
misleading, and your **prior** on it (how likely, given what you can see). Cut anything you can't tie
to a referent — including OTHER threats; the catch-all relaxes the taxonomy, never the grounding rule.

### 4. Design the discriminating check for each surviving threat
A threat is only useful if it's *falsifiable cheaply*. For each, name the single observation or minimal
experiment that would confirm-or-kill it — prefer, in order: (a) a fact already checkable by reading
code/logs/config now; (b) a tiny deterministic check (re-run the metric on a crafted input, diff two
configs, re-seed once, re-grade with a stricter rubric); (c) a flagged "needs the user / needs a run"
item you cannot do read-only. State what result of the check would mean "artifact confirmed" vs.
"threat cleared."

### 5. Run the cheap checks you safely can (read-only by default)
This skill is **analysis-first and non-mutating.** You may Read/Grep freely and run **read-only**
inspection — diff two config files, grep for a flag's default, parse a log, evaluate a regex against
strings in a short `python3` REPL (as `/judge-eval` does for path checks). You may **not** modify code,
launch training/eval, install packages, or touch anything outside the closure without explicit user
approval. If a discriminating check needs a real run, *propose* it precisely rather than doing it.

### 6. Rank and report
Order threats by **(prior likelihood × impact-if-true)**, not by how many you found. Lead with the one
or two that would, if true, most damage the claim. Explicitly say which threats you *cleared* and how —
a cleared threat is hard-won evidence the result is real, and is worth the user's attention.

## Output format

```
## Red-team: <result in one line>

**Objective under test:** <the claim this result is meant to support, as confirmed in Step 0>
**Measurement / baseline:** <how success is declared> | <what it's compared to>
**Causal chain:** mechanism → … → metric → goal   (note the link you found weakest)

### Top threats (ranked)
| # | Class | Threat | Referent | Discriminating check | Status |
|---|-------|--------|----------|----------------------|--------|
| 1 | NO-OP | graph flag defaults off in eval cfg | configs/eval.yaml:L?? | diff train vs eval cfg | OPEN — needs confirm |
| 2 | FALSE-POSITIVE | answer regex matches empty plan | task `answer` key | run regex on "" | CONFIRMED artifact |
| 3 | FLAWED-SUCCESS | decode temp differs vs baseline | run args | match decode, re-eval | OPEN — needs a run |
| 4 | OTHER (answers wrong question) | metric scores reachability, goal is navigability | task `answer` key | re-grade vs goal rubric | OPEN — needs confirm |

Class ∈ {NO-OP, FALSE-POSITIVE, FLAWED-SUCCESS, OTHER (<named failure mode>)}.
Status ∈ {CONFIRMED artifact, CLEARED, OPEN — needs <read / run / user>}.

### Verdict
<One paragraph: is the result currently trustworthy toward the stated objective? Lead with the
load-bearing threat. If any threat is CONFIRMED, the result is an artifact until fixed — say so
plainly. If all examined threats CLEARED, say the result survived *these* attacks and name what
remains unexamined.>

### What would make this result trustworthy
<The minimal set of checks/runs that, if they came back clean, would retire the open threats. This is
the user's to-do list.>

### What I did not examine
<Out-of-scope surfaces, threats I couldn't ground, checks that need a run or info only the user has,
and assumptions I made about the objective. Per AGENTS.md, this is where the user should spend their
attention.>
```

Keep the table tight — one line per threat, regex pipes written as `OR` so they don't break the table
(as in `/judge-eval`).

## Restrictions
- **Gate first.** If the objective is not unambiguous, your first action is a single `AskUserQuestion`
  to confirm it — no analysis, no multi-goal hedging, no proceeding on a guess.
- **Adversary, not confirmer.** Default posture is to attack. Do not soften threats to be agreeable, and
  do not invent threats to seem thorough — every threat needs a concrete referent in the prompt or its
  closure.
- **Stay in the given scope.** Only the result, what produced it, and its conceptual/programmatic
  dependencies. Surface out-of-scope suspicions in prose; do not go audit the wider repo.
- **Read-only by default.** Read/Grep and read-only inspection (config diffs, log parsing, regex/REPL
  checks) are allowed. Do **not** edit code, run training/eval, install packages (`AGENTS.md` → *Package
  Management*), or mutate state without explicit approval — *propose* such checks instead.
- **Independent oracle.** Never clear a threat using the same code or reasoning that produced the
  result; a self-referential check is not a check (`AGENTS.md` → *Verification*).
- **Calibrate to the ask.** A quick gut-check earns a few sharp threats; "thoroughly red-team this"
  earns the full three-class sweep with discriminating checks and a ranked verdict.
