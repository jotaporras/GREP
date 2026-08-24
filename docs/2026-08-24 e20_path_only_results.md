---

## experiment: e20 path-only prediction — results
date: 2026-08-24
status: results note — 5 of 6 cells complete; v3/text_edges control in cross-eval (updated in place)
sources: `outputs/e20_path_only_pred/*/eval_logs/` on betty, wandb tag `e20_path_only_pred`; design in [2026-08-24 e20_path_only_pred_design.md](2026-08-24%20e20_path_only_pred_design.md)

# e20 — strictly-path-prediction targets on n60

The adviser's hypothesis: the reasoning text in the SFT targets confuses the graph
model, and the task should be posed as *strict path prediction* — the target is the
bare node sequence `a -> b -> c`, nothing else. e20 tests this with a new
`data.response_format=route_only` axis (no `<think>` scaffold, no tools, no ICL; the
eval seam wraps the bare route back to `[answer(...)]` so grading is unchanged) and
two new corpora, evaluated on the **same frozen n60_v3 test graphs** (052–058, 84
questions) as e18/e19, so every number below is directly comparable to e19's
hop_depth 73.8 / cn7ub88q 72.6 / mask_a 56.0.

## 1. The two arms (what "edges present" means)

- **`hop_depth`** — the best e19 architecture by eval/accuracy (the v6sy0uhi recipe):
  learnable graph mask (e18 `mask_a`, decision gating 3.0) + V2 hop-separated
  post-fusion in depth mode (k=3, open gates, codebook 256). **`data.text_edge_list=none`**
  — the prompt contains *no* edge list; graph structure reaches the model only through
  the learned graph channel.
- **`text_edges`** — the plain-LLM baseline: `gnn.arch=llm`, base Gemma-4-31B + a
  fresh LoRA, **no graph tower at all**; the edge list is written into the prompt as
  text (`data.text_edge_list=present`).

So "edges present = True" is the *baseline LLM*, not a graph-enabled arch, and the
runs "without edges" are the graph arch — they still consume the full edge set, just
through Ψ embeddings instead of tokens. There is no LLM-with-no-edges cell in e20;
per e14, that configuration is a floor, and text_edges is the real control.

## 2. Corpora

| DATASET | corpus | graphs | targets | train/val |
|---|---|---|---|---|
| `pathonly` (e20a) | `n_60_pathonly_v1` | the 52 v3 train graphs, re-answered | Gemma-4-31B single-turn, **no thinking**, route-only; failures quarantined | 579 / 80 |
| `oracle` (e20b) | `n_60_oracle_v1` | **156 fresh skeletons (seeds 401+), 153 populated** — not v3 | NetworkX shortest paths via `derive_targets`, LLM-free, 1836/1836 self-verified | 1764 / 72 (3.05× e20a) |
| `v3` (control) | `n_60_vllm_v3` | original e19 corpus | original reasoning rollouts, `think_route` | original split |

All graphs ~63 nodes (48 regions + ~15 objects); eval is always the v3 `test_graphs`.

**Teacher-accuracy monitor** (the explicit design gate): no-think route-only
Gemma-4-31B passes **659/708 = 93.1%** of rollout gradings (all 49 failures
`wrong_route`, zero `no_route` — perfect format compliance) vs **696/708 = 98.3%**
for the same graphs regraded from the v3 reasoning rollouts. A −5.2 pt drop, not a
collapse — far above the ~50% abort threshold, so the e20a corpus is usable.
Zero-shot reference on the frozen test split: the base model answers route-only at
**80/84 = 95.2%** with text edges in the prompt (vLLM generation, not the HF eval
harness).

## 3. Results

Accuracy on the frozen n60_v3 test set (84 questions). Per-epoch numbers are the
in-memory `EvalCallback` family; cross is the post-train from-disk reload
(`eval_logs/cross_eval/`). Per [two eval families] these are NOT interchangeable —
compare within a column only.

| cell (job, wandb) | arch | corpus / format | e1 | e2 | e3 | cross |
|---|---|---|---|---|---|---|
| `e20_pathonly_hop_depth` (7827254, wkext9v3) | graph | pathonly / route_only | 69.0¹ | 77.4 | 84.5 | 83.3 |
| `e20_pathonly_text_edges` (7827257, zalttwkb) | LLM+text edges | pathonly / route_only | 94.0 | 91.7 | **98.8** | 89.3 |
| `e20_oracle_hop_depth` (7828117, sfixkba6) | graph | oracle / route_only | 75.0 | 77.4 | **92.9** | **92.9** |
| `e20_oracle_text_edges` (7828118, m2kn8nqs) | LLM+text edges | oracle / route_only | 79.8 | 78.6 | 81.0 | 89.3 |
| `e20_v3_hop_depth_think_route` (7829068, yifhyvxn) | graph | v3 / think_route | 66.7 | 65.5 | 73.8 | 70.2 |
| `e20_v3_text_edges_think_route` (7829067, 5q0j7owo) | LLM+text edges | v3 / think_route | 86.9 | 89.3 | 92.9 | *cross running* |

¹ Up to 8 epoch-1 samples in the route_only cells crashed on a fail-loud guard
(`route_only target has no extractable route`, fired on the model's own degenerate
single-node answers on the live-inference path) and were counted wrong; epochs 2–3
are unaffected. Fix (a `strict_route=False` seam for live inference) is written,
pending tests. Epoch-1 route_only numbers are therefore mildly pessimistic.

Reference points: e19 hop_depth think_route best = **73.8** per-epoch (v6sy0uhi);
same-config variance band on n60 ≈ 10 pts; base-model zero-shot route-only with
text edges = **95.2**.

### Path-quality metrics, `e20_oracle_hop_depth` final

`edge_val` 1.00 and `halluc` 0.00 on **all 7** test graphs; `valid_path` 0.83–1.00;
`hop_opt` 1.04–1.67. Per graph: 052 83.3 · 053 100 · 054 100 · 055 100 · 056 100 ·
057 83.3 · 058 83.3.

### Error-level comparison, e19 hop_depth → e20 pathonly hop_depth

20 questions fixed, 11 regressed, net +9 (62→71 of 84). Of e19's 22 errors, 11 never
emitted a parseable route at all (malformed `goto(` tool-call syntax) — i.e. half of
the e19 error budget was *format*, not navigation, and route_only eliminates that
class by construction. Hallucinated-edge answers under route_only: 2/84 (hop_depth,
both graded wrong) and 1/84 (text_edges).

## 4. Reading of the results

1. **The adviser's hypothesis is supported — strongly.** For the graph arch,
   holding everything else fixed, targets went reasoning→route-only and accuracy
   went 73.8 (e19 best-ever) → 84.5 (pathonly) → **92.9 (oracle)**. The in-harness
   think_route control (yifhyvxn) landed on **73.8 at epoch 3 — an exact
   replication of the e19 v6sy0uhi number** — so the jump is the target format
   and corpus, not the harness. The 2×2 is closed for the graph arch:
   think_route 73.8 vs route_only 84.5/92.9.
2. **92.9/92.9 is the best graph-channel result in the project by ~19 pts**, it is
   the first run where per-epoch and cross-eval agree exactly, and it is achieved
   with *zero* edges in the prompt — 2.3 pts below the base model's zero-shot
   ceiling *with* text edges (95.2, different harness, ±noise on 84 questions).
   The graph channel is, for the first time, carrying essentially the full edge
   information.
3. **Oracle vs pathonly is confounded** (3.05× more data AND cleaner targets AND
   fresh graphs). Separating size from purity needs an oracle-subsampled-to-579
   cell if we care.
4. **The text baseline inverts across corpora**: 98.8 on pathonly (self-distillation
   — Gemma fitting its own no-think route distribution) but only 78.6 at e2 on
   oracle, with NO epoch-3 rescue (final 81.0 per-epoch / 89.3 cross). On the
   oracle corpus the graph arch beats the text-edge baseline by 12 pts per-epoch
   (92.9 vs 81.0) — the first clean win for the graph channel over its control.
   Plausible reading: NetworkX tie-broken shortest paths are off-policy for the
   LLM's text-search priors, while the graph channel has no such prior to fight.
5. **Known grading gaps**, both small: hop_opt up to 1.67 means non-shortest valid
   routes count as correct, and one e20 "correct" answer visits `command_deck_1`
   twice (no simple-path check in the structural grader). Neither changes the
   ordering; both are worth fixing before a writeup.

## 5. Open items

- 7828118 / 7829067 / 7829068 finishing (monitor armed); table updated when they land.
- Commit the `strict_route` live-inference fix with tests (`tests/test_e20_route_only.py`).
- Optional cells: oracle subsampled to 579 (size vs purity), base-model eval through
  the actual HF eval harness (make 95.2 exactly comparable), simple-path grader check.

[two eval families]: ../docs (per-epoch in-memory vs cross_eval from-disk reload differ by up to 9 pts on this stack; see memory note 2026-08-24)
