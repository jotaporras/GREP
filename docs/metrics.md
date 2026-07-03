# GREP-PRISM evaluation metrics — the complete catalog

Internal reference: every metric the stack computes, where it lives, what it
measures, and — critically — what it is **blind** to. Written after the e11
investigation showed a saturated token metric hiding a broken model for months.

## The one rule

**Only free-generation task accuracy is causally connected to the research
question.** Every teacher-forced metric conditions on the gold prefix (and, under
`injection_scope=full_sequence`, on injection channels generation never has). Use
teacher-forced metrics to debug *mechanisms*, never to claim *capability*.

## 1. Generation (task-level) metrics — the ground truth

Computed by the deterministic grader in `src/prism/eval/path_validator.py`
(`validate_path`), aggregated by `aggregate_path_metrics`; logged per train-time
eval by `EvalCallback` (`src/prism/eval/callbacks.py`) and written to
`eval_logs/step_*.json`.

| key | meaning | notes |
|---|---|---|
| `eval/accuracy` | sample-weighted micro-average of per-graph objective correctness | RegEx/NetworkX keyword + path-validity check; THE headline number |
| `eval/acc/<graph>` | per-graph accuracy | 0.1-grained at 10 tasks/graph |
| `eval/valid_path_rate` | fraction of solvable A→B tasks with a fully valid route | denominator: solvable A→B samples only |
| `eval/path_optimality_rate` | valid AND cost-optimal routes | same denominator |
| `eval/hallucination_rate` | 1 − edge-validity over emitted hops | denominator: ALL samples with ≥2-node routes — different from the two above |
| `grep/path_*` | the remaining per-sample path metrics, averaged | see per-sample list below |

Per-sample fields (inside `eval_logs/*.json` `samples`): `nodes_exist_rate`,
`edge_validity_rate`, `full_path_valid`, `start_goal_ok`, `cost_optimality`
(emitted÷shortest weighted cost), `hop_optimality` (unweighted),
`path_from_reasoning` / `path_rescued` (provenance flags — route recovered from
the reasoning text / the gated Gemma rescue; disable rescue with
`GREP_PATH_RESCUE=0`), `judge_used` / `llm_judge_pass` (LLM-as-judge grades
subjective/yes-no tasks ONLY; judge and RegEx scores are kept disjoint).

## 2. Teacher-forced token metrics — mechanism debugging only

Computed by `GraphTokenAccuracyMixin` (`src/prism/eval/evaluate.py`) during
training forwards:

| key | meaning | blindness |
|---|---|---|
| `graph_acc/scene_block` | next-token argmax accuracy over node-name tokens in the prompt's scene block | prompt tokens; fine as a formatting sanity check |
| `graph_acc/answer_nodes` | same, over node-name tokens in the answer | **~96% of these are name-completions/repeats copyable with zero graph knowledge — the no-graph baseline scores 0.979.** Never cite as evidence the graph channel works. |
| `train/eval mean_token_accuracy` (TRL) | argmax accuracy over the loss-target span | with `loss_target=edge_list`, dominated by format/source/completion tokens; the adjacency decisions are ~10% of the span |
| `eval_loss` | CE over the loss-target span | inherits any train-time leak; the sharpest of the teacher-forced signals |

The honest teacher-forced instrument is the **decision-token diagnostic**
(`scripts/diag_injection_ablation.py` + `src/prism/eval/injection_diag.py`):
grades ONLY the first token of each node's first answer mention (the positions
where adjacency is actually required) under ablated injection conditions
(`train_style` / `prompt_only` / `no_injection`, plus `--gate-sweep`). Reading
rules are in that script's docstring; results land in `injection_diag*.json`
inside the checkpoint dir.

## 3. Training-dynamics metrics (`debug/*`)

`GradientDebugCallback` (`src/prism/eval/callbacks.py`), gated by
`trainer.gradient_debug`:

- `debug/grad_norm_{gnn,lora,pe_proj,pe_gain,gt_blocks,rpearl}` — per-component
  gradient norms, captured post-backward/pre-zero.
- `debug/pe_gain` — the RAW gate parameter (effective gate = `tanh(pe_gain)`).
  **An amplitude, not an ablation** — a rising gate does not mean the channel is
  useful (e11 lesson), and the trained value sits at the interference optimum
  (e12 gate sweep), so do not chase it upward.
- `debug/pe_output_norm`, `debug/pe_has_nan`, `debug/embedding_norm`,
  `debug/num_injections`, `debug/lr` — magnitudes/plumbing sanity.

## 4. Where results live on disk

```
<run_dir>/
  eval_logs/step_XXXXXX_epoch_Y.json   # per-interval generation eval (full samples)
  eval_logs/cross_eval/<graph>.json    # post-train per-graph eval
  injection_diag*.json                 # decision-token diagnostics (if run)
  train_config.json                    # run metadata (+ "gnn" section for graph archs)
```

Post-hoc tools: `src/prism/eval/scalability_evaluation.py` (re-evaluate a saved
checkpoint on arbitrary graph dirs), `scripts/apply_judge_to_eval_run.py`
(re-grade an eval JSON with the judge), `scripts/visualize_path_metrics.py` /
`src/prism/eval/render.py` (figures), `scripts/eval_viewer.html` (browse eval
JSONs).

## 5. Reporting hygiene (project policy)

1. Headline claims come from `eval/accuracy` on the full held-out graph set
   (`eval.num_graphs=-1`), everything else is supporting mechanism evidence.
2. Never report token accuracy over node spans without the
   decision/completion/repeat split.
3. Never report a gate/amplitude scalar as evidence of channel use.
4. Every architecture change ships with a single-flag A/B against a named
   historical run (wandb id in the lab note).
