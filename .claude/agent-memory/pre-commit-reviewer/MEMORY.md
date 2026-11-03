# Pre-Commit Reviewer Memory

## Project Style Rules (from AGENTS.md)
- No defensive programming: no `try/except ImportError`, no `if x is not None` guards, no silent fallback branches
- No `try/finally` blocks unless explicitly requested
- Straight-line code; let it fail loudly on unexpected paths
- Module-level imports using `from prism.X import Y` then `Y.func()` style
- 4-space indent, snake_case modules/functions, PascalCase classes, rich type hints
- No defaults in `__init__`, prefer argparse defaults instead
- `git add -A` is forbidden; secrets live in env vars, never committed

## Current Architecture (verified 2026-08-24, e20 review)
NOTE: `train_v2.py`, `run_eval.py`, `data_col.py` no longer exist in this repo —
the 2026-03-02 notes about them are obsolete and were removed from this file.
- Training entry point: `src/prism/training/train_v3.py` (`_validate_config`, `train_model`,
  `gnn_config`/`train_config` dicts written to `train_config.json` in the checkpoint dir)
- Eval entry point / library: `src/prism/eval/evaluate.py` (`eval_model_single_graph`,
  `eval_model_multiple_graphs`, `evaluate_model`) — has a known pre-existing `# TODO` about an
  import-time monkey-patch being global state (line ~40), not a blocker, just noted.
- Eval checkpoint config resolution: `src/prism/eval/checkpoint.py` — a family of
  `resolve_<thing>(checkpoint)` helpers (`resolve_prompt_policy`, `resolve_edge_weights`,
  `resolve_injection_scope`, `resolve_response_format`, ...) that read `train_config.json`,
  falling back to a nested `"gnn"` sub-dict, and default to the historical value if the key is
  absent — this is THE pattern for making new config axes backward-compatible with old
  checkpoints. Follow this exact shape when adding a new training-time config axis.
- Data pipeline: `src/prism/data/data.py` (`preprocess_dataset`, `load_and_split_dataset`),
  `src/prism/data/compact_prompt.py` (prompt/response rendering — `build_conversation`,
  `spine_to_compact_messages`, `_format_assistant`, `_answer_contract`), `src/prism/data/data_gen.py`
  (`DataGenerator` — SPINE rollout generation), `src/prism/data/utils.py` (`aggregate`, `strip_icl`,
  `try_load_json`, graph utils)
- Data generation CLI: `scripts/training_data_generation/generate_data_spine.py` (Phase 1 LLM
  populate + Phase 2 rollout generation); `scripts/training_data_generation/split_train_val.py`
  is the REAL source of the training corpus consumed by `train_v3.py`
  (`formatted_all_new__{train,val}.json`) — its `SAMPLE_RE = r"^sample_(\d+)_(\d+)\.json$"` strictly
  excludes `*_failed.json`. `data_gen.py`'s own `utils.aggregate(..., glob_str="sample*json", ...)`
  → `formatted.json` is a separate, less-strict diagnostic byproduct, NOT what training reads.
- Inference/eval model wrappers: `src/prism/models/inference.py` (`InMemoryLLM`,
  `GraphAugmentedInMemoryLLM`), `src/prism/models/vllm_graph/spine_client.py`
  (`VLLMInMemoryLLM`, `VLLMGraphInMemoryLLM`), `src/prism/models/loaders.py`
- Callbacks: `src/prism/eval/callbacks.py` (`EvalCallback`, `GradientDebugCallback`)
- Scalability/large-eval driver: `src/prism/eval/scalability_evaluation.py` — sets
  `CUDA_VISIBLE_DEVICES` itself from `--device`; don't rely on the env var being respected.
- Teacher LLM clients for data-gen: `src/prism/data/vllm_llm.py` (`VLLMSpineClient`, thread-safe
  via an engine lock + micro-batching gate — `_gated_generate`/`_ENGINE_LOCK`/`_PENDING_LOCK`) and
  `src/prism/data/local_llm.py` (`GemmaSpineClient`, HF backend, NOT thread-safe).

## Cross-cutting config-axis wiring pattern (seen with `data.response_format`, e20)
When a new train-time knob needs to be visible at eval time too, the established, fully
consistent seam is:
`experiments/*.yaml` default → `train_v3._validate_config` (fail-loud validation + cross-field
compatibility checks) → recorded into both trainer's `train_config`/`gnn_config` dicts → threaded
as a kwarg through `data.preprocess_dataset` / `EvalCallback` / `evaluate.eval_model_*` /
`evaluate_model` / `inference.InMemoryLLM`/`GraphAugmentedInMemoryLLM` / `vllm_graph.spine_client`
clients → recovered at eval-rebuild time via a new `checkpoint.resolve_<axis>` helper (see above).
Guard cross-field incompatibilities (e.g. `route_only` requiring `spine_tools=none`,
`icl_examples=0`, eval tools off) at EVERY layer that touches the axis, not just the top —
`train_v3`, `data.py`, `compact_prompt.py`, and `evaluate.py` each independently re-validate the
same invariant, which is intentional defense-in-depth, not duplication to trim.

## Data-gen rollout protocol conventions (`data_gen.py`, verified e20 review)
- Atomic writes everywhere: write to `<name>.partial` then `os.replace(tmp, final)` — both the
  success path (`_commit_plan_sample` et al.) and the failure path (`_quarantine`) use this.
  `_quarantine(failed_name, payload: dict)` is a single shared helper used by every rollout mode
  (spine/path_only/oracle); its payload is always a flat dict (never a message list), which is why
  `utils.aggregate`'s glob (`"sample*json"`, which also matches `*_failed.json`) is harmless — a
  dict fails `strip_icl`'s list-shape assumption and is silently skipped by aggregate's own
  `except Exception: print(...)` — verified, not a bug.
- Resume support: `_has_valid_rollout(path)` checks the FINAL (non-`.partial`, non-`_failed`) file
  is a parseable list of role/content dicts with a strippable ICL prefix, before deciding to
  re-run a task — lets a killed/resumed job skip already-good samples.
- A stale `*_failed.json` left over from an earlier failed attempt, followed by a later successful
  retry producing `*.json`, is handled explicitly by stats code checking for the paired `.json`
  before counting a `_failed.json` as a real failure.
- New non-SPINE rollout modes (`path_only`, `oracle` — e20) reuse this exact protocol; per-thread
  safety is achieved by always `dict(graph)`-copying before mutating (e.g. setting
  `robot_location`), never mutating the shared `graph` object passed to a `ThreadPoolExecutor` job.
- `ThreadPoolExecutor(max_workers=rollout_workers)` gate: workers > 1 requires either
  `VLLMSpineClient` (thread-safe) or a client-free mode (oracle); this is enforced with a
  `raise ValueError` before submission, not a silent fallback.
- Deterministic/leak-free grading: `path_validator.validate_structured(..., full_response=None)`
  disables the Gemma-judge/rescue path entirely — pure regex/NetworkX, safe to call concurrently
  and safe to call on data that must never see ground truth.

## Leak-check convention (generated training data must never contain GT strings)
Standard invariant, enforced via `grep -RlE 'acceptance_criterion|"answer"[[:space:]]*:' <dir>`
(exit 2 / abort the pipeline on a match) in data-gen sbatch scripts, run against
`generated_plans/` before any split/training step consumes it. When reviewing new data-gen code
paths, check every place a prompt or log payload is assembled that any GT field
(`task_entry["answer"]`, `task_entry["acceptance_criterion"]`) never gets serialized into it.

## Cluster/sbatch conventions (see also global CLAUDE.md)
- Paths under `/vast/projects/aribeiro/alelab/jporras/GREP-PRISM` (`$ALELAB_DRIVE/GREP-PRISM`)
- `dgx-b200` partition requires EXACTLY `--cpus-per-task=28 --mem=224G` per GPU (cli_filter enforced)
- Env dirs: `$ALELAB_DRIVE/envs/GREP-PRISM-v3` (training), `$ALELAB_DRIVE/envs/GREP-PRISM-vllm`
  (vLLM-backed data-gen/inference) — both are established, real envs used across many sbatch scripts
- `STAGE1_LORA` / `PE_GT` pattern: sbatch scripts define these as plain (non-exported) shell vars,
  then `source scripts/e18_arms.sh` / `source scripts/e19_arms.sh` — the sourced functions read
  them from the shared shell scope even though the sbatch script's own body never references the
  var again textually. Not a bug/unused-var when you see this shape — check the sourced arm file
  before flagging.
- Auto-resume retry loops (`while true; ... attempt=$((attempt+1))`) around a single `srun` call
  are a standard pattern for flaky multi-hour data-gen jobs, paired with `_has_valid_rollout`-style
  idempotent resume inside the Python side.

## Review methodology note
If no Bash tool is available in a review session, fall back to reading full current file contents
(Read/Grep/Glob) instead of diffing — cross-check against the explicit file-scope list the
requester gives you, and always disclose this substitution in the final report so the user can
independently confirm via `git diff` that nothing outside the described change crept in.
