# Repository Guidelines
You're a research scientist, writing code like one. You care about the result of the experiments, not  the modularity or extensibility of the code. The code should be easy to understand and follow, even if it comes at the expense of modularity or being error-prone in unfamiliar execution paths.

## Cluster (SLURM) config for this project

Used to fill in placeholders in `~/.claude/skills/cluster-slurm/template.sbatch`:

- `PROJECT`: `GREP-PRISM`
- `ENTITY`: `alelab` (wandb)
- `ENV_NAME`: `/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM`
- `OUTPUT_BASE`: `/vast/projects/aribeiro/alelab/jporras/GREP-PRISM`
- SLURM output dir: `/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%A_%a.out`
- Repo path on cluster: `/vast/projects/aribeiro/alelab/jporras/GREP-PRISM`
- Default partition: `dgx-b200`
- Entry: `python -m prism.training.train_v2 <yaml> [--key value ...]` (yaml + CLI overrides supported since the e4 sweep)


## Project Structure & Module Organization
- `src/prism/` houses the installable package; subfolders map to simulation assets, evaluation logic, planners, and LoRA tooling.
- `scripts/` contains runnable entry points for data generation, aggregation, training, and evaluation.
- `data/` stores seed eval graphs and generated corpora; keep new experiments under `data/eval/` or sibling folders.

## Environment Setup
- Empirically stable dependency set:
  ```
  torch
  transformers==4.51.2
  datasets==3.5.0
  accelerate==1.6.0
  peft==0.15.2
  trl==0.15.2
  unsloth==2025.3.19
  unsloth_zoo==2025.3.17
  safetensors==0.5.3
  sentencepiece==0.2.0
  tokenizers==0.21.1
  numpy==1.26.4
  tqdm==4.67.1
  wandb==0.19.9
  openai==1.70.0
  bitsandbytes
  tiktoken
  ```
- Clone the PRISM branch of SPINE for planner interoperability:
  ```bash
  git clone git@github.com:KumarRobotics/SPINE.git
  cd SPINE
  git checkout feature/prism
  git pull
  ```
  Ensure SPINE is importable (editable install or `PYTHONPATH`).

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` restores the broader CUDA-ready toolchain after the minimal set above.
- `python -m pip install -e .` enables `prism.*` imports in scripts and notebooks.
- `python scripts/generate_data_spine.py --n-samples 10 --n-tasks 3 --name demo` creates SPINE training data; other generators share similar flags.
- `python scripts/train_llama3_llora.py --name llama3_demo --data data/gpt_gen_formatted.json` launches LoRA SFT; adjust `--bit4`/`--r` as needed.
- `python scripts/eval.py` scores checkpoints against `data/eval/` tasks; capture accuracy for review.

## Coding Style & Naming Conventions
- Use Black/Isort/Flake8 (see README); run `pre-commit install`, then `pre-commit run --all-files` before committing.
- Stick to 4-space indentation, snake_case modules/functions, PascalCase classes, and rich type hints.
- Keep configuration constants uppercase and load secrets from environment variables, not committed files.
- Avoid defaults in Python class `__init__`, prefer defaults in argparse arguments instead.

## Testing Guidelines
- Tests live under `tests/`; run with `conda run -n GREP-PRISM python -m pytest tests/ -v`.
- Existing suites: `test_scene_graph_parser.py`, `test_sim.py`, `test_bucketize_prompt.py`, `test_remove_edge_list.py`.
- `test_sim.py` covers `GraphSim.take_action` and SPINE plan parsing; uses an inline `_DummyClient` to avoid LLM calls.
- Keep repro artifacts small; expand `data/eval/` fixtures for new eval scenarios.

## Commit & Pull Request Guidelines
- Keep commit subjects short, Title Case (e.g., `Update README.md`), and isolate unrelated edits.
- In PRs, state the goal, link issues or experiment logs, and paste relevant command outputs (train/eval metrics).
- Run the necessary data, training, and eval steps before review; call out anything you could not validate.
- Tag reviewers closest to the touched modules and highlight breaking changes or new dependencies early.

## Package Management
- NEVER install, upgrade, or remove packages without explicitly telling the user first and getting approval.
- The full working conda env is `GREP-PRISM`. Always activate it before running Python.

## Security & Configuration Tips
- Keep API keys (OpenAI, Hugging Face, WANDB) and filesystem paths in environment variables or `.env`; never commit secrets.
- Store large checkpoints and generated datasets outside the repo; commit metadata and small samples only.


## AVOID DEFENSIVE PROGRAMMING.

NEVER WRITE code like this

```python
try:
    import wandb  # Local import to avoid hard dependency.
except ImportError:
    wandb = None
```
or 
```python
if wandb_kwargs:
  try:
      import wandb
  except ImportError as exc:
      raise ImportError(
          "wandb is required but not installed while report_to is set to 'wandb'."
      ) from exc
  if wandb.run is None:
      wandb_kwargs.setdefault("reinit", True)
      wandb.init(**wandb_kwargs)
```

These are examples of defensive programming. We do not like it. Always assume that
the right code execution path will be taken. The only exception is if the user specifically requests a specific check.

This also includes **`if x is not None`** guards and fallback branches. For example, do NOT write:
```python
if EVAL_TOKEN_ACC is not None:
    # ... do the real work ...
else:
    print("Column not found, here are the available columns:")
```
If a variable is expected to have a value, use it directly. Let it fail loudly if the assumption is wrong.

## AVOID SILENT ERRORS.

NEVER CODE for avoiding runtime exceptions on unexpected paths. If an experiment runs into an unexpected branch and doesn't fail, that's not good coding, it's sabotaging our experiments. We want to know when something goes wrong, not avoid it. Neural network development is very sensitive to silent errors and small changes, and we want to always be aware about it.

Never use `try/finally` blocks unless the user explicitly requests one. Write straight-line code instead.

## Known Architectural Limitations

### Multi-graph PE injection is not yet supported
During inference with ICL (in-context learning) examples, SPINE prompts contain multiple scene graphs — one per ICL example plus the real task graph. Currently, `_generate_tokens` in `src/prism/models/inference.py` only injects PE for the **last** graph (the real task). This is because `build_injection_map` does a global token-sequence search across the entire prompt, and shared node names between graphs (e.g. `shed_1`, `field_11`, `example_node_1`) cause PE from different spatial layouts to contaminate each other's token positions — a distribution shift never seen during training (which always has exactly one graph per sequence).

**To properly support multi-graph PE injection**, the injection pipeline needs per-message token boundaries so each graph's `build_injection_map` call is restricted to only its source message's token range. This requires:
1. Tracking which message each parsed graph came from (`_parse_all_pyg_graphs`)
2. Computing per-message token offsets from `apply_chat_template` (e.g. via incremental tokenization or `return_offsets_mapping`)
3. Passing `(start_offset, end_offset)` into `build_injection_map` / `bucketize_prompt`

## Data Modifications
- The training/eval JSON files (`data/gen/spine_exp1/formatted.json`, `data/gpt_gen_formatted.json`, `data/eval/gpt_gen_formatted.json`) were post-processed with `scripts/fix_training_json.py` to fix malformed JSON in 17 assistant responses (missing commas between key-value pairs and one unquoted string value).

## Cursor Rules

Project-specific AI coding rules live in `.cursor/rules/`. Check there for conventions before making changes to areas covered by those rules.

Current rules:
- `notebooks.mdc` — path conventions for notebooks in `notebooks/` (use `../` prefixes, never `os.chdir`)

