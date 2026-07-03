# Repository Guidelines
You're a research engineering assisstant. You will optimize for **reliable knowledge created per unit of my
attention**, not lines of code or speed.

## Preliminaries
This codebase is inherited from a previous paper called `Distilling On-device Language Models for Robot Planning with Minimal Human Intervention`,
where the approach was to distill a small model on a stron model's outputs. In turn, this forks `SPINE: Online Semantic Planning for Missions with 
Incomplete Natural Language Specifications in Unstructured Environments`, which introduces an interactive environment for a robotic planner
to navigate. Spine shuld be forked to a specific branch tailored to this project under `SPINE`.

## Graph-Enhanced Planner (GREP) Project overview
The goal of this project is to design an LLM-based planning architecture on top of the SPINE/PRISM enhanced with scene graphs
via some graph-based architecture. We observe that

1. LLM-based planners are limited in their ability to navigate scene graphs, especially large ones.
2. Text-based graph representations hinder the planners scalability and generality. 

As part of the project, we will experiment with novel ways of integrating graph structure, including but not limited to: 
learnable graph positional encodings, masking, graph transformers, etc. The ideas that we try come from principled ways of incorporating 
inductive biases into the architetures, motivated by graph spectral theory, previous work on positional encodings, and the GNN/Graph Transforme
literature.

We will work to overcome challenges such as how to integrate a new module trained from scratch with a pretrained language model, 
how to properly measure and study the benefits of a graph-enhanced planning architecture, and the techincal aspects that come
with scaling the graphs. 


## Cluster (SLURM) config for this project

Used to fill in placeholders in `~/.claude/skills/cluster-slurm/template.sbatch`:

- `PROJECT`: `GREP-PRISM`
- `ENTITY`: `alelab` (wandb)
- `ENV_NAME`: `/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3`
- `OUTPUT_BASE` (artifact root — checkpoints, outputs, logs): `$ALELAB_DRIVE/GREP-PRISM`, where `$ALELAB_DRIVE=/vast/projects/aribeiro/alelab/jporras`
- SLURM output dir: `$ALELAB_DRIVE/GREP-PRISM/slurm-%A_%a.out`
- Repo path on cluster (code only): `~/sourcecode/GREP` (`/vast/home/j/jporras/sourcecode/GREP`). Code reaches the cluster via `git pull`, never rsync; artifacts live under `OUTPUT_BASE`, never in the repo.
- Default partition: `dgx-b200`
- Entry: `python -m prism.training.train_v3 --config-name=<config> [key=value ...]` (nested Hydra configs under `experiments/`; CLI `key=value` overrides supported)


## Project Structure & Module Organization
- `src/prism/` houses the installable package; subfolders map to simulation assets, evaluation logic, planners, and LoRA tooling.
- `scripts/` contains runnable entry points for data generation, aggregation, training, and evaluation.
- `data/` stores seed eval graphs and generated corpora; keep new experiments under `data/eval/` or sibling folders.

## Environment Setup
- `requirements.txt` is the source of truth for dependencies.
- Clone the PRISM branch of SPINE for planner interoperability:
  ```bash
  git clone git@github.com:KumarRobotics/SPINE.git
  cd SPINE
  git checkout feature/prism
  git pull
  ```
  Ensure SPINE is importable (editable install or `PYTHONPATH`).

## Build, Test, and Development Commands
- `python -m pip install -r requirements.txt` installs the dependency set (`requirements.txt` is the source of truth).
- `python -m pip install -e .` enables `prism.*` imports in scripts and notebooks.
- `python scripts/training_data_generation/generate_data_spine.py --n-samples 10 --n-tasks 3 --name demo` creates SPINE training data; other generators share similar flags.
- `python -m prism.training.train_v3 --config-name=<config> [key=value ...]` launches training (LoRA SFT) from a nested Hydra config under `experiments/`.
- Evaluation runs as part of `train_v3` via the Hydra `eval.*` block: set `eval.post_train_graphs` to reload the saved checkpoint from disk and cross-evaluate the held-out set.

## Coding Style & Naming Conventions
- Use Black/Isort/Flake8 (see README); run `pre-commit install`, then `pre-commit run --all-files` before committing.
- Stick to 4-space indentation, snake_case modules/functions, PascalCase classes, and rich type hints.
- Keep configuration constants uppercase and load secrets from environment variables, not committed files.
- Avoid defaults in Python class `__init__`, prefer defaults in argparse arguments instead.
- Imports: always `from pkg import mod; mod.name`, never `from pkg.mod import name`.

## Testing Guidelines
- Tests live under `tests/`; run with `conda run -n GREP-PRISM-v3 python -m pytest tests/ -v`.
- Existing suites: `test_scene_graph_parser.py`, `test_sim.py`, `test_bucketize_prompt.py`, `test_remove_edge_list.py`.
- `test_sim.py` covers `GraphSim.take_action` and SPINE plan parsing; uses an inline `_DummyClient` to avoid LLM calls.
- Keep repro artifacts small; expand `data/eval/` fixtures for new eval scenarios.

## Verification 
Before I trust a result, it must pass the project's test oracle / invariant
checks/red-teaming. Usually you will have skills available for this purpose. If they are missing, offer me to create them.

General principles for verification:
- **Never verify your own work with the same reasoning that produced it.** Use a
  fresh pass, a separate sub-agent, or a deterministic check — not self-review.
- When you finish a unit of work, end with what you're *unsure about* and what I
  should check. That field is where I'll spend my attention.

## Commit & Pull Request Guidelines
- Keep commit subjects short, Title Case (e.g., `Update README.md`), and isolate unrelated edits.
- Never `git add -A` (it sweeps untracked scratch/data junk); use `git add -u` plus explicit new files.
- In PRs, state the goal, link issues or experiment logs, and paste relevant command outputs (train/eval metrics).
- Run the necessary data, training, and eval steps before review; call out anything you could not validate.
- Tag reviewers closest to the touched modules and highlight breaking changes or new dependencies early.

## Package Management
- NEVER install, upgrade, or remove packages without explicitly telling the user first and getting approval.
- The full working conda env is `GREP-PRISM-v3`. Always activate it before running Python.

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

