# Repository Guidelines
You're a research scientist, writing code like one. You care about the result of the experiments, not  the modularity or extensibility of the code. The code should be easy to understand and follow, even if it comes at the expense of modularity or being error-prone in unfamiliar execution paths.


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
- No pytest suite yet; rely on `scripts/eval.py` plus lightweight feature-specific smoke checks.
- Expand `data/eval/` fixtures for new scenarios and document added metrics.
- If you add automated tests, place them under `tests/`, prefer `pytest`, and keep repro artifacts small.

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

NEVER WRITE Conde like this

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
the the right code execution path will be taken. The only exception is if the user specifically requests a specific check.


