# Pre-Commit Reviewer Memory

## Project Style Rules (from AGENTS.md)
- No defensive programming: no `try/except ImportError`, no `if x is not None` guards, no silent fallback branches
- No `try/finally` blocks unless explicitly requested
- Straight-line code; let it fail loudly on unexpected paths
- Module-level imports using `from prism.X import Y` then `Y.func()` style
- 4-space indent, snake_case modules/functions, PascalCase classes, rich type hints
- No defaults in `__init__`, prefer argparse defaults instead

## Known Architecture (confirmed)
- `GraphAugmentedLLM` in `src/prism/models/gnn_llm.py` — domain-agnostic wrapper
- `build_injection_map` / `bucketize_prompt` live in `gnn_llm.py` (NOT in data_col)
- `bucketize_prompt` returns `defaultdict(set)` — so `starts` in `build_injection_map` is a set of ints
- `SpineDataCollator` extends `DataCollatorForGraphAugmentedLLM` in `data_col.py`
- `loaders.py` has `from_pretrained` and `graph_augmented_llm_from_pretrained`
- `inference.py` has `InMemoryLLM` and `GraphAugmentedInMemoryLLM`

## Recurring Issues Found (2026-03-02 review)
- `tokenizer.encode(list_of_strings)` does NOT return `list[list[int]]` — it returns a flat
  `list[int]` when given a list; callers in `data_col.py` and `inference.py` rely on this
  returning per-node sequences. This is a latent semantic bug to watch for.
- `super()._generate(...)` called in `inference.py:78` but parent `InMemoryLLM` only defines
  `_generate_tokens`, not `_generate` — this is a dead-code/wrong-name bug triggered when
  `pyg_graph is None`.
- `run_eval.py:70` passes `inference=True` to `loaders.from_pretrained` which does not accept
  that parameter — pre-existing bug, not introduced by this refactor.
- Silent exception swallowing (`except Exception as e: print(...)`) in `data_col.py:80` violates
  the no-silent-errors policy. Pre-existing pattern inherited from base collator.
- `if x is not None` guards appear in `inference.py:74` (pyg_graph None check) — also violates
  AGENTS.md policy but is pre-existing.

## Key File Locations
- Training entry point: `src/prism/training/train_v2.py`
- Eval entry point: `src/prism/eval/run_eval.py`
- Data collation: `src/prism/data/data_col.py`
- Graph utilities: `src/prism/data/utils.py` (scene_graph_dict_to_pyg)
- Model loaders: `src/prism/models/loaders.py`
- Callbacks: `src/prism/eval/callbacks.py`
