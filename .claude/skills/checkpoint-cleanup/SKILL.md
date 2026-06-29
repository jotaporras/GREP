---
name: checkpoint-cleanup
description: Use to free disk space by deleting training checkpoint weights while keeping configs and eval_logs. Two entry points — (A) FOLDER-level: clean whole experiment folders the user names (e.g. "clean e9_gemma31b and e3", "betty cleanup", "free space under outputs/"); (B) RUN-level: keep specific run IDs within one folder and clean the rest ("cleanup checkpoints but keep run X and Y"). Trigger on "cleanup checkpoints", "delete old checkpoints", "betty cleanup", /checkpoint-cleanup, or /betty-cleanup.
---

# Checkpoint Cleanup

Delete the multi-GB checkpoint **weights** from training runs while always preserving configs
and `eval_logs/` (the actual results). Two entry points share the same structure knowledge and
the same dry-run → delete → verify discipline:

- **(A) Folder-level** — the user names whole experiment folders (`outputs/<exp>`); clean every
  run in them. ("clean e9_gemma31b and e3", "betty cleanup")
- **(B) Run-level** — the user names specific run IDs to **keep** within one folder; clean all
  the others. ("cleanup checkpoints but keep zqwguufs and 5pz198i0")

Decide which from how the user phrases it; if ambiguous, ask.

## Where outputs live

On the **betty cluster**, runs live under the shared artifact root, NOT the code repo:

```
$ALELAB_DRIVE/GREP-PRISM/outputs/<experiment>/
  = /vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs/<experiment>/
```

Expand an experiment name (e.g. "e9_gemma31b") to that full path. `du -sh
$ALELAB_DRIVE/GREP-PRISM/outputs/**` lists candidates + sizes. Local workstation runs live
under the repo's `outputs/` instead.

## Run-dir structure (the key thing this skill remembers)

```
outputs/<exp>/<run_dir>/                 # <run_dir> = <save_name>_<wandb_id>
  adapter_model.safetensors              # FINAL LoRA weights  ── reloadable
  gnn_weights.pt                         # FINAL PE/GNN weights ── reloadable
  adapter_config.json, gnn_config.json   # keep (tiny, describe the run)
  trainer_state.json, training_args.bin  # keep (metadata)
  tokenizer*, chat_template.jinja, README.md, *.txt   # keep
  eval_logs/                             # KEEP — the results (judge md + step *.json)
  checkpoint-NNN/                        # intermediate HF resume snapshots ── THE BULK
      adapter_model.safetensors, gnn_weights.pt,
      optimizer.pt, scheduler.pt, rng_state.pth, training_args.bin
```

Why it matters:
- **`checkpoint-*/` dirs are nearly all the bytes** — each carries an `optimizer.pt` ≈ 2× the
  trainable params. Deleting these alone recovers most of the space.
- **Top-level `adapter_model.safetensors` + `gnn_weights.pt` are the final, reloadable
  weights.** Keeping them lets the run still be reloaded for eval (the disk-reload eval path);
  deleting them makes the run results-only.
- **`eval_logs/`, configs, and metadata are tiny and precious — never delete them.**

## Two delete modes — pick by context, ask if unsure

| Mode | Deletes | Leaves | Use when |
|---|---|---|---|
| **Keep-reloadable** (recommended for active experiments / kept runs) | only `checkpoint-*/` dirs | final top-level weights + configs + eval_logs | run may still be reloaded/re-evaluated |
| **Full wipe** | `checkpoint-*/` dirs **and** top-level `*.safetensors` + `gnn_weights.pt` | configs + eval_logs only | experiment/run is done; keep results, not weights |

Modes differ only by whether step 2 below runs. It is the irreversible line that makes a run
non-reloadable — call it out before the user proceeds.

## Procedure

1. **Resolve scope.**
   - (A) Folder-level: the named experiment dirs → `DIRS=( … )`.
   - (B) Run-level: list run dirs under the one folder, classify each KEEP vs CLEAN against the
     user's keep-list. KEPT runs → keep-reloadable mode (optionally keep only the oldest
     `checkpoint-*`). CLEAN runs → full wipe.
2. **Dry run** — show exactly what goes and how much it frees, organized by run/folder.
3. **Confirm** — present the summary (table/list), name any *active* experiment in scope, and
   wait for explicit approval.
4. **Delete** — see command block.
5. **Verify** — sizes shrank and every `eval_logs/` survives.

## Command block

```bash
DIRS=(
  $ALELAB_DRIVE/GREP-PRISM/outputs/<exp1>
  $ALELAB_DRIVE/GREP-PRISM/outputs/<exp2>
)

# DRY RUN — what would go + space freed
find "${DIRS[@]}" -type d -name 'checkpoint-*' -printf '%p\n' | sort
find "${DIRS[@]}" -type f \( -name '*.safetensors' -o -name 'gnn_weights.pt' \) \
     -printf '%s\t%p\n' | awk -F'\t' '{s+=$1} END{printf "%d weight files, %.2f GB\n", NR, s/1073741824}'

# 1. intermediate resume checkpoints (the bulk) — BOTH modes
find "${DIRS[@]}" -type d -name 'checkpoint-*' -exec rm -rf {} +

# 2. FULL-WIPE ONLY — final top-level weights (depth 2 = <run_dir>/<file>)
find "${DIRS[@]}" -mindepth 2 -maxdepth 2 -type f \
     \( -name '*.safetensors' -o -name 'gnn_weights.pt' \) -delete

# VERIFY
du -sh "${DIRS[@]}"
find "${DIRS[@]}" -path '*/eval_logs/*' -name '*.json' | wc -l   # must be > 0
```

Keep-reloadable mode runs DRY-RUN → 1 → VERIFY and omits step 2. For run-level (B), scope the
`find` to the specific CLEAN run dirs (or pass them in `DIRS=`) and run step 1 only on KEPT runs.

Run-level confirmation example:
```
KEPT (resume snapshots removed, final weights kept):
  zqwguufs: rm checkpoint-892/
NON-KEPT (all weights removed):
  5pz198i0: rm checkpoint-*/, adapter_model.safetensors, gnn_weights.pt
  abcd1234: (no checkpoints / no root weights — nothing to remove)
```

## Cluster vs local — who runs it

- **Betty / cluster paths** (`$ALELAB_DRIVE/...`): I do NOT execute these. I hand over the
  copy-paste block and the user runs it on betty (per the don't-run-cluster-unprompted rule).
  Before they paste, surface scope (full paths), rough space freed, that step 2 is irreversible,
  and any active experiment in scope.
- **Local workstation paths**: I may execute via Bash, but only after showing the dry-run
  summary and getting explicit confirmation.

## Restrictions

- Never delete `eval_logs/`, any `*config*.json`, `trainer_state.json`, or a whole run/experiment dir.
- The deletion globs must not match `eval_logs/*.json` (they don't — those aren't `.safetensors`/`.pt`).
- Always verify after deletion by listing remaining checkpoints/weights and confirming eval_logs survive.
