---
name: checkpoint-cleanup
description: Use when the user wants to clean up checkpoint weight files from training runs, keeping only specified runs intact. Trigger on phrases like "cleanup checkpoints", "rm checkpoint weights", "delete old checkpoints", or when the user says /checkpoint-cleanup.
---

# Checkpoint Cleanup

The user provides a list of run IDs (or full directory names) to **keep**. All other runs in the same output directory have their weight files deleted. Configs and eval logs are never deleted.

## Behavior

1. **Discover** all run directories under the relevant output path (ask user if unclear).
2. **Classify** each run as KEEP or CLEAN based on the user's list.
3. **For KEPT runs**: delete only the newer checkpoint-* subdirs, keeping the single oldest one. Show what will be removed.
4. **For NON-KEPT runs**: delete all `checkpoint-*/` subdirs and root-level weight files (`*.safetensors`, `*.pt` files). Keep all config files and `eval_logs/`.
5. **Always confirm** before deleting. Present a clear summary of what will be removed, organized by run. Wait for explicit user approval.
6. Execute the deletions using `rm -rf` via Bash.
7. Verify the final state by listing remaining checkpoints and weight files for all runs.

## Weight files to delete

- `adapter_model.safetensors`
- `gnn_weights.pt`
- `checkpoint-*/` directories (contain `adapter_model.safetensors`, `optimizer.pt`, `rng_state.pth`, `scheduler.pt`)

## Files to always keep (never delete)

- `adapter_config.json`
- `gnn_config.json`
- `tokenizer_config.json`, `tokenizer.json`
- `chat_template.jinja`
- `training_args.bin`
- `trainer_state.json`
- `README.md`
- `eval_logs/` (entire directory)
- Any `.txt` output files

## Confirmation format

Before executing, show the user a table or structured list like:

```
KEPT (oldest checkpoint only):
  run_id_A: rm checkpoint-892/
  run_id_B: rm checkpoint-1700/ checkpoint-1782/

NON-KEPT (all weights removed):
  run_id_C: rm checkpoint-*/, adapter_model.safetensors, gnn_weights.pt
  run_id_D: rm adapter_model.safetensors
  run_id_E: (no checkpoints, no root weights — nothing to remove)
```

Only proceed after the user confirms.

## IMPORTANT RESTRICTIONS

- Never delete `eval_logs/` or any config files.
- Never delete the entire run directory — only weight files within it.
- Always verify the result after deletion by listing remaining checkpoints per run.
