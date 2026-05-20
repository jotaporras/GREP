#!/usr/bin/env bash
# Recover the e5 graph-oriented dataset after a partial / budget-exhausted run.
#
# State this repairs: Phase 1 (LLM populate) fully succeeded — all 50 graphs
# are in populated_graphs/ — but Phase 2 (SPINE rollouts) only partly ran:
# some rollouts are corrupt, ~240 failed on OpenAI insufficient_quota, and
# graphs 35-49 never ran at all.
#
# Stages:
#   1. Quarantine every failed (*_failed.json) and corrupt (unparseable
#      sample_*.json) rollout into trash/recovery_<timestamp>/.
#   2. Re-run Phase 2 over the 50 already-populated graphs. generate_example_plans
#      skips any task that already has a valid rollout, so only the missing /
#      failed / corrupt tasks are regenerated — the clean ones are kept.
#   3. Re-run the train/val split.
#
# No renumbering step: populated_graphs/ is already contiguous data_gen_0..49,
# so Phase 2 numbers sample_000..049 contiguously by construction.
#
# Requirements / caveats:
#   - OPENAI_API_KEY must have available quota. The SPINE planner is OpenAI-
#     backed; if the account is out of budget the rerun will not crash — it
#     will just write *_failed.json files again. Top up billing first.
#   - The LLMDataLogger r+ bug is NOT fixed here, so a fraction of the
#     regenerated rollouts will again come out corrupt. This script is safe to
#     re-run: each pass keeps the now-valid rollouts (trashing the rest) and
#     retries only what is still missing, so repeated runs converge.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Auto-load .env so OPENAI_API_KEY is present when invoked from a plain shell.
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

RUN_DIR="data/gen/e5_graph_oriented_data"
PLANS_DIR="${RUN_DIR}/generated_plans"
GRAPHS_DIR="${RUN_DIR}/populated_graphs"
SPLIT_DIR="${RUN_DIR}/split"
TRASH_DIR="${RUN_DIR}/trash/recovery_$(date +%Y%m%d_%H%M%S)"

N_TASKS=10
GEN_SEED=42

if [ ! -d "$GRAPHS_DIR" ]; then
  echo "ERROR: ${GRAPHS_DIR} not found — Phase 1 populate has not run."
  exit 1
fi

echo "=== Stage 1: Quarantine failed + corrupt rollouts ==="
mkdir -p "$TRASH_DIR"
# Preserve the pre-recovery run params — Stage 2 overwrites data_gen_params.json.
[ -f "${RUN_DIR}/data_gen_params.json" ] && \
  cp "${RUN_DIR}/data_gen_params.json" "${TRASH_DIR}/data_gen_params.pre_recovery.json"

python - "$PLANS_DIR" "$TRASH_DIR" <<'PYEOF'
import re
import shutil
import sys
from pathlib import Path

# Single source of truth: the same validity test generate_example_plans uses
# to decide what to skip, so quarantined == not-skipped exactly.
from prism.data.data_gen import DataGenerator

plans_dir, trash_dir = Path(sys.argv[1]), Path(sys.argv[2])
sample_re = re.compile(r"^sample_\d+_\d+\.json$")

kept = moved_failed = moved_corrupt = 0
for p in sorted(plans_dir.iterdir()) if plans_dir.is_dir() else []:
    if not p.is_file():
        continue
    if p.name.endswith("_failed.json"):
        shutil.move(str(p), str(trash_dir / p.name))
        moved_failed += 1
    elif sample_re.match(p.name):
        if DataGenerator._has_valid_rollout(str(p)):
            kept += 1
        else:
            shutil.move(str(p), str(trash_dir / p.name))
            moved_corrupt += 1
    # other files (e.g. a stale formatted.json) are left as-is; Phase 2's
    # aggregate step overwrites formatted.json anyway.

print(f"  kept     {kept} valid rollouts")
print(f"  trashed  {moved_failed} *_failed.json + {moved_corrupt} corrupt sample_*.json")
print(f"  trash -> {trash_dir}")
PYEOF

echo
echo "=== Stage 2: Re-run Phase 2 (SPINE rollouts) on the 50 populated graphs ==="
echo "    skip-clean is on — only missing / failed / corrupt tasks regenerate"
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "ERROR: OPENAI_API_KEY is not set (check ${REPO_ROOT}/.env)."
  exit 1
fi
python scripts/training_data_generation/generate_data_spine.py \
  --skip-populate \
  --data-dir "$GRAPHS_DIR" \
  --name "$RUN_DIR" \
  --n-tasks "$N_TASKS" \
  --seed "$GEN_SEED"

echo
echo "=== Stage 3: Train/val split ==="
python scripts/training_data_generation/split_train_val.py \
  --plans-dir "$PLANS_DIR" \
  --graphs-dir "$GRAPHS_DIR" \
  --output-dir "$SPLIT_DIR"

echo
echo "=== Done. ==="
echo "If split reports fewer than 500 clean rollouts, re-run this script to"
echo "converge — it will keep what is now valid and retry only the rest."
