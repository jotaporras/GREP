#!/usr/bin/env bash
# Extend the existing e4 v4 run (5 graphs) to 50 graphs WITHOUT regenerating
# the original 5. The original populated_graphs/data_gen_000..004.json and
# their rollouts (sample_000_*.json..sample_004_*.json) stay byte-identical;
# 45 new graphs are appended at indices 005..049.
#
# Stages:
#   1. Generate 45 new skeletons into data/training/training_v4b_skeletons/
#   2. Run generate_data_spine.py into a TEMP run dir data/gen/e4_training_data_v4b/
#   3. Merge v4b -> v4 with renumbering (data_gen_*, graph_gen_*, sample_GGG_*)
#   4. Rebuild data/gen/e4_training_data_v4/generated_plans/formatted.json over
#      the merged set
#   5. Re-run split_train_val on the merged set (replaces existing split/)
#   6. Leak check on the merged tree (halts on LEAK)
#   7. Print follow-up instructions

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

# Existing v4 run (must already contain populated_graphs/data_gen_000..004.json
# and the matching rollouts under generated_plans/).
RUN_DIR="data/gen/e4_training_data_v4"

# New skeletons + temp run dir for the 45 additions.
NEW_SKEL_DIR="data/training/training_v4b_skeletons"
TMP_RUN_DIR="data/gen/e4_training_data_v4b"

INTRA_PROB=0.6
INTER_PROB=0.05
OBJECT_RATE=0.3
DESC_PROB=0.05
N_TASKS=10

# Same five configs as e4_generate_training_data_v4.sh.
configs=(
  "30  3  8"
  "35  3 10"
  "40  4  9"
  "45  4 10"
  "50  4 11"
)

# 9 new seeds per config (seed 42 already produced graphs 000..004 in the
# original v4 run). 5 configs x 9 seeds = 45 new skeletons.
seeds=(101 102 103 104 105 106 107 108 109)

# ---------- Safety preconditions ----------
N_EXISTING=$(ls "${RUN_DIR}/populated_graphs/data_gen_"*.json 2>/dev/null | wc -l || true)
if [ "$N_EXISTING" != "5" ]; then
  echo "ERROR: expected exactly 5 existing populated graphs in ${RUN_DIR}/populated_graphs/, found ${N_EXISTING}."
  echo "       This script is designed to extend the original 5-graph v4 run."
  echo "       If you want to start over, use scripts/e4_generate_training_data_v4.sh."
  exit 2
fi

if [ -d "${TMP_RUN_DIR}" ]; then
  echo "ERROR: temp run dir ${TMP_RUN_DIR} already exists."
  echo "       Inspect/remove it before re-running this script."
  exit 2
fi

# ---------- Stage 1: 45 new skeletons ----------
mkdir -p "$NEW_SKEL_DIR"

echo "=== Stage 1: Generate 45 new skeletons (${#configs[@]} configs x ${#seeds[@]} seeds) ==="
for cfg in "${configs[@]}"; do
  read -r N NC NPC <<< "$cfg"
  for SEED in "${seeds[@]}"; do
    echo "--- N=${N} (${NC}x${NPC}) seed=${SEED} ---"
    python scripts/generate_eval_graphs.py \
      --n-communities "$NC" \
      --nodes-per-community "$NPC" \
      --intra-community-prob "$INTRA_PROB" \
      --inter-community-prob "$INTER_PROB" \
      --object-rate "$OBJECT_RATE" \
      --description-prob "$DESC_PROB" \
      --n-tasks "$N_TASKS" \
      --seed "$SEED" \
      --output "${NEW_SKEL_DIR}/eval_graph_unique_${N}_seed${SEED}.json"
  done
done

N_NEW_SKEL=$(ls "${NEW_SKEL_DIR}"/eval_graph_unique_*.json | wc -l)
if [ "$N_NEW_SKEL" != "45" ]; then
  echo "ERROR: expected 45 new skeletons, found ${N_NEW_SKEL}"
  exit 2
fi

# ---------- Stage 2: populate + run SPINE rollouts on the new skeletons ----------
echo ""
echo "=== Stage 2: Populate + rollouts (gpt-5.x reasoning=high) on 45 new skeletons ==="
python scripts/training_data_generation/generate_data_spine.py \
  --data-dir "$NEW_SKEL_DIR" \
  --name "$TMP_RUN_DIR"

# ---------- Stage 3: merge v4b -> v4 with renumbering (+5) ----------
echo ""
echo "=== Stage 3: Merge v4b into v4 with index shift +${N_EXISTING} ==="

# populated_graphs/data_gen_*.json
for src in "${TMP_RUN_DIR}/populated_graphs/data_gen_"*.json; do
  base=$(basename "$src" .json)
  i=${base#data_gen_}
  new_idx=$(printf "%03d" $((10#$i + N_EXISTING)))
  mv "$src" "${RUN_DIR}/populated_graphs/data_gen_${new_idx}.json"
done

# populated_graphs/graph_gen_*.json
for src in "${TMP_RUN_DIR}/populated_graphs/graph_gen_"*.json; do
  base=$(basename "$src" .json)
  i=${base#graph_gen_}
  new_idx=$(printf "%03d" $((10#$i + N_EXISTING)))
  mv "$src" "${RUN_DIR}/populated_graphs/graph_gen_${new_idx}.json"
done

# generated_plans/sample_GGG_TTT.json (and _failed variants)
# Format: sample_<3-digit graph idx>_<3-digit task idx>[_failed].json
for src in "${TMP_RUN_DIR}/generated_plans/sample_"*.json; do
  base=$(basename "$src" .json)
  # base = sample_GGG_TTT  OR  sample_GGG_TTT_failed
  ggg=$(echo "$base" | awk -F'_' '{print $2}')
  rest=$(echo "$base" | sed -E "s/^sample_${ggg}_//")   # captures TTT or TTT_failed
  new_ggg=$(printf "%03d" $((10#$ggg + N_EXISTING)))
  mv "$src" "${RUN_DIR}/generated_plans/sample_${new_ggg}_${rest}.json"
done

# Discard stale temp formatted.json (we'll rebuild from the merged set)
rm -f "${TMP_RUN_DIR}/generated_plans/formatted.json"

# v4b temp dir should now be empty of useful artifacts. Sanity-clean it.
rmdir "${TMP_RUN_DIR}/populated_graphs" 2>/dev/null || true
rmdir "${TMP_RUN_DIR}/generated_plans"  2>/dev/null || true
# Keep data_gen_params.json from v4b for traceability — move it next to its temp dir parent.
if [ -f "${TMP_RUN_DIR}/data_gen_params.json" ]; then
  mv "${TMP_RUN_DIR}/data_gen_params.json" "${RUN_DIR}/data_gen_params_extend.json"
fi
rmdir "${TMP_RUN_DIR}" 2>/dev/null || true

# ---------- Stage 4: rebuild formatted.json over the merged rollouts ----------
echo ""
echo "=== Stage 4: Rebuild formatted.json over merged rollouts ==="
PYTHONPATH=src python -c "
from prism.data.utils import aggregate
aggregate(
    root_dir='${RUN_DIR}/generated_plans',
    glob_str='sample*json',
    out_file='${RUN_DIR}/generated_plans/formatted.json',
)
print('aggregate done')
"

# ---------- Stage 5: re-run split_train_val on the merged set ----------
echo ""
echo "=== Stage 5: Re-run train/val split over 50 graphs ==="
rm -rf "${RUN_DIR}/split"
python scripts/training_data_generation/split_train_val.py \
  --plans-dir "${RUN_DIR}/generated_plans" \
  --graphs-dir "${RUN_DIR}/populated_graphs" \
  --output-dir "${RUN_DIR}/split"

# ---------- Stage 6: leak check + count assertion ----------
echo ""
echo "=== Stage 6: Leak check on merged set ==="

if grep -RlE 'acceptance_criterion|"answer"[[:space:]]*:' "${RUN_DIR}/generated_plans/" >/dev/null 2>&1; then
  echo "LEAK in generated_plans/ — DO NOT TRAIN on this data"
  exit 2
fi
echo "CLEAN: generated_plans/"

if grep -E 'acceptance_criterion|"answer"[[:space:]]*:' "${RUN_DIR}/generated_plans/formatted.json" >/dev/null 2>&1; then
  echo "LEAK in formatted.json — DO NOT TRAIN on this data"
  exit 2
fi
echo "CLEAN: formatted.json"

N_POPULATED=$(ls "${RUN_DIR}/populated_graphs/data_gen_"*.json | wc -l)
if [ "$N_POPULATED" != "50" ]; then
  echo "ERROR: expected 50 populated graphs after merge, found ${N_POPULATED}"
  exit 2
fi
echo "OK: 50 populated graphs"

# ---------- Stage 7: follow-up instructions ----------
echo ""
echo "=== Extend complete. NEXT — open Claude Code and run: ==="
echo "  audit the e4 v4 run at ${RUN_DIR}"
echo ""
echo "(This runs /verify-tasks data/gen/e4_training_data_v4/populated_graphs --sample-size 30"
echo " + Sonnet/Haiku judging on 15 rollouts, and overwrites ${RUN_DIR}/audit_report.md.)"
