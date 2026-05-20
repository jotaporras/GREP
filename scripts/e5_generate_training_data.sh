#!/usr/bin/env bash
# Generate e5 graph-oriented training data: 50 skeletons -> populated graphs +
# SPINE rollouts -> train/val split.
#
# e5 biases the task mix toward graph navigation and away from single-node
# lookups via --task-proportions 0.125 0.125 0.15 0.6
#   (12.5% Existence, 12.5% Positionality, 15% Reachability, 60% Navigability).
#
# Skeletons: 5 (n_communities, nodes_per_community) configs x 10 seeds = 50.
# Skeleton filenames embed the seed so provenance is recoverable from filenames.
#
# Run scripts/e5_generate_single_example.sh FIRST to spot-check task/rollout
# quality before paying for this full run.
#
# After this script finishes, open Claude Code and run:
#   "audit the e5 run at <RUN_DIR>"

set -euo pipefail

# Auto-load .env from the repo root so OPENAI_API_KEY and friends are present
# when invoked from a plain bash shell. `set -a` auto-exports every sourced var.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

SKEL_DIR="data/training/e5_graph_oriented_skeletons"
RUN_DIR="data/gen/e5_graph_oriented_data"
SPLIT_DIR="${RUN_DIR}/split"

INTRA_PROB=0.6
INTER_PROB=0.05
OBJECT_RATE=0.3
DESC_PROB=0.05
N_TASKS=10

# Navigation-heavy task mix (EXIST, POS, REACH, NAV). GEN_SEED makes the
# task-type multinomial sampling reproducible.
TASK_PROPORTIONS="0.125 0.125 0.15 0.6"
GEN_SEED=42

# (N, n_communities, nodes_per_community). N is the nominal size label (e4
# convention); regions = n_communities x nodes_per_community.
configs=(
  "25  3  6"
  "30  3  8"
  "35  3 10"
  "40  4  9"
  "45  4 10"
)

# 10 seeds x 5 configs = 50 skeletons.
seeds=(42 101 102 103 104 105 106 107 108 109)

mkdir -p "$SKEL_DIR"

echo "=== Stage 1: Generate skeletons (${#configs[@]} configs x ${#seeds[@]} seeds = $((${#configs[@]} * ${#seeds[@]})) skeletons) ==="
for cfg in "${configs[@]}"; do
  read -r N NC NPC <<< "$cfg"
  for SEED in "${seeds[@]}"; do
    echo "--- N=${N} (${NC} communities x ${NPC} nodes/community) seed=${SEED} ---"
    python scripts/generate_eval_graphs.py \
      --n-communities "$NC" \
      --nodes-per-community "$NPC" \
      --intra-community-prob "$INTRA_PROB" \
      --inter-community-prob "$INTER_PROB" \
      --object-rate "$OBJECT_RATE" \
      --description-prob "$DESC_PROB" \
      --n-tasks "$N_TASKS" \
      --seed "$SEED" \
      --output "${SKEL_DIR}/eval_graph_unique_${N}_seed${SEED}.json"
  done
done

echo ""
echo "=== Stage 2: Populate (gpt-5.1 reasoning=low) + run SPINE rollouts ==="
echo "    task mix: ${TASK_PROPORTIONS} (EXIST POS REACH NAV)"
# shellcheck disable=SC2086
python scripts/training_data_generation/generate_data_spine.py \
  --data-dir "$SKEL_DIR" \
  --name "$RUN_DIR" \
  --task-proportions $TASK_PROPORTIONS \
  --n-tasks "$N_TASKS" \
  --seed "$GEN_SEED"

echo ""
echo "=== Stage 3: Train/val split ==="
python scripts/training_data_generation/split_train_val.py \
  --plans-dir "${RUN_DIR}/generated_plans" \
  --graphs-dir "${RUN_DIR}/populated_graphs" \
  --output-dir "$SPLIT_DIR"

echo ""
echo "=== Stage 4: Leak check (must print CLEAN twice) ==="
# Match either the acceptance_criterion field name or the "answer" regex field
# anywhere under generated_plans/. Both should be confined to populated_graphs/.
if grep -RlE 'acceptance_criterion|"answer"[[:space:]]*:' "${RUN_DIR}/generated_plans/" >/dev/null 2>&1; then
  echo "LEAK in generated_plans/ — investigate before training on this data"
  exit 2
else
  echo "CLEAN: generated_plans/"
fi

if [ -f "${RUN_DIR}/generated_plans/formatted.json" ]; then
  if grep -E 'acceptance_criterion|"answer"[[:space:]]*:' "${RUN_DIR}/generated_plans/formatted.json" >/dev/null 2>&1; then
    echo "LEAK in formatted.json — investigate before training on this data"
    exit 2
  else
    echo "CLEAN: formatted.json"
  fi
else
  echo "WARN: ${RUN_DIR}/generated_plans/formatted.json does not exist (aggregate may have skipped)"
fi

echo ""
echo "=== Stage 5: NEXT — open Claude Code and run: ==="
echo "  audit the e5 run at ${RUN_DIR}"
