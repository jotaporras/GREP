#!/usr/bin/env bash
# Generate e4 v4 training data: 50 skeletons -> populated graphs + SPINE rollouts -> train/val split.
#
# Skeletons: 5 (n_communities, nodes_per_community) configs x 10 seeds = 50.
# Skeleton filenames embed the seed so provenance is recoverable from filenames.
#
# Phase 1 (in generate_data_spine.py) uses gpt-5.x with reasoning=high.
# Each task carries an `acceptance_criterion` field for offline grading ONLY —
# it is never shown to the planner, and `src/prism/data/data_gen.py` asserts
# this invariant explicitly at the planner call site.
#
# After this script finishes, open Claude Code and run:
#   "audit the e4 v4 run at <RUN_DIR>"
# to run /verify-tasks on 30 graphs and Sonnet/Haiku judging on 15 rollouts.
#
# To incrementally extend an existing 5-graph v4 run (without re-paying for
# graphs 0..4), use scripts/e4_extend_training_data.sh instead.

set -euo pipefail

# Auto-load .env from the repo root (or wherever this script lives) so
# OPENAI_API_KEY and friends are present when invoked from a plain bash shell.
# `set -a` auto-exports every var assigned by `source`.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

SKEL_DIR="data/training/training_v4_skeletons"
RUN_DIR="data/gen/e4_training_data_v4"
SPLIT_DIR="${RUN_DIR}/split"

INTRA_PROB=0.6
INTER_PROB=0.05
OBJECT_RATE=0.3
DESC_PROB=0.05
N_TASKS=10

# (N, n_communities, nodes_per_community) — matches generate_training_graphs.sh sizing.
configs=(
  "30  3  8"
  "35  3 10"
  "40  4  9"
  "45  4 10"
  "50  4 11"
)

# 10 seeds × 5 configs = 50 skeletons. Seed 42 matches the original single-seed
# run; 101–109 match the seed range used by e4_extend_training_data.sh, so a
# fresh run here produces the same (config, seed) coverage as extend.
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
echo "=== Stage 2: Populate (gpt-5.x reasoning=high) + run SPINE rollouts ==="
python scripts/training_data_generation/generate_data_spine.py \
  --data-dir "$SKEL_DIR" \
  --name "$RUN_DIR"

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
echo "  audit the e4 v4 run at ${RUN_DIR}"
echo ""
echo "(This runs /verify-tasks on 30 graphs + Sonnet/Haiku judging on 15"
echo " rollouts, and writes ${RUN_DIR}/audit_report.md.)"
