#!/usr/bin/env bash
# e5 single-example probe: 1 graph, 20 tasks, navigation-heavy task mix.
#
# Run this BEFORE the full scripts/e5_generate_training_data.sh. It generates a
# single populated graph with 20 tasks and their SPINE rollouts so task and
# rollout quality can be spot-checked before paying for the full 50-graph run.
#
# Task mix: --task-proportions 0.125 0.125 0.15 0.6
#   (12.5% Existence, 12.5% Positionality, 15% Reachability, 60% Navigability)
#
# No train/val split is produced here — split_train_val.py needs >= 2 graphs.

set -euo pipefail

# Auto-load .env from the repo root so OPENAI_API_KEY is present when invoked
# from a plain bash shell. `set -a` auto-exports every var assigned by `source`.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

SKEL_DIR="data/training/e5_single_example_skeleton"
RUN_DIR="data/gen/e5_single_example"

N_TASKS=20
TASK_PROPORTIONS="0.125 0.125 0.15 0.6"
SEED=42

# One mid-band graph — same sizing as the "40 4 9" config in the full run.
NC=4
NPC=9

mkdir -p "$SKEL_DIR"

echo "=== Stage 1: Generate one skeleton (${NC} communities x ${NPC} nodes/community, ${N_TASKS} tasks) ==="
python scripts/generate_eval_graphs.py \
  --n-communities "$NC" \
  --nodes-per-community "$NPC" \
  --intra-community-prob 0.6 \
  --inter-community-prob 0.05 \
  --object-rate 0.3 \
  --description-prob 0.05 \
  --n-tasks "$N_TASKS" \
  --seed "$SEED" \
  --output "${SKEL_DIR}/eval_graph_unique_40_seed${SEED}.json"

echo ""
echo "=== Stage 2: Populate (gpt-5.1 reasoning=low) + run SPINE rollouts ==="
echo "    task mix: ${TASK_PROPORTIONS} (EXIST POS REACH NAV)"
# shellcheck disable=SC2086
# python scripts/training_data_generation/generate_data_spine.py \
#   --data-dir "$SKEL_DIR" \
#   --name "$RUN_DIR" \
#   --task-proportions $TASK_PROPORTIONS \
#   --n-tasks "$N_TASKS" \
#   --seed "$SEED"

echo ""
echo "=== Stage 3: Leak check (must print CLEAN) ==="
if grep -RlE 'acceptance_criterion|"answer"[[:space:]]*:' "${RUN_DIR}/generated_plans/" >/dev/null 2>&1; then
  echo "LEAK in generated_plans/ — investigate before trusting this data"
  exit 2
else
  echo "CLEAN: generated_plans/"
fi

echo ""
echo "=== Stage 4: NEXT — spot-check quality before the full run ==="
echo "  /verify-tasks ${RUN_DIR}/populated_graphs --sample-size 1"
echo ""
echo "  Then inspect the ${N_TASKS} rollouts in ${RUN_DIR}/generated_plans/."
echo "  For rollout-quality judging use the e4-style audit flow, NOT /judge-eval"
echo "  (/judge-eval targets checkpoint eval logs, not data-gen rollouts)."
