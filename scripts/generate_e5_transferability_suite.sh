#!/usr/bin/env bash
# Generate the e5 transferability eval suite (10 graphs per size):
#   1. Skeleton scene graphs at 5 sizes x 10 seeds via generate_eval_graphs.py.
#   2. Populate each via scripts/populate_eval_graph.py (uses the DataGenerator
#      default model — currently gpt-5.1).
#
# Unlike scripts/generate_eval_suite.sh (1 graph per size), this generates 10
# graphs per size so each accuracy-vs-size point averages over 10 graphs.
#
# Runs locally on the workstation (needs OPENAI_API_KEY from .env). ~50 sequential
# GPT populate calls — run it in the background. After it finishes, rsync
# data/eval/e5_transferability/ to the cluster repo and submit
# scripts/e5_transferability.sbatch.
#
# Usage: bash scripts/generate_e5_transferability_suite.sh

set -euo pipefail

# Auto-load .env so OPENAI_API_KEY is present in a plain bash shell.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

SKEL_DIR="data/eval/e5_transferability_skeletons"
OUT_DIR="data/eval/e5_transferability"

INTRA_PROB=0.6
INTER_PROB=0.05
OBJECT_RATE=0.3
DESC_PROB=0.05
N_TASKS=10

# 10 distinct seeds -> 10 distinct graphs per size.
SEEDS=(42 142 242 342 442 542 642 742 842 942)

# Target N    n_communities  nodes_per_community
# (n_communities * nodes_per_community + Poisson(object_rate) ≈ N)
# N=500 and N=1000 dropped: the populate model silently truncates the graph at
# those sizes (drops most regions, leaves dangling edges, breaks every task).
configs=(
  "10    2   4"
  "30    3   8"
  "50    4  10"
  "100   5  15"
  "250   6  32"
)

mkdir -p "$SKEL_DIR" "$OUT_DIR"

echo "=== Stage 1: Generate skeletons (${#configs[@]} sizes x ${#SEEDS[@]} seeds) ==="
for cfg in "${configs[@]}"; do
  read -r N NC NPC <<< "$cfg"
  for SEED in "${SEEDS[@]}"; do
    echo "--- N=${N}  seed=${SEED}  (${NC} communities × ${NPC} nodes/community) ---"
    python scripts/generate_eval_graphs.py \
      --n-communities "$NC" \
      --nodes-per-community "$NPC" \
      --intra-community-prob "$INTRA_PROB" \
      --inter-community-prob "$INTER_PROB" \
      --object-rate "$OBJECT_RATE" \
      --description-prob "$DESC_PROB" \
      --n-tasks "$N_TASKS" \
      --seed "$SEED" \
      --output "${SKEL_DIR}/eval_graph_unique_${N}_s${SEED}.json"
  done
done

echo ""
echo "=== Stage 2: Populate ==="
for cfg in "${configs[@]}"; do
  read -r N _ _ <<< "$cfg"
  for SEED in "${SEEDS[@]}"; do
    OUT="${OUT_DIR}/eval_graph_unique_${N}_s${SEED}.json"
    if [ -f "$OUT" ]; then
      echo "--- N=${N}  seed=${SEED} — already populated, skipping ---"
      continue
    fi
    echo "--- N=${N}  seed=${SEED} ---"
    python scripts/populate_eval_graph.py \
      --skeleton "${SKEL_DIR}/eval_graph_unique_${N}_s${SEED}.json" \
      --output   "$OUT"
  done
done

echo ""
echo "=== Done. Next steps ==="
echo "  /verify-tasks ${OUT_DIR} --sample-size 5"
echo "  rsync -av --delete ${OUT_DIR}/ betty:/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/${OUT_DIR}/"
echo "  sbatch scripts/e5_transferability.sbatch    # on the cluster"
