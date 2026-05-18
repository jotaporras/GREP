#!/usr/bin/env bash
# Generate the e4 transferability eval suite:
#   1. Skeleton scene graphs at 7 sizes via generate_eval_graphs.py.
#   2. GPT-5.5 populate (reasoning=xhigh) via scripts/populate_eval_graph.py.
#
# Runs locally on the workstation (needs OPENAI_API_KEY from .env). After it
# finishes, rsync data/eval/e4_transferability/ to the cluster repo and submit
# scripts/e4_transferability.sbatch.
#
# Usage: bash scripts/generate_eval_suite.sh

set -euo pipefail

# Auto-load .env so OPENAI_API_KEY is present in a plain bash shell.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "${REPO_ROOT}/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

SKEL_DIR="data/eval/e4_transferability_skeletons"
OUT_DIR="data/eval/e4_transferability"

INTRA_PROB=0.6
INTER_PROB=0.05
OBJECT_RATE=0.3
DESC_PROB=0.05
N_TASKS=10
SEED=42

# Target N    n_communities  nodes_per_community
# (n_communities * nodes_per_community + Poisson(object_rate) ≈ N)
configs=(
  "10    2   4"
  "30    3   8"
  "50    4  10"
  "100   5  15"
  "250   6  32"
  "500   7  55"
  "1000  8  96"
)

mkdir -p "$SKEL_DIR" "$OUT_DIR"

echo "=== Stage 1: Generate skeletons ==="
for cfg in "${configs[@]}"; do
  read -r N NC NPC <<< "$cfg"
  echo "--- N=${N}  (${NC} communities × ${NPC} nodes/community) ---"
  python scripts/generate_eval_graphs.py \
    --n-communities "$NC" \
    --nodes-per-community "$NPC" \
    --intra-community-prob "$INTRA_PROB" \
    --inter-community-prob "$INTER_PROB" \
    --object-rate "$OBJECT_RATE" \
    --description-prob "$DESC_PROB" \
    --n-tasks "$N_TASKS" \
    --seed "$SEED" \
    --output "${SKEL_DIR}/eval_graph_unique_${N}.json"
done

echo ""
echo "=== Stage 2: Populate (gpt-5.5 reasoning=xhigh) ==="
for cfg in "${configs[@]}"; do
  read -r N _ _ <<< "$cfg"
  echo "--- N=${N} ---"
  python scripts/populate_eval_graph.py \
    --skeleton "${SKEL_DIR}/eval_graph_unique_${N}.json" \
    --output   "${OUT_DIR}/eval_graph_unique_${N}.json"
done

echo ""
echo "=== Done. Next steps ==="
echo "  /verify-tasks ${OUT_DIR} --sample-size ${#configs[@]}"
echo "  rsync -av --delete ${OUT_DIR}/ <cluster>:/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/${OUT_DIR}/"
echo "  sbatch scripts/e4_transferability.sbatch    # on the cluster"
