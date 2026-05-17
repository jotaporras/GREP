#!/usr/bin/env bash
# Generate eval graph skeletons at various sizes, then print fill-graph
# instructions for the Claude skill (Stage 2).
#
# Usage: bash scripts/generate_eval_suite.sh
#
# Parameters from docs/plan.md; n_communities and nodes_per_community are
# chosen so that n_regions ≈ N / 1.3 (objects add ~30% via Poisson rate 0.3).

set -euo pipefail

OUTDIR="data/training/training_v1"

INTRA_PROB=0.6
INTER_PROB=0.05
OBJECT_RATE=0.3
DESC_PROB=0.05
N_TASKS=10
SEED=42

# Target N    n_communities  nodes_per_community
configs=(
  "30    3   8"
  "35    3   10"
  "40    4   9"
  "45    4   10"
  "50    4   11"
)

echo "=== Stage 1: Generating skeletons ==="

for cfg in "${configs[@]}"; do
  read -r N NC NPC <<< "$cfg"
  OUT="${OUTDIR}/eval_graph_unique_${N}.json"
  echo ""
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
    --output "$OUT"
done

echo ""
echo "=== Stage 2: Fill skeletons with Claude skill ==="
echo "Run the following commands in Cursor/Claude to semantically fill each skeleton:"
echo ""
for cfg in "${configs[@]}"; do
  read -r N _ _ <<< "$cfg"
  echo "  /fill-graph ${OUTDIR}/train_graph_unique_${N}.json"
done
echo ""
echo "Done."
