#!/usr/bin/env bash
set -euo pipefail

OUTDIR="data/eval/dev_deleteme__e4_transferability_skeletons__deleteme"
mkdir -p "$OUTDIR"

for i in $(seq 2 20); do
    SEED=$((42 + i - 1))
    OUT="$OUTDIR/eval_graph_unique_100_${i}.json"

    python scripts/generate_eval_graphs.py \
        --n-communities 5 \
        --nodes-per-community 15 \
        --intra-community-prob 0.6 \
        --inter-community-prob 0.05 \
        --object-rate 0.3 \
        --description-prob 0.05 \
        --n-tasks 10 \
        --seed "$SEED" \
        --output "$OUT"

    python scripts/populate_eval_graph.py \
        --skeleton "$OUT" \
        --output   "$OUT"
done
