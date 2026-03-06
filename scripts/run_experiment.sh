#!/bin/bash
set -e
# Usage: ./scripts/run_experiment.sh experiments/e1_rpearl_llm.yaml [experiments/e1_llm.yaml ...]
for config in "$@"; do
    echo "=== Running: $config ==="
    python -m prism.training.train_v2 "$config"
    echo "=== Done: $config ==="
done
