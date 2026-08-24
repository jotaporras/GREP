#!/bin/bash
#SBATCH --job-name=spine-n60-gen
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs / 224 GB per
# GPU on dgx-b200.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e19 SPINE closed-loop corpus (docs/2026-08-21 e18_spine_closed_loop_design.md
# + the 2026-08-24 decisions in the e19 design note): teacher tool-call
# trajectories WITH reasoning on the EXISTING n60_v3 graphs/tasks.
#
#   Stage 1  Phase-2 rollouts only (--skip-populate): Gemma-4-31B-it teacher,
#            vLLM, SPINE tools ON, goto ratification ON (default), a fraction
#            of rollouts get fake 2-hop shortcut edges in the PROMPT graph so
#            rejection-recovery turns actually occur (--fake-edge-frac).
#   Stage 2  Rejection filter: replay-grade every rollout (structural route
#            validation, judge off) and quarantine incorrect ones.
#   Stage 3  split_train_val with the SAME held-out graphs as n60_v3
#            (VAL_IDS 052-058) so all e17-e19 baselines stay comparable.
#   Stage 4  answer-leak check + turn-shape stats.
#
# Submit:  sbatch scripts/nav_n60_spine_generate.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
# GREP-PRISM-vllm, not -v3: the rollout backend needs vllm (same env the v3
# corpus generation used).
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-vllm}

SRC=${SRC:-$PROJ/data/n_60_vllm_v3/gen/nav_n60_gemma_data}
OUT=${OUT:-$PROJ/data/n_60_spine_v1/gen/nav_n60_spine_data}
FAKE_EDGE_FRAC=${FAKE_EDGE_FRAC:-0.3}
FAKE_EDGES_N=${FAKE_EDGES_N:-2}
GEN_SEED=${GEN_SEED:-42}
ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-8}
VAL_IDS=${VAL_IDS:-"052 053 054 055 056 057 058"}
MAX_GRAPHS=${MAX_GRAPHS:-}          # smoke runs: MAX_GRAPHS=2 OUT=... sbatch ...
MAX_GRAPHS_ARGS=()
[ -n "$MAX_GRAPHS" ] && MAX_GRAPHS_ARGS=(--max-graphs "$MAX_GRAPHS")

export PRISM_LLM_BACKEND=vllm
# Compute nodes have no nvcc; flashinfer's JIT sampler dies on CUDA_HOME.
export VLLM_USE_FLASHINFER_SAMPLER=0
export PRISM_HF_MODEL=${PRISM_HF_MODEL:-google/gemma-4-31B-it}
export PRISM_VLLM_MAX_LEN=${PRISM_VLLM_MAX_LEN:-32768}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_HUB_DISABLE_PROGRESS_BARS=1
# Filter grading must be deterministic (structural route validation only).
export GREP_JUDGE=0
# PATH prepend, not `conda activate`: compute-node batch shells have no conda.
export PATH="$ENV_NAME/bin:$PATH"

test -d "$SRC/populated_graphs" || { echo "missing $SRC/populated_graphs"; exit 2; }
mkdir -p "$OUT"

echo "[$(date --iso-8601=seconds)] stage 1: rollouts src=$SRC out=$OUT fake_edge_frac=$FAKE_EDGE_FRAC"
python scripts/training_data_generation/generate_data_spine.py \
    --name "$OUT" \
    --data-dir "$SRC/populated_graphs" \
    --skip-populate \
    --rollout-workers "$ROLLOUT_WORKERS" \
    --fake-edge-frac "$FAKE_EDGE_FRAC" \
    --fake-edges-n "$FAKE_EDGES_N" \
    --seed "$GEN_SEED" \
    ${MAX_GRAPHS_ARGS[@]+"${MAX_GRAPHS_ARGS[@]}"}

echo "[$(date --iso-8601=seconds)] stage 2: rejection filter"
python scripts/training_data_generation/filter_correct_rollouts.py \
    --graphs-dir "$SRC/populated_graphs" \
    --plans-dir "$OUT/generated_plans"

echo "[$(date --iso-8601=seconds)] stage 3: split (val ids: $VAL_IDS)"
python scripts/training_data_generation/split_train_val.py \
    --plans-dir "$OUT/generated_plans" \
    --graphs-dir "$SRC/populated_graphs" \
    --output-dir "$OUT/split" \
    --val-ids $VAL_IDS

echo "[$(date --iso-8601=seconds)] stage 4: leak check + turn stats"
if grep -rl "acceptance_criterion" "$OUT/generated_plans"/sample_*.json \
        --include="sample_[0-9][0-9][0-9]_[0-9][0-9][0-9].json" 2>/dev/null; then
    echo "LEAK: acceptance_criterion found in committed samples"; exit 3
fi
python - "$OUT/split/formatted_all_new__train.json" <<'PYEOF'
import json, sys
convs = [c["conversations"] for c in json.load(open(sys.argv[1]))]
lens = [len(c) for c in convs]
multi = sum(1 for L in lens if L > 3)
asst = [sum(1 for m in c if m["role"] == "assistant") for c in convs]
print(f"train: {len(convs)} episodes; >3-message (multi-turn): {multi} "
      f"({100*multi/max(1,len(convs)):.1f}%); "
      f"assistant turns mean {sum(asst)/max(1,len(asst)):.2f} max {max(asst or [0])}")
PYEOF

echo "[$(date --iso-8601=seconds)] DONE -> $OUT/split"
