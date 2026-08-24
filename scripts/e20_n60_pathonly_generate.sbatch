#!/bin/bash
#SBATCH --job-name=e20-n60-pathonly-gen
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e20a — path-only distillation corpus on the SAME n60_v3 populated graphs.
#
# Phase 1 is skipped entirely (--skip-populate over n_60_vllm_v3's
# populated_graphs/), so graphs, node names, and all 12 tasks/graph are
# IDENTICAL to the e19 training corpus — only the teacher responses change:
# ONE route-only no-think query per task (generate_data_spine.py --path-only),
# graded with the deterministic eval scorer; wrong routes are discarded.
#
# Two monitors land next to the data:
#   generated_plans/rollout_stats.json         path-only teacher pass rate
#   (v3 corpus)/reasoning_grade_stats.json     the SAME grader over the original
#                                              reasoning rollouts (paired
#                                              teacher-accuracy comparison)
#
# The split pins the SAME 7 frozen test graphs (052-058) so e20 evals remain
# comparable to e19/cn7ub88q.
#
# Optional: PATH_ONLY_THINKING=1 keeps the teacher's <think> mode on (block is
# stripped before route extraction) — the ablation for "is the accuracy drop
# from no-think or from the route-only contract?".
#
# Submit:  sbatch scripts/e20_n60_pathonly_generate.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export ALELAB_DRIVE="${ALELAB_DRIVE:-/vast/projects/aribeiro/alelab/jporras}"
PROJ=$ALELAB_DRIVE/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB_DRIVE/envs/GREP-PRISM-vllm}
export HF_HOME=${HF_HOME:-$ALELAB_DRIVE/hf_cache}

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PRISM_LLM_BACKEND=vllm
export VLLM_USE_FLASHINFER_SAMPLER=0
export PRISM_VLLM_MAX_LEN=32768
export PRISM_VLLM_TP=1
export PRISM_HF_MODEL="google/gemma-4-31B-it"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"

V3_RUN="$PROJ/data/n_60_vllm_v3/gen/nav_n60_gemma_data"
V3_POPULATED="$V3_RUN/populated_graphs"
DATA_ROOT="${DATA_ROOT:-$PROJ/data/n_60_pathonly_v1}"
RUN_DIR="$DATA_ROOT/gen/nav_n60_gemma_data"
SPLIT_DIR="$RUN_DIR/split"
VAL_IDS="${VAL_IDS:-052 053 054 055 056 057 058}"
MAX_GRAPHS=59

PATH_ONLY_ARGS=(--path-only)
if [ "${PATH_ONLY_THINKING:-0}" = "1" ]; then
  PATH_ONLY_ARGS+=(--path-only-thinking)
fi

test -d "$V3_POPULATED" || { echo "missing $V3_POPULATED"; exit 2; }
mkdir -p "$RUN_DIR"

echo "[e20a] path-only distill gen over $V3_POPULATED (args: ${PATH_ONLY_ARGS[*]})"

# Auto-resume loop (same contract as nav_n100_gemma_generate.sh Stage 2).
MAX_ATTEMPTS="${PRISM_MAX_ATTEMPTS:-5}"
attempt=1
while true; do
  if srun --ntasks=1 python scripts/training_data_generation/generate_data_spine.py \
      --skip-populate \
      --data-dir "$V3_POPULATED" \
      --name "$RUN_DIR" \
      --max-graphs "$MAX_GRAPHS" \
      --rollout-workers "$ROLLOUT_WORKERS" \
      "${PATH_ONLY_ARGS[@]}"; then
    echo "[e20a] generation completed on attempt ${attempt}/${MAX_ATTEMPTS}."
    break
  else
    rc=$?
    if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
      echo "ERROR: generation still failing after ${MAX_ATTEMPTS} attempts (rc=${rc})" >&2
      exit "$rc"
    fi
    echo "[e20a] attempt ${attempt} exited rc=${rc}; resuming in 10s..." >&2
    attempt=$((attempt + 1))
    sleep 10
  fi
done

echo "[e20a] train/val split (frozen test graphs: $VAL_IDS)"
# shellcheck disable=SC2086
srun --ntasks=1 python scripts/training_data_generation/split_train_val.py \
  --plans-dir "$RUN_DIR/generated_plans" \
  --graphs-dir "$V3_POPULATED" \
  --output-dir "$SPLIT_DIR" \
  --val-ids $VAL_IDS

echo "[e20a] leak check"
if grep -RlE 'acceptance_criterion|"answer"[[:space:]]*:' "$RUN_DIR/generated_plans/" >/dev/null 2>&1; then
  echo "LEAK in generated_plans/ — investigate before training on this data"
  exit 2
fi
echo "CLEAN: generated_plans/"

echo "[e20a] paired comparison: grade the ORIGINAL v3 reasoning rollouts with the same scorer"
srun --ntasks=1 python scripts/e20_grade_rollouts.py \
  --plans-dir "$V3_RUN/generated_plans" \
  --graphs-dir "$V3_POPULATED" \
  --out "$V3_RUN/generated_plans/reasoning_grade_stats.json"

echo "[e20a] teacher-accuracy comparison:"
echo "  path-only : $RUN_DIR/generated_plans/rollout_stats.json"
echo "  reasoning : $V3_RUN/generated_plans/reasoning_grade_stats.json"
echo "[e20a] DONE"
