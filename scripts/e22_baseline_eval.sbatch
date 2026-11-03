#!/bin/bash
#SBATCH --job-name=e22-baseline
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=03:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter enforces EXACTLY 28 CPUs / 224 GB per GPU on dgx-b200.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e22 set D: cross_eval of an EXISTING checkpoint (default: 4ghta3jm, the e21 v2c
# hop_depth cell that the robot runs) on the e22 eval sets, through the same
# scalability_evaluation path as every cross_eval number in the e21 notes.
#
#   set C: data/hospital_graphs/eval_c/test_graphs  (scripts/hospital_eval_set.py,
#          default mode: robot at each task's start)
#   set B: $PROJ/data/n_geo_v1_test/.../test_graphs once shard 4 of
#          e22_geo_v1_generate.sbatch has run
#
# Output: $PROJ/results/e22_baseline/<run basename>/<set name>/ (per-graph JSONs
# in the eval_viewer / judge-eval layout), never the run dir's own cross_eval/.
#
# Submit:  sbatch scripts/e22_baseline_eval.sbatch
#          GRAPHS=<dir> SET_NAME=geo_test sbatch scripts/e22_baseline_eval.sbatch
#          RUN_DIR=<other run dir> sbatch scripts/e22_baseline_eval.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-v3}

RUN_DIR=${RUN_DIR:-$PROJ/outputs/e20_path_only_pred/e20_oracle_v2c_hop_depth_route_only_4ghta3jm}
GRAPHS=${GRAPHS:-$PROJ/data/hospital_graphs/eval_c/test_graphs}
SET_NAME=${SET_NAME:-hospital_c}
OUT=${OUT:-$PROJ/results/e22_baseline/$(basename "$RUN_DIR")/$SET_NAME}

# gemma-4-31B-it lives under $ALELAB/.cache/huggingface, not the older hf_cache dir.
export HF_HOME=${HF_HOME:-$ALELAB/.cache/huggingface}
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"
export PATH="$ENV_NAME/bin:$PATH"
# route_only checkpoints: tools off, no ICL (same as the e20/e21 cross_eval jobs).
export PRISM_DISABLE_SPINE_TOOLS=1
# 100-region geo prompts + looping generations hit a 32 GiB attention alloc with
# 63 GiB reserved-but-unallocated (job 9989430, 3 OOM tasks); let the allocator grow.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Generation cap (prism.models.inference): a route_only answer never needs the
# 2048 default; looping outputs ran to it and still OOMed (1 task even with
# expandable_segments in job 10016958). Valid routes are unaffected.
export PRISM_EVAL_MAX_NEW_TOKENS=${PRISM_EVAL_MAX_NEW_TOKENS:-512}

# RUN_DIR may be a run dir or one of its checkpoint-N subdirs (both carry
# train_config.json + gnn_weights.pt + the adapter); set OUT for a subdir.
test -f "$RUN_DIR/train_config.json" || { echo "missing $RUN_DIR/train_config.json"; exit 2; }
test -d "$GRAPHS" || { echo "missing graphs dir $GRAPHS"; exit 2; }
mkdir -p "$OUT"

echo "[$(date --iso-8601=seconds)] e22 baseline run_dir=$RUN_DIR set=$SET_NAME graphs=$GRAPHS out=$OUT commit=$(git rev-parse --short HEAD)"
echo "=== python: $(which python) ==="

srun python -m prism.eval.scalability_evaluation \
    --checkpoint "$RUN_DIR" \
    --graphs "$GRAPHS" \
    --output "$OUT" \
    --use-icl false \
    --device 0

echo "[$(date --iso-8601=seconds)] e22 baseline $SET_NAME DONE -> $OUT"
