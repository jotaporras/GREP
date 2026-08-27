#!/bin/bash
#SBATCH --job-name=e21-act-probe
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=3:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs / 224 GB per
# GPU on dgx-b200.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e21 activation/attention probe on the v2c final checkpoint (94.0% cell):
# re-generates 12 selected eval questions (the 5 residual errors + 7 matched
# correct controls) through the real eval loop, then replays each sequence
# teacher-forced under channel ablations (on / no_mask / no_pf / no_decision /
# off), dumping hidden-state deltas, attention rows and probability shifts.
# See scripts/e21_activation_probe.py.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-v3}

CHECKPOINT=${CHECKPOINT:-$PROJ/outputs/e20_path_only_pred/e20_oracle_v2c_hop_depth_route_only_4ghta3jm}
GRAPHS=${GRAPHS:-$PROJ/data/n_60_vllm_v3/gen/nav_n60_gemma_data/split/test_graphs}
OUT=${OUT:-$PROJ/outputs/e21_probe/v2c_final}
# 5 residual errors + 7 matched correct controls (see the e21 v2c results note).
SAMPLES=${SAMPLES:-data_gen_052:1,data_gen_052:7,data_gen_052:10,data_gen_052:11,data_gen_055:4,data_gen_055:5,data_gen_055:7,data_gen_055:9,data_gen_058:2,data_gen_058:8,data_gen_058:10,data_gen_058:11}

export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
export PATH="$ENV_NAME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# route_only checkpoints forbid SPINE tool prompts at eval.
export PRISM_DISABLE_SPINE_TOOLS=1

echo "[$(date --iso-8601=seconds)] probe: CHECKPOINT=$CHECKPOINT OUT=$OUT"
echo "=== python: $(which python) ==="

srun python scripts/e21_activation_probe.py \
    --checkpoint "$CHECKPOINT" \
    --graphs "$GRAPHS" \
    --samples "$SAMPLES" \
    --out "$OUT"

echo "[$(date --iso-8601=seconds)] e21-act-probe DONE"
