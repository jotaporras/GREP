#!/bin/bash
#SBATCH --job-name=e4-eval
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A.out
#SBATCH --time=6:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G

# Post-hoc eval of a single checkpoint against the 10 v4 val test graphs.
#
# Usage:
#   sbatch scripts/e4_run_eval.sbatch <checkpoint_dir>
#   CHECKPOINT=<dir> sbatch scripts/e4_run_eval.sbatch
#
# Architecture (LLM vs graph-augmented) is auto-detected from the checkpoint
# contents by scripts/eval_checkpoint_on_graphs.py:_load_checkpoint.
# Results land in <checkpoint>/eval_logs/cross_eval/<graph>.json.

set -euo pipefail

PROJECT=${PROJECT:-GREP-PRISM}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM}
OUTPUT_BASE=${OUTPUT_BASE:-/vast/home/j/jporras/sourcecode/GREP}
GRAPHS=${GRAPHS:-data/gen/e4_training_data_v4/split/test_graphs}

CHECKPOINT=${CHECKPOINT:-${1:-}}
if [[ -z "$CHECKPOINT" ]]; then
    echo "Usage: sbatch $0 <checkpoint_dir>   (or set CHECKPOINT env var)" >&2
    exit 1
fi

mkdir -p "${OUTPUT_BASE}"
cd "$(dirname "$0")/.."

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

log() {
    echo "[$(date --iso-8601=seconds)] $*"
}

if [[ ! -d "$CHECKPOINT" ]]; then
    log "Checkpoint dir does not exist: $CHECKPOINT"
    exit 1
fi

log "Evaluating checkpoint: $CHECKPOINT"
log "Graphs: $GRAPHS"

CMD=(
    srun
    python scripts/eval_checkpoint_on_graphs.py
    --checkpoint "$CHECKPOINT"
    --graphs "$GRAPHS"
)

log "CMD: ${CMD[*]}"
"${CMD[@]}"

log "Eval complete. Results under: ${CHECKPOINT}/eval_logs/cross_eval/"
