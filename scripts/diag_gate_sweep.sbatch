#!/bin/bash
#SBATCH --job-name=diag-gate-sweep
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=2:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --exclude=dgx024
#
# Gate-sweep diagnostic (e12 follow-up): on the leak-free e12 stage-3 no-edge
# checkpoint, is the trained gate value (tanh(pe_gain) ~= 0.23) amplitude-limiting
# for the current circuit? Runs the prompt_only injection map at effective gate
# values {0, 0.1, 0.25, 0.5, 0.75, 1.0} + trained; grades decision tokens.
# Forwards only, no training. Read via scripts/diag_injection_ablation.py docstring.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
OUTPUTS=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/outputs
CHECKPOINT=${CHECKPOINT:-$OUTPUTS/e12_leakfree_multistage/stage3/e12_ms_stage3__no_edge_list_wh0537au}
VAL_FILE=${VAL_FILE:-data/revised/gen/nav100_n30_gemma_data/split/formatted_all_new__val.json}

export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

module load anaconda3
source activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] gate sweep: CHECKPOINT=${CHECKPOINT}"
srun python scripts/diag_injection_ablation.py \
    --checkpoint "$CHECKPOINT" \
    --val-file "$VAL_FILE" \
    --gate-sweep "0,0.1,0.25,0.5,0.75,1.0"
