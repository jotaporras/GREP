#!/bin/bash
#SBATCH --job-name=verify-loss-masking
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%j.out
#SBATCH --time=00:45:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#
# Small single-GPU verification of the loss_target token-masking framework against
# the REAL stack: the gemma-4-12B tokenizer + the e8 SPINE training split (not a
# local proxy). Runs scripts/verify_loss_masking.py sections:
#   A unit (synthetic)  B real-data masking  C collator  D real-model loss behaviour
# Exits nonzero if any check FAILs.
#
# Submit:
#   sbatch scripts/verify_loss_masking.sbatch
# Or inside an interactive alloc (srun -p dgx-b200 --gpus=1 --pty bash):
#   bash scripts/verify_loss_masking.sbatch
# Overrides (env):
#   DATA=<spine conversations json>  TOKENIZER=<hf id>  MODEL=<hf id for section D>
#   SKIP_MODEL=1   # A/B/C only (no model forward — fastest)

set -euo pipefail

PROJECT=${PROJECT:-GREP-PRISM}
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}

# Real stack. DATA is repo-relative (as the e8 training configs read it); override
# if the split lives elsewhere on the cluster. TOKENIZER/MODEL = the real base.
DATA=${DATA:-data/revised/gen/nav100_n30_gemma_data/split/formatted_all_new__train.json}
TOKENIZER=${TOKENIZER:-google/gemma-4-12B-it}
MODEL=${MODEL:-google/gemma-4-12B-it}
NUM_EXAMPLES=${NUM_EXAMPLES:-4}

# Stable CWD. Under SLURM $0 is the spooled script — use SLURM_SUBMIT_DIR.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

log() { echo "[$(date --iso-8601=seconds)] $*"; }

[[ -f "$DATA" ]] || { log "ERROR: data not found: $DATA  (set DATA=... to the e8 split path)"; exit 1; }

CMD=(
    srun
    python scripts/verify_loss_masking.py
    --tokenizer "$TOKENIZER"
    --data "$DATA"
    --model "$MODEL"
    --num-examples "$NUM_EXAMPLES"
)
[[ "${SKIP_MODEL:-0}" == "1" ]] && CMD+=(--skip-model)

log "Verify: ${CMD[*]}"
"${CMD[@]}"
log "Verify done."
