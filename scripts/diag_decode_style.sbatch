#!/bin/bash
#SBATCH --job-name=diag-decode
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=06:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-2
#
# Decode-realizability gate for decode-time mask injection (decode-time design
# note §3): rerun the injection-ablation diagnostic with the two new
# decode-realizable conditions (decode_style, decode_trail) alongside
# train_style / prompt_only / no_injection. The train_style-vs-decode_* gap is
# the part of the leak-trained channel decode-time injection CANNOT legally
# restore.
#
#   idx  checkpoint                                          why
#   0    e10_integ_rpe__no_edge_list_tyvhwlmx (leaky, RPE)   the 0.865 circuit — primary
#   1    e11_integ_rpe_noedges_promptonly_22lq43i6 (clean)   does the leak-free circuit read decode wiring zero-shot?
#   2    rv_gmask_noedges_ror8gtet (leaky, param-free)       mechanism replicates without learned Ψ?
#
# Output: injection_diag_decode.json in each run dir (does NOT clobber the
# historical injection_diag.json cited in the findings doc).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras

export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

module load anaconda3
source activate "$ENV_NAME"

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
case "$TASK_ID" in
    0) CKPT=$ALELAB/GREP-PRISM/outputs/dev_e10/e10_integ_rpe/e10_integ_rpe__no_edge_list_tyvhwlmx ;;
    1) CKPT=$ALELAB/GREP-PRISM/outputs/e11_injection_scope/e11_integ_rpe_noedges_promptonly_22lq43i6 ;;
    2) CKPT=$ALELAB/GREP-PRISM/outputs/refactor_verify_fulltest/rv_gmask_noedges_ror8gtet ;;
    3) CKPT=$ALELAB/GREP-PRISM/outputs/dev_e10/e10_integ_rpe/e10_integ_rpe__with_edge_list_wo83xg96 ;;
    *) echo "bad task id $TASK_ID" >&2; exit 1 ;;
esac
echo "[$(date --iso-8601=seconds)] decode-style diag on $CKPT"

srun python scripts/diag_injection_ablation.py \
    --checkpoint "$CKPT" \
    --val-file data/revised/gen/nav100_n30_gemma_data/split/formatted_all_new__val.json \
    --out "$CKPT/injection_diag_decode.json"

echo "[$(date --iso-8601=seconds)] DONE: $CKPT/injection_diag_decode.json"
