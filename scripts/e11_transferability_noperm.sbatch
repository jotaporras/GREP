#!/bin/bash
#SBATCH --job-name=e11-transfer-noperm
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#
# e11 no-permutation eval: identical invocation to scripts/e11_transferability.sbatch
# but WITHOUT --permutation-seed. Absence of that flag keeps scalability_evaluation
# in cross_eval layout, writing one per-graph JSON directly into <OUTPUT>:
#   <OUTPUT>/<graph>.json   (the eval_viewer.html / judge-eval shape)
#
# All relative paths from the original command are anchored under the cluster
# project dir $PROJ (=/vast/.../GREP-PRISM), independent of the submit CWD.
#
# Submit:  sbatch scripts/e11_transferability_noperm.sbatch
# Local :  bash   scripts/e11_transferability_noperm.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM

CHECKPOINT=$PROJ/outputs/e13f_alpha_khops/e13f_alpha00_ai8c2bm0
GRAPHS=$PROJ/data/n_100/gen/nav_n100_gemma_data/test_graphs
OUTPUT=$PROJ/results/e11_transferability

# Home dir sits near its quota; keep the HF hub cache (base gemma-4-31B-it
# weights, loaded 4-bit) on shared vast — downloaded once, reused across jobs.
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
# SLURM 22.05+ stopped propagating --cpus-per-task to srun; pass it explicitly
# to avoid "CPU binding outside of job step allocation" on dgx-b200 nodes.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-14}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] e11 no-perm eval"
echo "  checkpoint: $CHECKPOINT"
echo "  graphs:     $GRAPHS"
echo "  output:     $OUTPUT/"

srun uv run -m prism.eval.scalability_evaluation \
    --checkpoint "$CHECKPOINT" \
    --graphs "$GRAPHS" \
    --output "$OUTPUT" \
    --four-bit \
    --text-edge-list none \
    --use-icl false

echo "[$(date --iso-8601=seconds)] e11 no-perm eval DONE"
