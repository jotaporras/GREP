#!/bin/bash
#SBATCH --job-name=e22-postgen
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=01:00:00
#SBATCH --partition=genoa-std-mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
# no --mem: the cli_filter derives memory from CPUs (5632M each on genoa-std-mem).
#
# e22 post-generation gate (docs/2026-11-02 e22_geometric_scale_design.md §6),
# run as a CPU job (never on the login node) once the four training shards of
# scripts/e22_geo_v1_generate.sbatch have finished:
#
#   1. scripts/e21_merge_shards.py  -> <ROOT>/split/formatted_all_new__{train,val}.json
#   2. scripts/e22_corpus_audit.py  -> exit 1 on any FAIL gate
#
# Exit status is the audit's, so an SFT job submitted with
# --dependency=afterok:<this job> never starts on a corpus that failed the gate.
#
# Submit:  sbatch --dependency=afterok:<gen>_0:<gen>_1:<gen>_2:<gen>_3 scripts/e22_postgen.sbatch
#          ROOT=<other corpus root> sbatch scripts/e22_postgen.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_NAME=${ENV_NAME:-$ALELAB/envs/GREP-PRISM-v3}
ROOT=${ROOT:-$PROJ/data/n_geo_v1}
GEN_SUBDIR=${GEN_SUBDIR:-gen/nav_ngeo_gemma_data/split}
COMPARE=${COMPARE:-$PROJ/data/n_60_oracle_v2c_r2/split/formatted_all_new__train.json}

export HF_HOME=${HF_HOME:-$ALELAB/.cache/huggingface}
export WANDB_MODE=offline
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] host=$(hostname) commit=$(git rev-parse --short HEAD) root=$ROOT"
echo "=== merge ==="
python scripts/e21_merge_shards.py --root "$ROOT" --gen-subdir "$GEN_SUBDIR"

echo "=== audit ==="
COMPARE_ARGS=()
test -f "$COMPARE" && COMPARE_ARGS=(--compare "$COMPARE")
python scripts/e22_corpus_audit.py --root "$ROOT" ${COMPARE_ARGS[@]+"${COMPARE_ARGS[@]}"}
echo "[$(date --iso-8601=seconds)] e22 postgen gate PASSED for $ROOT"
