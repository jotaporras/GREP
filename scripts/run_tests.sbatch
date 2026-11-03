#!/bin/bash
#SBATCH --job-name=grep-tests
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=01:30:00
#SBATCH --partition=genoa-std-mem
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# no --mem: the cli_filter derives memory from CPUs (5632M each on genoa-std-mem)
#
# Run the pytest suite inside a CPU job (never on the login node).
#
#   sbatch scripts/run_tests.sbatch                 # whole tests/ dir
#   sbatch scripts/run_tests.sbatch tests/test_x.py # a subset (extra args go to pytest)
#
# Env: GREP-PRISM-v3 (the full working env, real SPINE). Tests that need a GPU
# skip or fail here; read the summary at the end of the log.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

export ALELAB_DRIVE="${ALELAB_DRIVE:-/vast/projects/aribeiro/alelab/jporras}"
ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
export HF_HOME=${HF_HOME:-/vast/projects/aribeiro/alelab/jporras/hf_cache}
export WANDB_MODE=offline

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] host=$(hostname) commit=$(git rev-parse --short HEAD) python=$(which python)"
python -m pytest "${@:-tests/}" -q -p no:cacheprovider -rfE
echo "[$(date --iso-8601=seconds)] pytest exit=$?"
