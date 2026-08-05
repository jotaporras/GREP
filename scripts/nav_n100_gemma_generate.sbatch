#!/bin/bash
#SBATCH --job-name=nav-n100-gemma31b-gen
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#
# SLURM wrapper for scripts/nav_n100_gemma_generate.sh (local Gemma 4 nav-only
# ~100-node training-data generation — no OpenAI calls). All generation logic,
# env-var defaults, staging, auto-resume and leak checks live in that script;
# this file only allocates a GPU node and invokes it.
#
# GPU id passed to the inner script is 0 (NOT a physical device id): --gpus=1
# already masks all but the allocated GPU, which CUDA enumerates as device 0.
# The inner script pins CUDA_VISIBLE_DEVICES=0 + PRISM_REQUIRE_SINGLE_GPU=1.
#
# FIXED run config (baked into the srun invocation below, not overridable):
#   MODEL     google/gemma-4-31B-it  (dense 31B; forced via the inner script's
#             $2 slot, which outranks $PRISM_HF_MODEL and the 26B-A4B default)
#   N_GRAPHS  33                     (3rd positional arg)
#
# Overridable inner-script knob (export before sbatch, or use --export):
#   PRISM_HF_QUANT   none|4bit|8bit  (default 4bit; a b200 fits none=bf16 ~62GB)
#
# Submit:
#   sbatch scripts/nav_n100_gemma_generate.sbatch
#   PRISM_HF_QUANT=none sbatch scripts/nav_n100_gemma_generate.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
# Home dir sits near its quota; keep the HF hub cache on shared vast (downloads
# the ~26B weights once, reused across jobs).
export HF_HOME=${HF_HOME:-/vast/projects/aribeiro/alelab/jporras/hf_cache}

module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Inner script args: [GPU_ID] [MODEL] [N_GRAPHS]. GPU_ID=0 (allocation-relative);
# MODEL forced to the dense 31B ($2 outranks $PRISM_HF_MODEL); N_GRAPHS=33.
srun bash scripts/nav_n100_gemma_generate.sh 0 "google/gemma-4-31B-it" 33
