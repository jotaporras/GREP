#!/bin/bash
#SBATCH --job-name=e14_transferability_spine
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e14 no-permutation eval WITH SPINE tool-calling ENABLED — the tools-on counterpart
# to scripts/e14_transferability.sbatch (which sets PRISM_DISABLE_SPINE_TOOLS). This
# script leaves that env var unset, so the eval harness
# (evaluate._spine_tools_disabled) keeps tool-calling on. Same cross_eval output
# layout: one per-graph JSON directly into <OUTPUT>:
#   <OUTPUT>/<graph>.json   (the eval_viewer.html / judge-eval shape)
#
# Checkpoint / graphs / output are overridable, in priority order:
#   positional arg  >  same-named env var  >  built-in default.
# A relative override is still anchored under the cluster project dir $PROJ
# (=/vast/.../GREP-PRISM); an absolute override passes through untouched.
# CHECKPOINT has NO built-in default: the e14 run dir carries the wandb id
# (outputs/e14_stage1to3/<save_name>_<wandbid>), unknown until training has run,
# so a missing checkpoint fails loud here instead of evaluating the wrong run.
#
# Submit:  sbatch scripts/e14_transferability_spine.sbatch [CHECKPOINT] [GRAPHS] [OUTPUT]
#   env  :  CHECKPOINT=... GRAPHS=... OUTPUT=... sbatch scripts/e14_transferability_spine.sbatch
# Local :  bash   scripts/e14_transferability_spine.sbatch [CHECKPOINT] [GRAPHS] [OUTPUT]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM

# Anchor a relative path under $PROJ; leave an absolute path unchanged.
_anchor() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s' "$PROJ/$1" ;; esac; }

_CHECKPOINT_IN=${1:-${CHECKPOINT:-}}
if [ -z "$_CHECKPOINT_IN" ]; then
  echo "ERROR: no checkpoint. Pass the e14 run dir, e.g." >&2
  echo "  sbatch scripts/e14_transferability_spine.sbatch outputs/e14_stage1to3/e14_stage1to3_binary_<wandbid>" >&2
  exit 1
fi
CHECKPOINT=$(_anchor "$_CHECKPOINT_IN")
GRAPHS=$(_anchor "${2:-${GRAPHS:-data/n_100/gen/nav_n100_gemma_data/test_graphs}}")
OUTPUT=$(_anchor "${3:-${OUTPUT:-results/e14_transferability_spine}}")

# ICL policy, stated explicitly so it cannot drift with the config defaults:
#   USE_ICL       -> eval.use_icl. ON: eval prompts carry few-shot SPINE demos.
#   DATA_ICL_EXAMPLES -> data.icl_examples. 2: training runs few-shot.
# Only USE_ICL is consumed here — scalability_evaluation is an eval-only entrypoint and
# has no training stage, so data.icl_examples is recorded for the record rather
# than passed. ICL lengthens every eval prompt, so expect a larger KV
# cache and slower generation than the zero-shot arm this replaces.
USE_ICL=${USE_ICL:-true}
# Records the TRAINING policy the evaluated checkpoint was produced under — e14 training
# now runs FEW-SHOT (data.icl_examples=2), so 0 here would misstate it. scalability_
# evaluation is eval-only and has no training stage, so this value has no consumer: it is
# echoed rather than passed, because silently carrying an unused knob is how a train/eval
# policy mismatch goes unnoticed.
DATA_ICL_EXAMPLES=${DATA_ICL_EXAMPLES:-2}
echo "  training policy (recorded, not consumed here): data.icl_examples=$DATA_ICL_EXAMPLES"

# Home dir sits near its quota; keep the HF hub cache (base gemma-4-31B-it
# weights, loaded 4-bit) on shared vast — downloaded once, reused across jobs.
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
# SLURM 22.05+ stopped propagating --cpus-per-task to srun; pass it explicitly
# to avoid "CPU binding outside of job step allocation" on dgx-b200 nodes.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"
# NOTE: PRISM_DISABLE_SPINE_TOOLS is intentionally NOT set here — SPINE tool-calling
# stays enabled (this is the tools-on arm of the A/B).

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] e14 no-perm eval (SPINE tools ENABLED)"
echo "  checkpoint: $CHECKPOINT"
echo "  graphs:     $GRAPHS"
echo "  output:     $OUTPUT/"

srun python -m prism.eval.scalability_evaluation \
    --checkpoint "$CHECKPOINT" \
    --graphs "$GRAPHS" \
    --output "$OUTPUT" \
    --four-bit \
    --text-edge-list none \
    --use-icl "$USE_ICL"

echo "[$(date --iso-8601=seconds)] e14 no-perm eval (SPINE tools ENABLED) DONE"
