#!/bin/bash
#SBATCH --job-name=e16-rl-mask
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs and 224 GB
# per GPU on dgx-b200. ONE GPU: no vLLM engine exists for the mask arch —
# rollouts run through the policy itself (MaskGRPOTrainer batched HF
# generation), so the whole job is a single-process, single-GPU trainer.
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e16 — GRPO RL training of the MASK architecture (learnable_graph_mask),
# warm-started from the best e13f binary-GT SFT checkpoint. The GT (Ψ tower)
# is UNFROZEN and trains on the RL reward through the structural mask
# (loss → bias → ΨΨᵀ → GT). Rollouts are batched HF generation with the
# decode-consistent injector (BatchedMaskDecodeInjector) — semantics identical
# to eval and to SFT training, and exactly on-policy (the policy IS the
# rollout model; no engine, no weight sync).
#
# Tower LR: e13f trained its GT at structural_lr_mult=0.012; RL warm-starting
# a converged tower defaults to mult=1 (tower at the RL-safe base LR) — a
# 20x boost like the cold-start additive runs would plough a trained GT.
#
# The wandb tag is LOCKED as e16_rl_training — never drift. RUN_NAME is the
# only identity knob a rerun needs.
#
# Submit:  sbatch scripts/e16_rl_mask.sbatch [RUN_NAME]
#   env  :  RUN_NAME=... MAX_STEPS=... INIT_CHECKPOINT=... \
#             sbatch scripts/e16_rl_mask.sbatch
# Local :  bash scripts/e16_rl_mask.sbatch [RUN_NAME]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_DIR=${ENV_DIR:-$ALELAB/envs/GREP-PRISM-rl}
CHECKPOINT_DIR=$PROJ/outputs/e16_rl_training

# Warm start: best e13f (alpha=0 binary GT, decode_consistent) run dir.
INIT_CHECKPOINT=${INIT_CHECKPOINT:-$PROJ/outputs/e13f_alpha_khops/e13f_alpha00_binary_nyvi1tww}

# RL prompt source: data_gen_*.json graph/task files — the SAME n30 corpus
# e13f trained on (user directive: same dataset, known to work).
DATA_ROOT=${DATA_ROOT:-$PROJ/data/n_30/gen/nav100_n30_gemma_data/split}
TRAIN_GRAPHS=${TRAIN_GRAPHS:-$DATA_ROOT/train_graphs}

MAX_STEPS=${MAX_STEPS:-300}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
MAX_COMPLETION=${MAX_COMPLETION:-1024}
GRAD_ACCUM=${GRAD_ACCUM:-8}
ROLLOUT_BATCH=${ROLLOUT_BATCH:-16}
STRUCTURAL_LR_MULT=${STRUCTURAL_LR_MULT:-1}
POLICY_BIT4=${POLICY_BIT4:-true}
# Space-separated extra hydra overrides, e.g. "+trainer.rl.grpo.temperature=1.15".
# Expanded unquoted on purpose so each override becomes its own argv entry.
EXTRA_OVERRIDES=${EXTRA_OVERRIDES:-}

RUN_NAME=${1:-${RUN_NAME:-e16_rl_mask_e13f}}
# LOCKED campaign tag (see AGENTS.md memory: never invent/change experiment tags).
TAG=e16_rl_training

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"

source "$ENV_DIR/bin/activate"

echo "[$(date --iso-8601=seconds)] $RUN_NAME e16 mask RL (init=$INIT_CHECKPOINT, tag=$TAG, bit4=$POLICY_BIT4, mult=$STRUCTURAL_LR_MULT)"

srun python -m prism.training.train_rl \
    --config-name=e16_rl_config \
    model.path="google/gemma-4-31B-it" \
    gnn.structural_lr_mult="$STRUCTURAL_LR_MULT" \
    data.text_edge_list=none \
    data.edge_weights=binary \
    trainer.bit4="$POLICY_BIT4" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.per_device_train_batch_size=1 \
    trainer.gradient_accumulation_steps="$GRAD_ACCUM" \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.rl.init_checkpoint="$INIT_CHECKPOINT" \
    trainer.rl.data_files="$TRAIN_GRAPHS" \
    trainer.rl.policy_device=0 \
    trainer.rl.num_generations="$NUM_GENERATIONS" \
    trainer.rl.max_completion_length="$MAX_COMPLETION" \
    trainer.rl.rollout_batch_size="$ROLLOUT_BATCH" \
    +trainer.rl.grpo.gradient_checkpointing=true \
    +trainer.rl.grpo.save_steps=50 \
    +trainer.rl.grpo.save_strategy=steps \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG" \
    $EXTRA_OVERRIDES

echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
