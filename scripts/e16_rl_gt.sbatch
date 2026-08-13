#!/bin/bash
#SBATCH --job-name=e16-rl-gt
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=2
# betty's SLURM cli_filter (2026-08-03) enforces EXACTLY 28 CPUs and 224 GB
# per GPU on dgx-b200, and reads --cpus-per-task as a PER-GPU figure — so the
# request must be stated in per-GPU units.
#SBATCH --cpus-per-gpu=28
#SBATCH --mem-per-gpu=224G
#
# e16 — GRPO RL training of the ADDITIVE GT architecture (gt_llm) with
# graph-conditioned vLLM rollouts, v2 semantics: the Ψ tower is UNFROZEN and
# trains on the RL reward (the point of e16), loss-side Ψ is rebuilt through
# the live tower each micro-batch, and rollout transports are invalidated
# every optimizer step. See docs/2026-08-11 e16_rl_poc_implementation_note.md.
#
# Topology: 2 GPUs — the in-process vLLM rollout engine pins cuda:0 of the
# visible set, the policy trains on cuda:1 (trainer.rl.policy_device=1 +
# ACCELERATE_TORCH_DEVICE, same split validated on plaza).
#
# Precision: B200s (180 GB) fit the 31B policy AND engine in bf16 — no nf4
# anywhere, unlike the 48 GB plaza cards. BIT4=true restores the plaza recipe
# (nf4 policy + in-flight bnb nf4 engine) for smaller cards.
#
# Tower LR: gnn.structural_lr_mult defaults to 20 so the tower trains at its
# SFT-native 2e-4 while the LoRA stays at the RL-safe base 1e-5 (the measured
# gate drift at mult=1 — ~5e-6/step — would leave the graph channel inert for
# an entire 300-step run). STRUCTURAL_LR_MULT=1 reverts.
#
# The wandb tag is LOCKED as e16_rl_training — never drift. RUN_NAME is the
# only identity knob a rerun needs.
#
# Submit:  sbatch scripts/e16_rl_gt.sbatch [RUN_NAME]
#   env  :  RUN_NAME=... MAX_STEPS=... BIT4=true DATA_ROOT=... \
#             sbatch scripts/e16_rl_gt.sbatch
# Local :  bash scripts/e16_rl_gt.sbatch [RUN_NAME]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
ENV_DIR=${ENV_DIR:-$ALELAB/envs/GREP-PRISM-rl}
CHECKPOINT_DIR=$PROJ/outputs/e16_rl_training

# RL prompt source: data_gen_*.json graph/task files (NOT the formatted SFT
# json — the RL dataset builds prompts itself and needs scene_graph_dict).
DATA_ROOT=${DATA_ROOT:-data/n_30/gen/nav100_n30_gemma_data/split}
TRAIN_GRAPHS=${TRAIN_GRAPHS:-$DATA_ROOT/train_graphs}

MAX_STEPS=${MAX_STEPS:-300}
NUM_GENERATIONS=${NUM_GENERATIONS:-8}
MAX_COMPLETION=${MAX_COMPLETION:-1024}
GRAD_ACCUM=${GRAD_ACCUM:-8}
STRUCTURAL_LR_MULT=${STRUCTURAL_LR_MULT:-20}
BIT4=${BIT4:-false}

RUN_NAME=${1:-${RUN_NAME:-e16_rl_gt_v2}}
# LOCKED campaign tag (see AGENTS.md memory: never invent/change experiment tags).
TAG=e16_rl_training

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-56}"
# Policy on the second GPU; the in-process vLLM worker owns cuda:0.
export ACCELERATE_TORCH_DEVICE=cuda:1
# On B200 (sm100) vLLM reaches for flashinfer kernels that are JIT-compiled
# with nvcc — betty compute nodes carry no CUDA toolkit, so those paths die at
# engine start. Two switches: FLASH_ATTN is the precompiled attention backend
# every parity test validated (also what plaza runs), and the flashinfer
# SAMPLER (used on sm100 regardless of attention backend) must be disabled in
# favor of the torch-native sampler plaza sampled with.
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}
export VLLM_USE_FLASHINFER_SAMPLER=0

source "$ENV_DIR/bin/activate"

# Engine kwargs: bf16 always; bnb in-flight nf4 only under BIT4 (48 GB cards).
ENGINE_ARGS=(+trainer.rl.engine.dtype=bfloat16
             +trainer.rl.engine.max_model_len=8192
             +trainer.rl.engine.gpu_memory_utilization=0.85)
if [ "$BIT4" = "true" ]; then
  ENGINE_ARGS+=(+trainer.rl.engine.quantization=bitsandbytes
                +trainer.rl.engine.load_format=bitsandbytes)
fi

echo "[$(date --iso-8601=seconds)] $RUN_NAME e16 RL (v2 unfrozen tower, tag=$TAG, bit4=$BIT4, mult=$STRUCTURAL_LR_MULT)"

srun python -m prism.training.train_rl \
    --config-name=e16_rl_config \
    model.path="google/gemma-4-31B-it" \
    gnn.arch=gt_llm \
    gnn.pe_node_features=word_embeddings \
    gnn.pe_gt_from=null \
    gnn.structural_lr_mult="$STRUCTURAL_LR_MULT" \
    data.text_edge_list=none \
    data.edge_weights=binary \
    trainer.bit4="$BIT4" \
    trainer.max_steps="$MAX_STEPS" \
    trainer.per_device_train_batch_size=1 \
    trainer.gradient_accumulation_steps="$GRAD_ACCUM" \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.rl.data_files="$TRAIN_GRAPHS" \
    trainer.rl.policy_device=1 \
    trainer.rl.num_generations="$NUM_GENERATIONS" \
    trainer.rl.max_completion_length="$MAX_COMPLETION" \
    +trainer.rl.grpo.gradient_checkpointing=true \
    +trainer.rl.grpo.save_steps=50 \
    +trainer.rl.grpo.save_strategy=steps \
    "${ENGINE_ARGS[@]}" \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG"

echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
