#!/bin/bash
#SBATCH --job-name=e9-multistage
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%A_%a.out
    #SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-1

# Manual e9 because I'm sick of reading AI Slop.

# Multistage requires training with the next stages's weights and some config changes.
# So let's have a base cofig.
BASE_CONFIG=e9_base_config
# TODO make sure they get created
#CHECKPOINT_STAGE1_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1"
CHECKPOINT_STAGE1_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/refactor_verify/rv_rpearl_edges_hdkca8ed"
CHECKPOINT_STAGE2_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage2"
# Stage 3 inits from this specific stage-2 epoch checkpoint (not the run root).
STAGE3_INIT_FROM="$CHECKPOINT_STAGE2_DIR/e9_ms_stage2_wd9gbmoj/checkpoint-200"
CHECKPOINT_STAGE3_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage3"

RUN_STAGE1=${RUN_STAGE1:-0}
RUN_STAGE2=${RUN_STAGE2:-0}
RUN_STAGE3=${RUN_STAGE3:-1}

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3

# Stage 1 - SFT LLM for 0.5 epochs with usual LR.
# Stages 1/2 are single runs; only execute on the first array task to avoid duplication.
if [[ "$RUN_STAGE1" == "1" && "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    srun python -m prism.training.train_v3 \
        --config-name="$BASE_CONFIG" \
        trainer.save_name="e9_ms_stage1" \
        wandb.run_name="e9_ms_stage1" \
        trainer.checkpoint_dir="$CHECKPOINT_STAGE1_DIR" \
        trainer.freeze_pe=true \
        trainer.learning_rate=0.0002 \
        trainer.epochs=0.5
fi

# Stage 2 - PE-only on edge-list loss with LLM+adapter frozen. High LR.
if [[ "$RUN_STAGE2" == "1" && "${SLURM_ARRAY_TASK_ID:-0}" == "0" ]]; then
    srun python -m prism.training.train_v3 \
        --config-name="$BASE_CONFIG" \
        trainer.save_name="e9_ms_stage2" \
        wandb.run_name="e9_ms_stage2" \
        trainer.init_lora_from="$CHECKPOINT_STAGE1_DIR" \
        trainer.checkpoint_dir="$CHECKPOINT_STAGE2_DIR" \
        trainer.freeze_lora=true \
        trainer.freeze_pe=false \
        trainer.loss_target=edge_list \
        trainer.learning_rate=0.005 \
        trainer.epochs=4
fi

# Stage 3 - joint PE + LoRA, edges removed. Small LR for both, smaller for PE.
# Run as a 2-task array: task 0 = no edge list, task 1 = with edge list.

if [[ "$RUN_STAGE3" == "1" ]]; then
    case "$SLURM_ARRAY_TASK_ID" in
        0)
            VARIANT="no_edge_list"
            TEXT_EDGE_LIST="none"
            ;;
        1)
            VARIANT="with_edge_list"
            TEXT_EDGE_LIST="present"
            ;;
        *)
            echo "Unexpected SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID (expected 0 or 1)" >&2
            exit 1
            ;;
    esac

    srun python -m prism.training.train_v3 \
        --config-name="$BASE_CONFIG" \
        trainer.save_name="e9_ms_stage3__${VARIANT}" \
        wandb.run_name="e9_ms_stage3__${VARIANT}" \
        trainer.init_lora_from="$STAGE3_INIT_FROM" \
        trainer.init_pe_from="$STAGE3_INIT_FROM" \
        trainer.checkpoint_dir="$CHECKPOINT_STAGE3_DIR" \
        trainer.freeze_lora=false \
        trainer.freeze_pe=false \
        data.text_edge_list="$TEXT_EDGE_LIST" \
        trainer.learning_rate=0.0001 \
        trainer.epochs=2
fi