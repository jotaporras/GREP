#!/bin/bash
#SBATCH --job-name=e9-multistage
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G

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
RUN_STAGE2=${RUN_STAGE2:-1}
RUN_STAGE3=${RUN_STAGE3:-0}

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3

# Stage 1 - SFT LLM for 0.5 epochs with usual LR.
# TODO add skip eval flag.
if [[ "$RUN_STAGE1" == "1" ]]; then
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

if [[ "$RUN_STAGE2" == "1" ]]; then
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

if [[ "$RUN_STAGE3" == "1" ]]; then
    srun python -m prism.training.train_v3 \
        --config-name="$BASE_CONFIG" \
        trainer.save_name="e9_ms_stage3" \
        wandb.run_name="e9_ms_stage3" \
        trainer.init_lora_from="$STAGE3_INIT_FROM" \
        trainer.init_pe_from="$STAGE3_INIT_FROM" \
        trainer.checkpoint_dir="$CHECKPOINT_STAGE3_DIR" \
        trainer.freeze_lora=false \
        trainer.freeze_pe=false \
        trainer.learning_rate=0.0002 \
        trainer.epochs=1.5
fi