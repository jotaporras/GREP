#!/bin/bash
#SBATCH --job-name=e9-multistage
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G

# This runs 2-epoch gemma 31B baseline after the refactors to have as a reference vs multistage.
# We expect this to be close to 100%
BASE_CONFIG=e9_base_config
CHECKPOINT_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/e9_llm_baseline"

# Base LLM for all stages.
MODEL_PATH="google/gemma-4-31B-it"

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3

srun python -m prism.training.train_v3 \
    --config-name="$BASE_CONFIG" \
    trainer.save_name="e9_llm_baseline__gemma31b" \
    wandb.run_name="e9_llm_baseline__gemma31b" \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    model.path="$MODEL_PATH" \
    gnn.arch=llm \
    data.text_edge_list=present \
    trainer.freeze_pe=true \
    trainer.freeze_lora=false \
    triner.loss_target=assistant_only \
    trainer.learning_rate=0.0002 \
    trainer.epochs=2