#!/bin/bash
#SBATCH --job-name=e9-multistage
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%j.out
#SBATCH --time=3:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --array=0-1
#SBATCH --mem=256G

# Dev script to manually run some baselines.
# We expect this to be close to 100%
BASE_CONFIG=e9_base_config
NAME=(e9_llm_baseline_noedges__gemma31b e9_rpearl_llm_edges__gemma31b)
CHECKPOINT_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/$NAME"
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-${TASK_ID:-0}}}

# Base LLM for all stages.
MODEL_PATH="google/gemma-4-31B-it"

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3


if [ "$ARRAY_TASK_ID" -eq 0 ]; then
    srun python -m prism.training.train_v3 \
        --config-name="$BASE_CONFIG" \
        trainer.save_name="$NAME" \
        wandb.run_name="$NAME" \
        trainer.checkpoint_dir="$CHECKPOINT_DIR" \
        model.path="$MODEL_PATH" \
        gnn.arch=llm \
        # gnn.arch=rpearl_llm \
        # gnn.pe_gain_init=0.0 \
        #data.text_edge_list=present \
        data.text_edge_list=none \
        trainer.freeze_lora=false \
        trainer.learning_rate=0.0002 \
        trainer.epochs=2
fi

# RPEARL LLM with edges
if [ "$ARRAY_TASK_ID" -eq 1 ]; then
    srun python -m prism.training.train_v3 \
        --config-name="$BASE_CONFIG" \
        trainer.save_name="$NAME" \
        wandb.run_name="$NAME" \
        trainer.checkpoint_dir="$CHECKPOINT_DIR" \
        model.path="$MODEL_PATH" \
        gnn.arch=rpearl_llm \
        gnn.pe_gain_init=0.0 \
        data.text_edge_list=present \
        trainer.freeze_lora=false \
        trainer.learning_rate=0.0002 \
        trainer.epochs=2
fi