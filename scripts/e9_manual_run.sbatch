#!/bin/bash
#SBATCH --job-name=e9-multistage
#SBATCH --output=/vast/home/j/jporras/sourcecode/GREP/slurm-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G

# Manual e9 because I'm sick of reading AI Slop.
#
# Single sequential job: stage1 -> stage2 -> stage3(no_edge_list) -> stage3(with_edge_list).
# Each stage chains off the checkpoint the PREVIOUS stage actually produced. The run dir is
# {checkpoint_dir}/{save_name}_{wandb_run_id} and wandb_run_id is random, so we resolve the
# real path at runtime instead of hardcoding it.
BASE_CONFIG=e9_base_config
CHECKPOINT_STAGE1_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1"
CHECKPOINT_STAGE2_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage2"
CHECKPOINT_STAGE3_DIR="$ALELAB_DRIVE/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage3"

RUN_STAGE1=${RUN_STAGE1:-0}
RUN_STAGE2=${RUN_STAGE2:-1}
RUN_STAGE3=${RUN_STAGE3:-1}

# Base LLM for all stages.
MODEL_PATH="google/gemma-4-31B-it"

# Loading env stuff
module load anaconda3 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3

# Resolve the run dir a stage produced: {checkpoint_dir}/{save_name}_{wandb_run_id}. The
# wandb_run_id is random, so pick the most-recently-modified matching subdir. Final artifacts
# (LoRA adapter + gnn_weights.pt + tokenizer) are saved to this run root by trainer.save_model().
resolve_run_dir() {
    local parent="$1" save_name="$2" dir
    dir=$(ls -dt "$parent/${save_name}_"*/ 2>/dev/null | head -n1)
    if [[ -z "$dir" ]]; then
        echo "ERROR: no run dir found under $parent matching ${save_name}_*" >&2
        exit 1
    fi
    echo "${dir%/}"
}

# rpearl_gt_llm architecture overrides shared by every stage.
GNN_ARGS=(
    model.path="$MODEL_PATH"
    gnn.arch=rpearl_gt_llm
    gnn.d_model=2048
    gnn.dropout=0.05
    gnn.pe_hidden_channels=256
    gnn.pe_num_layers=2
    gnn.num_samples=40
    gnn.k_pe=2
    gnn.gt_num_layers=5
    gnn.gt_heads=8
    gnn.k_gt=2
    # Eval on the WHOLE held-out set (all 10 graphs), not the num_graphs=5 default.
    eval.data=data/revised/gen/nav100_n30_gemma_data/split/test_graphs
    eval.num_graphs=-1
)

# Stage 1 - SFT LLM for 0.5 epochs with usual LR.
if [[ "$RUN_STAGE1" == "1" ]]; then
    srun python -m prism.training.train_v3 \
        --config-name="$BASE_CONFIG" \
        "${GNN_ARGS[@]}" \
        trainer.save_name="e9_ms_stage1" \
        wandb.run_name="e9_ms_stage1" \
        trainer.checkpoint_dir="$CHECKPOINT_STAGE1_DIR" \
        trainer.freeze_pe=true \
        trainer.learning_rate=0.0002 \
        trainer.epochs=1
fi

# Stage 2 - PE-only on edge-list loss with LLM+adapter frozen. High LR.
# Inits the (frozen) LoRA adapter from the stage-1 run that just finished.
if [[ "$RUN_STAGE2" == "1" ]]; then
    STAGE1_RUN_DIR=$(resolve_run_dir "$CHECKPOINT_STAGE1_DIR" "e9_ms_stage1")
    echo "[chain] stage2 init_lora_from=$STAGE1_RUN_DIR"
    srun python -m prism.training.train_v3 \
        --config-name="$BASE_CONFIG" \
        "${GNN_ARGS[@]}" \
        trainer.save_name="e9_ms_stage2" \
        wandb.run_name="e9_ms_stage2" \
        trainer.init_lora_from="$STAGE1_RUN_DIR" \
        trainer.checkpoint_dir="$CHECKPOINT_STAGE2_DIR" \
        trainer.freeze_lora=true \
        trainer.freeze_pe=false \
        trainer.loss_target=edge_list \
        trainer.learning_rate=0.005 \
        trainer.epochs=2
fi

# Stage 3 - joint PE + LoRA, edges removed. Small LR for both, smaller for PE.
# Both variants run sequentially and init from the stage-2 run that just finished.
#   no_edge_list   = connectivity must come from the graph PE
#   with_edge_list = edge list kept in the prompt
if [[ "$RUN_STAGE3" == "1" ]]; then
    STAGE2_RUN_DIR=$(resolve_run_dir "$CHECKPOINT_STAGE2_DIR" "e9_ms_stage2")
    echo "[chain] stage3 init_lora_from/init_pe_from=$STAGE2_RUN_DIR"
    for VARIANT_SPEC in "no_edge_list:none" "with_edge_list:present"; do
        VARIANT="${VARIANT_SPEC%%:*}"
        TEXT_EDGE_LIST="${VARIANT_SPEC##*:}"
        srun python -m prism.training.train_v3 \
            --config-name="$BASE_CONFIG" \
            "${GNN_ARGS[@]}" \
            trainer.save_name="e9_ms_stage3__${VARIANT}" \
            wandb.run_name="e9_ms_stage3__${VARIANT}" \
            trainer.init_lora_from="$STAGE2_RUN_DIR" \
            trainer.init_pe_from="$STAGE2_RUN_DIR" \
            trainer.checkpoint_dir="$CHECKPOINT_STAGE3_DIR" \
            trainer.freeze_lora=false \
            trainer.freeze_pe=false \
            data.text_edge_list="$TEXT_EDGE_LIST" \
            trainer.learning_rate=0.00025 \
            trainer.epochs=3
    done
fi
