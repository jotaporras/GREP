#!/bin/bash
#SBATCH --job-name=e11-inj-scope
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-1
#
# e11: train/inference injection alignment (Step 2a of the no-edge-list investigation).
# Retrain no-edge arms with data.injection_scope=prompt_only — the graph channel is
# attached ONLY to prompt node mentions during training, exactly matching generation
# (decode steps receive no injection). Each arm changes ONLY injection_scope relative
# to an existing full_sequence run, giving a clean A/B:
#
#   idx  arm                          paired historical run (full_sequence)
#   0    gmask_noedges_promptonly     rv_gmask_noedges_ror8gtet   (gen acc 0.13)
#   1    integ_rpe_noedges_promptonly e10_integ_rpe__no_edge_list_tyvhwlmx (gen acc 0.14)
#
# Arm 0 is the parameter-free adjacency mask (no cold-start confound). Arm 1 is the
# method's best shot: learnable relative-PE mask, GT warm-started from the pretrained
# edge detector, LoRA warm-started from stage 1.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
CHECKPOINT_DIR=$ALELAB/GREP-PRISM/outputs/e11_injection_scope
TEST_GRAPHS=data/revised/gen/nav100_n30_gemma_data/split/test_graphs

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
# Home dir sits at its quota and its model caches were purged; the hub cache
# lives on vast (shared across jobs, downloads once), datasets cache on
# node-local scratch.
export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

module load anaconda3
source activate "$ENV_NAME"

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
echo "[$(date --iso-8601=seconds)] e11 task ${TASK_ID}"

if [ "$TASK_ID" -eq 0 ]; then
    # Parameter-free adjacency mask, no edges in text; identical to
    # refactor_verify/rv_gmask_noedges except injection_scope.
    NAME=e11_gmask_noedges_promptonly
    srun python -m prism.training.train_v3 \
        --config-name=refactor_verify/rv_gmask_noedges \
        data.injection_scope=prompt_only \
        trainer.save_name="$NAME" \
        trainer.checkpoint_dir="$CHECKPOINT_DIR" \
        wandb.run_name="$NAME" \
        wandb.tag=e11_injection_scope \
        eval.data="$TEST_GRAPHS" \
        eval.num_graphs=-1
else
    # Learnable relative-PE mask with pretrained GT + stage-1 LoRA; identical to
    # scripts/dev_e10_integrate_rpretrained_rpe.sbatch no_edge_list arm except
    # injection_scope. GT hparams MUST match the edge detector (strict load).
    NAME=e11_integ_rpe_noedges_promptonly
    LLM_CHECKPOINT="$ALELAB/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100"
    PE_INIT_DIR="$ALELAB/GREP-PRISM/outputs/dev_e10/pe_init_edge_detector_gt"
    srun python -m prism.training.train_v3 \
        --config-name=e9_base_config \
        model.path="google/gemma-4-31B-it" \
        gnn.arch=learnable_graph_mask \
        gnn.d_model=1024 \
        gnn.dropout=0.1 \
        gnn.pe_hidden_channels=256 \
        gnn.pe_num_layers=5 \
        gnn.num_samples=320 \
        gnn.k_pe=3 \
        gnn.gt_num_layers=3 \
        gnn.gt_heads=8 \
        gnn.k_gt=2 \
        gnn.eps=1e-6 \
        gnn.use_layer_norm=true \
        gnn.pe_node_features=random \
        gnn.mask_alpha=0.7 \
        gnn.mask_layer_scope=dense \
        gnn.mask_psi_scale=cosine \
        gnn.mask_k_hops=1 \
        gnn.mask_symmetrize=true \
        gnn.mask_use_edges=true \
        gnn.structural_lr_mult=5.0 \
        data.text_edge_list=none \
        data.injection_scope=prompt_only \
        trainer.init_lora_from="$LLM_CHECKPOINT" \
        trainer.init_pe_from="$PE_INIT_DIR" \
        trainer.freeze_lora=false \
        trainer.freeze_pe=false \
        trainer.learning_rate=0.00025 \
        trainer.epochs=3 \
        trainer.save_name="$NAME" \
        trainer.checkpoint_dir="$CHECKPOINT_DIR" \
        wandb.run_name="$NAME" \
        wandb.tag=e11_injection_scope \
        eval.data="$TEST_GRAPHS" \
        eval.num_graphs=-1
fi
