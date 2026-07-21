#!/bin/bash
#SBATCH --job-name=e13-smoke
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A.out
#SBATCH --time=03:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#
# e13 SMOKE: end-to-end wiring check of the NavigatorPE integration on the cluster,
# BEFORE the real arms. Trains 4 optimizer steps of learnable_graph_mask in navigator
# mode (RANDOMINIT stand-in GT + the real AGT — identical code paths, no science),
# runs the in-config 1-graph eval, then re-evaluates FROM DISK via
# scalability_evaluation to exercise the checkpoint-reload navigator branch
# (loaders.graph_augmented_llm_from_pretrained + train_config.json pe_gt_from keys).
# report_to=none: a smoke is not an experiment.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
CHECKPOINT_DIR=$ALELAB/GREP-PRISM/outputs/e13_nav_pe
PE_DIR=$ALELAB/GREP-PRISM/pretrained_pe
TEST_GRAPHS=data/revised/gen/nav100_n30_gemma_data/split/test_graphs

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

module load anaconda3
source activate "$ENV_NAME"

NAME=e13_smoke_navigator
srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    gnn.arch=learnable_graph_mask \
    gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5 \
    gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2 \
    gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random \
    gnn.mask_alpha=0.7 gnn.mask_layer_scope=dense gnn.mask_psi_scale=cosine \
    gnn.mask_k_hops=1 gnn.mask_symmetrize=true gnn.mask_use_edges=true \
    gnn.pe_gt_from="$PE_DIR/RANDOMINIT_navigator_gt.pt" \
    gnn.semantic_gt_from="$PE_DIR/path_navigator_agt.pt" \
    gnn.structural_lr_mult=0.012 \
    data.text_edge_list=none data.injection_scope=prompt_only \
    trainer.init_lora_from="$ALELAB/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100" \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.max_steps=4 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$NAME" \
    trainer.report_to=none \
    eval.data="$TEST_GRAPHS" eval.num_graphs=1 \
    wandb.run_name="$NAME" wandb.tag=e13_nav_pe

RUN_DIR=$(ls -dt "$CHECKPOINT_DIR/${NAME}"*/ | head -n1 | sed 's:/*$::')
echo "[smoke] run dir: $RUN_DIR"

# From-disk reload check on ONE graph (copy it to scratch so --graphs sees only one).
ONE_GRAPH_DIR="${TMPDIR:-/tmp}/e13_smoke_one_graph_${SLURM_JOB_ID:-manual}"
mkdir -p "$ONE_GRAPH_DIR"
cp "$(ls $TEST_GRAPHS/*.json | head -n1)" "$ONE_GRAPH_DIR/"
srun python -m prism.eval.scalability_evaluation \
    --checkpoint "$RUN_DIR" \
    --graphs "$ONE_GRAPH_DIR" \
    --device 0

echo "[smoke] PASS: train + save + in-config eval + from-disk reload eval all ran"
