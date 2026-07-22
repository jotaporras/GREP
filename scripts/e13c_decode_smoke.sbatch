#!/bin/bash
#SBATCH --job-name=e13c-smoke
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A.out
#SBATCH --time=03:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#
# e13c SMOKE: wiring check of decode_consistent BEFORE the real arms. Trains 4
# steps of the e13c task-0 configuration (asymmetric q/k maps through the
# collator/trainer), runs the 1-graph in-config generation eval (exercises
# MaskDecodeInjector on real Gemma decode), then re-evaluates FROM DISK via
# scalability_evaluation (exercises resolve_injection_scope + the reload path).
# report_to=none: a smoke is not an experiment.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
CHECKPOINT_DIR=$ALELAB/GREP-PRISM/outputs/e13c_decode_consistent
TEST_GRAPHS=data/revised/gen/nav100_n30_gemma_data/split/test_graphs

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

module load anaconda3
source activate "$ENV_NAME"

NAME=e13c_smoke_decode
LLM_CHECKPOINT="$ALELAB/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100"
PE_INIT_DIR="$ALELAB/GREP-PRISM/outputs/dev_e10/pe_init_edge_detector_gt"
srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    gnn.arch=learnable_graph_mask \
    gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5 \
    gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2 \
    gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random \
    gnn.mask_alpha=0.7 gnn.mask_layer_scope=dense gnn.mask_psi_scale=cosine \
    gnn.mask_k_hops=1 gnn.mask_symmetrize=true gnn.mask_use_edges=true \
    gnn.structural_lr_mult=5.0 \
    data.text_edge_list=none \
    data.injection_scope=decode_consistent \
    trainer.init_lora_from="$LLM_CHECKPOINT" \
    trainer.init_pe_from="$PE_INIT_DIR" \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.max_steps=4 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$NAME" \
    trainer.report_to=none \
    eval.data="$TEST_GRAPHS" eval.num_graphs=1

RUN_DIR=$(ls -dt "$CHECKPOINT_DIR/${NAME}"*/ | head -n1 | sed 's:/*$::')
echo "[smoke] run dir: $RUN_DIR"

ONE_GRAPH_DIR="${TMPDIR:-/tmp}/e13c_smoke_one_graph_${SLURM_JOB_ID:-manual}"
mkdir -p "$ONE_GRAPH_DIR"
cp "$(ls $TEST_GRAPHS/*.json | head -n1)" "$ONE_GRAPH_DIR/"
srun python -m prism.eval.scalability_evaluation \
    --checkpoint "$RUN_DIR" \
    --graphs "$ONE_GRAPH_DIR" \
    --device 0

echo "[smoke] PASS: decode_consistent train + save + gen eval + from-disk reload all ran"
