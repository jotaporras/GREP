#!/bin/bash
#SBATCH --job-name=e13c-decode
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%A_%a.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --array=0-1
#
# e13c: decode-consistent injection (train WITH the decode-realizable answer-side
# wiring, generate WITH decode-time injection). Gate evidence: the decode-style
# diagnostic (notebooks/2026-07-21 e13b) — decode-realizable wiring retains ~97%
# of the leak circuit's decision channel (0.840 vs 0.865), while leak-free
# prompt-side training reaches 0.82. Question here: do the two channels STACK
# when trained together, i.e. does decision quality (and free-gen) clear the
# ~0.83 convergence level?
#
# Single-flag A/Bs vs the e11 leak-free arms (identical configs, ONLY
# data.injection_scope changes prompt_only -> decode_consistent; both paired runs
# used mask_layer_scope=dense already):
#
#   idx  arm                paired e11 run (prompt_only)              free-gen
#   0    e13c_rpe_decode    e11_integ_rpe_noedges_promptonly_22lq43i6  0.48
#   1    e13c_gmask_decode  e11_gmask_noedges_promptonly_bznw3x9p      0.39
#
# Eval: in-config full 10-graph generation eval; the eval client arms
# MaskDecodeInjector automatically (injection_scope recorded in train_config.json
# and threaded through EvalCallback / evaluate / inference).

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

TASK_ID=${SLURM_ARRAY_TASK_ID:-${1:-0}}
echo "[$(date --iso-8601=seconds)] e13c task ${TASK_ID}"

if [ "$TASK_ID" -eq 0 ]; then
    # Learnable relative-PE mask; identical to e11 task 1 except injection_scope.
    NAME=e13c_rpe_decode
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
        trainer.learning_rate=0.00025 trainer.epochs=3 \
        +trainer.sft.save_total_limit=1 \
        trainer.checkpoint_dir="$CHECKPOINT_DIR" \
        trainer.save_name="$NAME" \
        wandb.run_name="$NAME" wandb.tag=e13c_decode_consistent \
        eval.data="$TEST_GRAPHS" eval.num_graphs=-1
else
    # Parameter-free adjacency mask; identical to e11 task 0 except injection_scope.
    NAME=e13c_gmask_decode
    srun python -m prism.training.train_v3 \
        --config-name=refactor_verify/rv_gmask_noedges \
        data.injection_scope=decode_consistent \
        +trainer.sft.save_total_limit=1 \
        trainer.save_name="$NAME" \
        trainer.checkpoint_dir="$CHECKPOINT_DIR" \
        wandb.run_name="$NAME" wandb.tag=e13c_decode_consistent \
        eval.data="$TEST_GRAPHS" \
        eval.num_graphs=-1
fi

echo "[$(date --iso-8601=seconds)] e13c task ${TASK_ID} DONE: $NAME"
