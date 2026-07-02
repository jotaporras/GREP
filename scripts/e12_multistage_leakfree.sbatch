#!/bin/bash
#SBATCH --job-name=e12-ms-leakfree
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=256G
#SBATCH --exclude=dgx024
#
# e12: leak-free multistage (the additive-family counterpart of e11).
# Reruns the e9 multistage chain from the SAME stage-1 checkpoint with the graph
# channel never attached to supervised tokens:
#
#   stage 2  loss_target=edge_list  ->  injection_scope=exclude_supervised
#            (e9 leak: the edge-bullet target token's own q/k carried its node's psi
#             — the label in q-space — so reconstruction could shortcut instead of
#             inferring the neighbor from the anchor node's features)
#   stage 3  loss_target=responses ->  injection_scope=prompt_only
#            (e11 fix: answer-side mentions carried the ground-truth node's channel,
#             which decode never provides)
#
# Everything else matches the paired historical chain exactly (resolved from the
# wandb configs of e9_ms_stage2_3bitwckz -> e9_ms_stage3 kfuu2djo/6lefhd76):
# rpearl_gt_llm d2048 L5, stage-1 LoRA e9_ms_stage1_sqgk4o3j, stage-2 lr 5e-3 x4
# epochs frozen-LoRA, stage-3 lr 2.5e-4 x3 epochs joint. Historical no-edge gen
# acc: 0.10 (kfuu2djo). Prediction (lab note §6): stage 3 lands above the
# 0.10-0.16 floor and the final diagnostic shows a generation-compatible NLL gap
# well past ms_stage3's -0.2 nats.
#
# Stage-3 runs BOTH variants (no_edge_list = the result; with_edge_list = the
# prompt_only parity check from the e11 verification debt), then the
# injection-ablation diagnostic on the no-edge checkpoint.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
OUT=$ALELAB/GREP-PRISM/outputs/e12_leakfree_multistage
STAGE1_RUN_DIR=$ALELAB/GREP-PRISM/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j
TEST_GRAPHS=data/revised/gen/nav100_n30_gemma_data/split/test_graphs
VAL_FILE=data/revised/gen/nav100_n30_gemma_data/split/formatted_all_new__val.json

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
# Home dir sits near its quota; hub cache on vast (shared, downloads once),
# datasets cache on node-local scratch.
export HF_HOME=/vast/projects/aribeiro/alelab/jporras/hf_cache
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"

module load anaconda3
source activate "$ENV_NAME"

RUN_STAGE2=${RUN_STAGE2:-1}
RUN_STAGE3=${RUN_STAGE3:-1}
RUN_DIAG=${RUN_DIAG:-1}

# {checkpoint_dir}/{save_name}_{wandb_run_id}: wandb_run_id is random, resolve the
# most recent matching run dir (same helper as e9_manual_run.sbatch).
resolve_run_dir() {
    local parent="$1" save_name="$2" dir
    dir=$(ls -dt "$parent/${save_name}_"*/ 2>/dev/null | head -n1)
    if [[ -z "$dir" ]]; then
        echo "ERROR: no run dir found under $parent matching ${save_name}_*" >&2
        exit 1
    fi
    echo "${dir%/}"
}

# rpearl_gt_llm overrides shared by every stage — verbatim from e9_manual_run.sbatch
# (matches the wandb gnn config of the paired chain).
GNN_ARGS=(
    model.path="google/gemma-4-31B-it"
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
    # FULL held-out set (all 10 graphs) at every stage — e11 reporting convention.
    # (Paired e9 chain evaluated 5 graphs; superset, so the A/B still reads.)
    eval.data="$TEST_GRAPHS"
    eval.num_graphs=-1
)

# Stage 2 — PE-only edge-list reconstruction, LLM+adapter frozen. ONLY delta vs
# e9_ms_stage2_3bitwckz: injection_scope=exclude_supervised (edge-bullet tokens
# carry no psi; scene-block mentions keep theirs, so every node has a prompt span).
if [[ "$RUN_STAGE2" == "1" ]]; then
    srun python -m prism.training.train_v3 \
        --config-name=e9_base_config \
        "${GNN_ARGS[@]}" \
        trainer.save_name="e12_ms_stage2_leakfree" \
        wandb.run_name="e12_ms_stage2_leakfree" \
        wandb.tag=e12_leakfree_multistage \
        trainer.init_lora_from="$STAGE1_RUN_DIR" \
        trainer.checkpoint_dir="$OUT/stage2" \
        trainer.freeze_lora=true \
        trainer.freeze_pe=false \
        trainer.loss_target=edge_list \
        data.injection_scope=exclude_supervised \
        trainer.learning_rate=0.005 \
        trainer.epochs=4
fi

# Stage 3 — joint PE + LoRA from the stage-2 checkpoint. ONLY delta vs
# kfuu2djo/6lefhd76: injection_scope=prompt_only (plus the stage-2 init above).
if [[ "$RUN_STAGE3" == "1" ]]; then
    STAGE2_RUN_DIR=$(resolve_run_dir "$OUT/stage2" "e12_ms_stage2_leakfree")
    echo "[chain] stage3 init_lora_from/init_pe_from=$STAGE2_RUN_DIR"
    for VARIANT_SPEC in "no_edge_list:none" "with_edge_list:present"; do
        VARIANT="${VARIANT_SPEC%%:*}"
        TEXT_EDGE_LIST="${VARIANT_SPEC##*:}"
        srun python -m prism.training.train_v3 \
            --config-name=e9_base_config \
            "${GNN_ARGS[@]}" \
            trainer.save_name="e12_ms_stage3__${VARIANT}" \
            wandb.run_name="e12_ms_stage3__${VARIANT}" \
            wandb.tag=e12_leakfree_multistage \
            trainer.init_lora_from="$STAGE2_RUN_DIR" \
            trainer.init_pe_from="$STAGE2_RUN_DIR" \
            trainer.checkpoint_dir="$OUT/stage3" \
            trainer.freeze_lora=false \
            trainer.freeze_pe=false \
            data.text_edge_list="$TEXT_EDGE_LIST" \
            data.injection_scope=prompt_only \
            trainer.learning_rate=0.00025 \
            trainer.epochs=3
    done
fi

# Decision-token diagnostic on the no-edge stage-3 checkpoint: measures the
# generation-compatible channel (prompt_only vs no_injection NLL gap) against
# ms_stage3's -0.2 nats.
if [[ "$RUN_DIAG" == "1" ]]; then
    STAGE3_RUN_DIR=$(resolve_run_dir "$OUT/stage3" "e12_ms_stage3__no_edge_list")
    echo "[chain] diagnostic on $STAGE3_RUN_DIR"
    srun python scripts/diag_injection_ablation.py \
        --checkpoint "$STAGE3_RUN_DIR" \
        --val-file "$VAL_FILE"
fi
