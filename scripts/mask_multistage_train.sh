#!/usr/bin/env bash
# e11 injection-scope run — LearnableGraphMaskLLM on google/gemma-4-31B-it, GT-only mask Ψ.
#
# Runs the Stage-3 joint step: warm-starts a Stage-1 LoRA adapter (trained HERE first if
# TRAIN_STAGE_ONE=true, else resolved by glob search), loads the suite navigator's pretrained GT as
# the STANDALONE mask Ψ producer via gnn.pe_gt_from (no AGT/Semantic GT), trains PE+LoRA jointly
# with edges removed (text_edge_list=none) under prompt-only Ψ injection, then the in-config eval.
#
# The ONLY required input is the location of the pretrained PE model (the GT); everything else is
# resolved from env vars / glob SEARCH (exactly as before) and LOGGED below for verification:
#   * pretrained PE  — positional $1 (REQUIRED): the path_navigator_gt.pt file, or a dir holding it
#   * Stage-1 LoRA   — INIT_LORA env, else newest gt_pe_stage1_* run (+ its newest checkpoint-N)
#                      under $STAGE1_BASE (default $OUTPUTS_ROOT/gt_pe_multistage), via
#                      resolve_run_dir/resolve_checkpoint (as before)
#   * train Stage 1  — TRAIN_STAGE_ONE env (default false): true trains the Stage-1 SFT warm-up
#                      first (on $MODEL) into $STAGE1_BASE, then Stage 3 warm-starts from it
#   * base model     — MODEL env (default google/gemma-4-31B-it): drives BOTH stages so the LoRA
#                      adapter can never drift out of base-model compatibility
#   * output dir     — FROM experiments (trainer.checkpoint_dir in base_config); CHECKPOINT_DIR env overrides
#   * wandb tag      — FROM experiments (wandb.tag in e9_base_config = the multistage-loop grouping)
#   * wandb run_name — this run's stage + edge condition: e9_ms_stage3_{no_edge_list|with_edge_list}
#   * roots          — OUTPUTS_ROOT env (default repo-relative ./outputs; set to /vast/… on cluster)
# Only wandb names/tags and the output path are sourced from experiments/ — never a hyperparameter
# (every gnn.*/data.*/trainer.* hyperparameter is pinned on the CLI below to the pasted values).
#
# Two deliberate reconciliations vs. the pasted e11 param set:
#   * GT init — <suite>/path_navigator_gt.pt via gnn.pe_gt_from (a RAW state_dict → the GT-only
#     path). The pasted trainer.init_pe_from=<dev_e10 edge detector> is DROPPED: with pe_gt_from
#     set it would overwrite the suite GT (train_v3 loads pe_gt_from first, then init_pe_from).
#   * GT LR — forced to 3e-6 = trainer.learning_rate(2.5e-4) × gnn.structural_lr_mult(0.012); the
#     pasted structural_lr_mult=5.0 (=1.25e-3, a boost) is OVERRIDDEN to honor the 3e-6 GT LR.
#
# GT/PE hyperparameters below shape-match <suite>/path_navigator_gt.pt (verified: 72-key exact —
# d_model=1024, pe_hidden_channels=256, pe_num_layers=5, k_pe=3, gt_num_layers=3). The GT-only
# load in load_navigator_pe_into is fail-loud: any missing/unexpected key or size mismatch raises.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# --- sole required input: the location of the pretrained PE model (the GT) ---
# Accept either the path_navigator_gt.pt file directly, or a directory containing it.
PE_ARG="${1:?usage: $0 <pretrained_PE_model_path | dir_containing_path_navigator_gt.pt>  (all else resolved from env/search)}"
if [ -d "$PE_ARG" ]; then PE_GT="$PE_ARG/path_navigator_gt.pt"; else PE_GT="$PE_ARG"; fi
[ -f "$PE_GT" ] || { echo "ERROR: pretrained PE model not found: $PE_GT" >&2; exit 1; }

# --- everything else resolved from env vars / glob search (exactly as before) ---
OUTPUTS_ROOT="${OUTPUTS_ROOT:-outputs}"
STAGE1_BASE="${STAGE1_BASE:-$OUTPUTS_ROOT/gt_pe_multistage}"

# Base model — single source of truth for BOTH stages, so a Stage-1 adapter (base-model-specific)
# always loads into the same base at Stage 3. MODEL env overrides; default is the pinned base.
MODEL="${MODEL:-google/gemma-4-31B-it}"

# Shared gnn.* args: the SAME mask Ψ producer + knobs for Stage 1 and Stage 3, so the Stage-1
# warm-up adapts to the exact attention mask Stage 3 uses. structural_lr_mult is Stage-3-only
# (Stage 1 freezes the PE), so it is NOT in here.
GNN_ARGS=(
    gnn.arch=learnable_graph_mask
    ++gnn.pe_gt_from="$PE_GT"
    gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5
    gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2
    gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random
    gnn.mask_alpha=0.7 gnn.mask_layer_scope=dense gnn.mask_psi_scale=cosine
    gnn.mask_k_hops=1 gnn.mask_symmetrize=true gnn.mask_use_edges=true
)

# Output dir + wandb name/tag come FROM the experiments/ configs (no hard-coded e11 values, and no
# hyperparameter is sourced from there). Output path is read from experiments/base_config.yaml
# (trainer.checkpoint_dir); wandb run_name/tag are inherited from e9_base_config at run time and
# read here only for the verification log. CHECKPOINT_DIR env still overrides the output dir.
yaml_val() { sed -n "s/^[[:space:]]*$2:[[:space:]]*//p" "$1" | head -n1; }
CFG_OUT=$(yaml_val experiments/base_config.yaml checkpoint_dir)
[ -n "$CFG_OUT" ] || { echo "ERROR: could not read trainer.checkpoint_dir from experiments/base_config.yaml" >&2; exit 1; }
OUT="${CHECKPOINT_DIR:-$CFG_OUT}"
WANDB_TAG=$(yaml_val experiments/e9_base_config.yaml tag)   # multistage loop grouping (inherited)

# WandB run NAME = this run's stage in the multistage loop + its edge-list condition, derived from
# data.text_edge_list so the display name always matches the run (a name, never a hyperparameter):
#   e9_ms_stage3_{no_edge_list|with_edge_list}. Stage 3 = the joint PE+LoRA stage this script runs.
MS_STAGE=3
TEXT_EDGE_LIST=none
case "$TEXT_EDGE_LIST" in
    none)    EDGE_TAG=no_edge_list ;;
    present) EDGE_TAG=with_edge_list ;;
    *)       EDGE_TAG="${TEXT_EDGE_LIST}_edge_list" ;;
esac
WANDB_RUN_NAME="e9_ms_stage${MS_STAGE}_${EDGE_TAG}"

# Newest matching run dir by mtime (trailing slash stripped); empty if none.
resolve_run_dir()   { ls -dt "$1/${2}_"*/    2>/dev/null | head -n1 | sed 's:/*$::' || true; }
# Newest checkpoint-N under a run dir; fall back to the run dir itself (adapter at its root).
resolve_checkpoint() { local c; c=$(ls -dt "$1/checkpoint-"*/ 2>/dev/null | head -n1 | sed 's:/*$::' || true); echo "${c:-$1}"; }

# --- optional Stage-1 SFT warm-up (default OFF) ---
# TRAIN_STAGE_ONE=true trains the Stage-1 LoRA first: PE frozen (e9_ms_stage1 sets freeze_pe=true),
# edges IN text (text_edge_list=present), same $MODEL + GNN_ARGS as Stage 3, saved under
# $STAGE1_BASE as gt_pe_stage1 so the search below picks it up. Default false → reuse an existing
# gt_pe_stage1_* adapter. A failed Stage 1 aborts the script (set -e) before Stage 3 runs.
TRAIN_STAGE_ONE="${TRAIN_STAGE_ONE:-false}"
if [ "$TRAIN_STAGE_ONE" = "true" ]; then
    echo "[stage1] TRAIN_STAGE_ONE=true — training Stage-1 SFT warm-up ($MODEL) into $STAGE1_BASE"
    uv run -m prism.training.train_v3 --config-name=e9_ms_stage1 \
        model.path="$MODEL" "${GNN_ARGS[@]}" \
        data.text_edge_list=present data.injection_scope=prompt_only \
        trainer.save_name=gt_pe_stage1 trainer.checkpoint_dir="$STAGE1_BASE" \
        wandb.run_name=e9_ms_stage1_sft wandb.tag=gt_pe_multistage
    INIT_LORA=""   # discard any pinned adapter; force the search to use the run just trained
fi

# Stage-1 adapter: explicit INIT_LORA env wins; otherwise search (like the original resolve_run_dir).
if [ -z "${INIT_LORA:-}" ]; then
    S1_RUN=$(resolve_run_dir "$STAGE1_BASE" gt_pe_stage1)
    [ -n "$S1_RUN" ] || { echo "ERROR: no gt_pe_stage1_* run under $STAGE1_BASE (set INIT_LORA or STAGE1_BASE)" >&2; exit 1; }
    INIT_LORA=$(resolve_checkpoint "$S1_RUN")
    S1_SRC="search: newest gt_pe_stage1_* under $STAGE1_BASE"
else
    S1_RUN="(not searched — INIT_LORA env override)"
    S1_SRC="env: INIT_LORA"
fi
[ -d "$INIT_LORA" ] || { echo "ERROR: Stage-1 LoRA checkpoint not found: $INIT_LORA" >&2; exit 1; }

# --- log every acquired path for verification ---
echo "[resolve] pretrained PE (GT)  : $PE_GT   (from arg: $PE_ARG)"
echo "[resolve] base model (MODEL)  : $MODEL"
echo "[resolve] train stage 1       : $TRAIN_STAGE_ONE"
echo "[resolve] OUTPUTS_ROOT        : $OUTPUTS_ROOT"
echo "[resolve] Stage-1 search base : $STAGE1_BASE"
echo "[resolve] Stage-1 run dir     : $S1_RUN"
echo "[resolve] Stage-1 LoRA ckpt   : $INIT_LORA   ($S1_SRC)"
echo "[resolve] output dir          : $OUT   ($( [ -n "${CHECKPOINT_DIR:-}" ] && echo 'env: CHECKPOINT_DIR' || echo 'experiments/base_config.yaml: checkpoint_dir' ))"
echo "[resolve] wandb run_name/tag  : $WANDB_RUN_NAME / $WANDB_TAG   (name=stage${MS_STAGE}+${EDGE_TAG}; tag from experiments/e9_base_config.yaml)"

# Stage 3 (joint PE+LoRA): same $MODEL + GNN_ARGS as Stage 1, plus the Stage-3-only knobs
# (structural_lr_mult, no edges, trainable PE, warm-started Stage-1 adapter).
uv run -m prism.training.train_v3 --config-name=e9_base_config \
    model.path="$MODEL" \
    "${GNN_ARGS[@]}" \
    gnn.structural_lr_mult=0.012 \
    data.text_edge_list="$TEXT_EDGE_LIST" data.injection_scope=prompt_only \
    trainer.init_lora_from="$INIT_LORA" \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 \
    trainer.checkpoint_dir="$OUT" \
    wandb.run_name="$WANDB_RUN_NAME" \
    eval.num_graphs=-1
