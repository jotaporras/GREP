#!/bin/bash
#SBATCH --job-name=e14-stage1to3-binary
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e14 — the e13f_alpha00_binary recipe (pure learned sim on legal edges, mask_alpha=0.0,
# nav-GT Psi x decode_consistent, binary adjacency) run as a DIRECT Stage 1 -> Stage 3
# jump. No Stage 2 anywhere: the LoRA is carried from the Stage-1 SFT adapter and the
# Psi producer comes straight from the pretrained navigator GT.
#
# TWO deliberate differences vs scripts/e13f_alpha00_binary.sbatch:
#
#   1. gnn.pe_gt_from -> outputs/e9_multistage_training/suite8/path_navigator_gt.pt
#      (the suite8 notebook run, the same artifact experiments/e9_ms_stage2.yaml and
#      experiments/e9_navigator_gt.yaml were repointed to). NOT $PROJ/pretrained_pe,
#      which carries no suite provenance. Verified: this state_dict loads into the
#      standalone GraphTransformer built from the gnn.* block below with ZERO
#      missing/unexpected keys, so the strict PE-GT load in
#      loaders.load_navigator_pe_into cannot silently drop weights.
#      Override with NAV_GT=... if the file lives elsewhere on the cluster.
#      gnn.semantic_gt_from=null below is REQUIRED, not incidental: Psi is the
#      navigator's PE stage alone (gt.NavigatorPE), and the Semantic GT is the AGT head
#      that lives in gt.NavigatorGT. A non-null value would select the LEGACY two-stage
#      producer gt.TwoStagePE (Psi = SemanticGT(PE_GT(graph))), which exists only to
#      reload pre-split e13 checkpoints.
#
#   2. NO trainer.init_pe_from — DELIBERATE, and it is what preserves the navigator.
#      train_v3.train_model loads the navigator GT FIRST (load_navigator_pe_into) and
#      then, if init_pe_from is set, load_pe_weights_into OVERWRITES the whole pe_model
#      from that checkpoint's gnn_weights.pt. Pointing init_pe_from at the Stage-1 dir
#      would therefore either (a) throw, because Stage 1 trains arch=rpearl_llm whose
#      pe_model keys do not match a standalone GraphTransformer, or (b) if it did
#      match, discard the suite8 navigator Psi this run exists to use — and Stage 1
#      freezes the PE and gates it off (gnn.pe_gain_init=0), so those weights are
#      untrained noise regardless. Psi therefore comes ONLY from gnn.pe_gt_from, and
#      trainer.freeze_pe=false lets Stage 3 fine-tune it from the navigator init.
#
# The training corpus (data.train_files / data.val_files) and the generation-eval graphs
# (eval.data) all derive from DATA_ROOT below, defaulting to data/n_30/... — the path that
# exists in this repo. The e13f template's data/revised/... is stale and no longer present,
# and base_config.yaml still defaults train_files/val_files to it, so those two MUST be
# passed explicitly or the run dies at dataset load.
#
# DISABLE_GRAPH_TOKEN_ROPE (model.disable_graph_token_rope, default false) is now LIVE for
# this arch: build_planner_model forwards it to LearnableGraphMaskLLM, whose forward zeroes
# the RoPE positions of the injected (query-role) spans. It is orthogonal to the mask — the
# mask says which nodes may attend, this says node tokens carry no sequence position — so
# true and false are two different models, not two labels for one.
# Under data.injection_scope=decode_consistent (below) it covers the prompt spans plus the
# single knowability position of each answer mention, and MaskDecodeInjector re-zeroes the
# same positions at decode, so training and generation stay in agreement.
#
# NOTE the default is false, which is what earlier e14 runs actually COMPUTED (the flag
# reached no code for this arch) even though they recorded true. Keeping false therefore
# preserves comparability with those runs; DISABLE_GRAPH_TOKEN_ROPE=true is a NEW arm.
# The value is now written into train_config.json, so eval rebuilds it correctly — before
# this it was absent and loaders always defaulted to false at eval.
#
# The wandb names are overridable, in priority order:
#   positional arg  >  same-named env var  >  built-in default.
#   RUN_NAME (arg 1) -> wandb.run_name AND trainer.save_name (checkpoint subdir)
#   TAG      (arg 2) -> wandb.tag (run group)
#
# Paths are env-overridable: DATA_ROOT (the training corpus; TRAIN_FILES / VAL_FILES /
# TEST_GRAPHS derive from it and are individually overridable too), STAGE1_LORA (the
# stage-1 LoRA this run continues from) and NAV_GT (the pretrained navigator GT).
# STAGE1_LORA and NAV_GT default to a specific prior run id, so a rerun needs them
# pointed at the new one.
#
# Submit:  sbatch scripts/e14_stage1to3_binary.sbatch [RUN_NAME] [TAG]
#   env  :  RUN_NAME=... TAG=... DATA_ROOT=... STAGE1_LORA=... NAV_GT=... \
#             DISABLE_GRAPH_TOKEN_ROPE=false sbatch scripts/e14_stage1to3_binary.sbatch
# Local :  bash   scripts/e14_stage1to3_binary.sbatch [RUN_NAME] [TAG]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
CHECKPOINT_DIR=$PROJ/outputs/e14_stage1to3
NAV_GT=${NAV_GT:-$PROJ/outputs/e9_multistage_training/suite8/path_navigator_gt.pt}
# Overridable like NAV_GT above: the default names one specific stage-1 run, and a
# rerun (or a different cluster) produces a different run id, so pinning it here
# would make the script silently unusable rather than reconfigurable.
STAGE1_LORA=${STAGE1_LORA:-$PROJ/outputs/e9_multistage_training/e9_ms_stage1/e9_ms_stage1_sqgk4o3j/checkpoint-100}
# TRAINING DATA SOURCE. One root; train/val/test all derive from it, so switching
# corpora cannot leave the generation eval pointing at a different corpus than the SFT.
# This must be set explicitly: base_config.yaml:183-184 still defaults train_files/
# val_files to data/revised/..., which does NOT exist in this repo — inheriting it makes
# the run die at datasets.load_dataset. Only eval.data was ever repointed here.
# Override the root for a different corpus (e.g. DATA_ROOT=data/n_100/gen/nav_n100_gemma_data/split),
# or the individual files for a variant split (e.g. TRAIN_FILES=.../formatted_all_new_2turn__train.json).
DATA_ROOT=${DATA_ROOT:-data/n_30/gen/nav100_n30_gemma_data/split}
TRAIN_FILES=${TRAIN_FILES:-$DATA_ROOT/formatted_all_new__train.json}
VAL_FILES=${VAL_FILES:-$DATA_ROOT/formatted_all_new__val.json}
TEST_GRAPHS=${TEST_GRAPHS:-$DATA_ROOT/test_graphs}
# Few-shot TRAINING: 2 = the SPINE_ICL_EXAMPLES pair, matching eval.use_icl=true on the
# transferability side (base_config.yaml:207 asks for the two to agree, since ICL also
# moves the scene graph out of the system message into the query turn — PE injection
# still scopes to the query graph, which stays last).
#
# CAVEAT, left as a decision rather than flipped silently: base_config.yaml:212 says the
# canned examples call goto/map_region, so icl_examples>0 "should only pair with
# spine_tools=present". This run leaves spine_tools at its default (none), because a
# corpus audit measured ~80% of training targets as [answer(...)]-only with almost no
# tool calls — switching the contract to an action list would ask the model for something
# the data never demonstrates. Set DATA_ICL_EXAMPLES=0 to go back to zero-shot.
DATA_ICL_EXAMPLES=${DATA_ICL_EXAMPLES:-2}
# Identity-RoPE (position_id 0) on the injected node-token spans — a real architectural
# arm for this arch (see the note above), not just recorded metadata.
DISABLE_GRAPH_TOKEN_ROPE=${DISABLE_GRAPH_TOKEN_ROPE:-false}

# wandb names: positional arg > env var > default. RUN_NAME also drives the
# checkpoint subdir (trainer.save_name) so the on-disk run stays identifiable.
RUN_NAME=${1:-${RUN_NAME:-e14_stage1to3_binary}}
TAG=${2:-${TAG:-e14_stage1to3}}

# Home dir sits near its quota; keep the HF hub cache (base gemma-4-31B-it
# weights) on shared vast — downloaded once, reused across jobs.
export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
export HF_HOME=${HF_HOME:-$ALELAB/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
# SLURM 22.05+ stopped propagating --cpus-per-task to srun; pass it explicitly
# to avoid "CPU binding outside of job step allocation" on dgx-b200 nodes.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] $RUN_NAME training (stage1 -> stage3, no stage 2, tag=$TAG)"
echo "  stage-1 LoRA: $STAGE1_LORA"
echo "  navigator GT: $NAV_GT   (Psi init; no init_pe_from, nothing overwrites it)"

srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    model.disable_graph_token_rope="$DISABLE_GRAPH_TOKEN_ROPE" \
    gnn.arch=learnable_graph_mask \
    gnn.d_model=1024 gnn.dropout=0.1 gnn.pe_hidden_channels=256 gnn.pe_num_layers=5 \
    gnn.num_samples=320 gnn.k_pe=3 gnn.gt_num_layers=3 gnn.gt_heads=8 gnn.k_gt=2 \
    gnn.eps=1e-6 gnn.use_layer_norm=true gnn.pe_node_features=random \
    gnn.mask_alpha=0.0 gnn.mask_layer_scope=dense gnn.mask_psi_scale=cosine \
    gnn.mask_k_hops=1 gnn.mask_symmetrize=true gnn.mask_use_edges=true \
    gnn.pe_gt_from="$NAV_GT" \
    gnn.semantic_gt_from=null \
    gnn.structural_lr_mult=0.0012 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.text_edge_list=none \
    data.injection_scope=decode_consistent \
    data.edge_weights=binary \
    data.icl_examples="$DATA_ICL_EXAMPLES" \
    trainer.init_lora_from="$STAGE1_LORA" \
    trainer.freeze_lora=false trainer.freeze_pe=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$RUN_NAME" \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG" \
    eval.data="$TEST_GRAPHS" eval.num_graphs=-1

echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
