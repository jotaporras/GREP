#!/bin/bash
#SBATCH --job-name=e14-baseline-llm
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e14 BASELINE — the plain-LLM control for scripts/e14_stage1to3_binary.sbatch — same base
# model, same corpus, same prompt format, same optimizer settings, same eval set, same
# SLURM resources; the ONLY intended difference is gnn.arch=llm, i.e. no graph channel at
# all (no Psi, no attention mask, no PE). Checkpoints land in the same CHECKPOINT_DIR and
# the default TAG matches, so the two runs group together in wandb.
#
# THREE FORCED DIFFERENCES vs the e14 script — none of them optional, all load-bearing:
#
#   1. NO trainer.init_lora_from. train_v3.py:145-148 RAISES for a non-graph arch
#      ("init_pe_from / init_lora_from are only supported for graph architectures"), so
#      this baseline trains a FRESH LoRA for 3 epochs instead of continuing the Stage-1
#      SFT adapter that e14 carries. READ THE COMPARISON ACCORDINGLY: e14 has seen the
#      Stage-1 SFT epochs on top of these 3; the baseline has not. Matching that would
#      mean relaxing the guard at train_v3.py:145 to allow init_lora_from for arch=llm
#      (the Stage-1 adapter targets LLM modules only, so PeftModel.from_pretrained would
#      attach cleanly) — a deliberate code change, not something this script assumes.
#      trainer.freeze_lora stays false (base_config.yaml:279); true would additionally
#      raise here, since freeze_lora is implemented via the attach path.
#
#   2. NO gnn.* hyperparameters. architectures.py:173-177 returns the bare LLM plus
#      TokenIndexCollator for arch=llm — d_model / pe_* / gt_* / mask_* / pe_gt_from /
#      structural_lr_mult are all inert, and BaselineSFTTrainer records only
#      architecture / base_model / text_edge_list / spine_tools / icl_examples in
#      train_config.json (train_v3.py:310-325). Passing them would put unread values in
#      the wandb config and imply a graph channel that does not exist.
#      model.disable_graph_token_rope is dropped for the same reason (it is inert even in
#      e14 — build_planner_model forwards it only to the GraphAugmentedLLM variants).
#
#   3. NO data.injection_scope / data.edge_weights. Both describe how the scene graph
#      reaches a GNN; with no GNN they are set on the collator and never read. Dropped so
#      the recorded config cannot be misread as "decode_consistent binary baseline".
#
# TEXT_EDGE_LIST is the one knob you may actually want to flip, and it is left at e14's
# value (none) for exact parity, NOT because it is the only interesting setting:
#   none    (default, = e14) — the no-connectivity FLOOR. The prompt carries node names
#           only, and unlike e14 nothing supplies the edges. This is the number the graph
#           channel has to beat.
#   present — the informed BRACKET: edges written into the prompt text. Pair the two to
#           bracket e14 the way experiments/e9_baseline_llm.yaml and
#           experiments/e9_baseline_llm_no_edges.yaml do.
# Note the eval side follows automatically: train_v3 passes
# include_edge_list=(text_edge_list=="present") to the EvalCallback.
#
# Everything else is copied verbatim from e14: DATA_ROOT-derived train/val/test (the
# e13f template's data/revised/... is stale and base_config.yaml:183-184 still defaults
# to it, so train_files/val_files MUST be passed explicitly), icl_examples=2, lr 2.5e-4,
# 3 epochs, save_total_limit=1, eval over the full test_graphs set.
#
# The wandb names are overridable, in priority order:
#   positional arg  >  same-named env var  >  built-in default.
#   RUN_NAME (arg 1) -> wandb.run_name AND trainer.save_name (checkpoint subdir)
#   TAG      (arg 2) -> wandb.tag (run group)
#
# Submit:  sbatch scripts/e14_baseline_llm.sbatch [RUN_NAME] [TAG]
#   env  :  RUN_NAME=... TAG=... DATA_ROOT=... TEXT_EDGE_LIST=present \
#             sbatch scripts/e14_baseline_llm.sbatch
# Local :  bash   scripts/e14_baseline_llm.sbatch [RUN_NAME] [TAG]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ENV_NAME=${ENV_NAME:-/vast/projects/aribeiro/alelab/jporras/envs/GREP-PRISM-v3}
ALELAB=/vast/projects/aribeiro/alelab/jporras
PROJ=$ALELAB/GREP-PRISM
# Same folder as the e14 run it controls for; trainer.save_name keeps them apart.
CHECKPOINT_DIR=$PROJ/outputs/e14_stage1to3
# TRAINING DATA SOURCE — identical to e14. One root; train/val/test all derive from it,
# so the baseline cannot silently end up on a different corpus than the model it brackets.
DATA_ROOT=${DATA_ROOT:-data/n_30/gen/nav100_n30_gemma_data/split}
TRAIN_FILES=${TRAIN_FILES:-$DATA_ROOT/formatted_all_new__train.json}
VAL_FILES=${VAL_FILES:-$DATA_ROOT/formatted_all_new__val.json}
TEST_GRAPHS=${TEST_GRAPHS:-$DATA_ROOT/test_graphs}
# See the header: "none" (e14 parity) is the no-connectivity floor; "present" is the
# edges-in-text bracket.
TEXT_EDGE_LIST=${TEXT_EDGE_LIST:-none}
# 2 = the SPINE_ICL_EXAMPLES pair, same as e14 (and same caveat: those canned examples
# call goto/map_region while spine_tools stays at its default "none", left as a decision
# rather than flipped, so the baseline's prompt format matches e14's exactly).
DATA_ICL_EXAMPLES=${DATA_ICL_EXAMPLES:-2}
# Post-training save->load round-trip: reload the just-saved checkpoint from disk
# and re-eval it on these graphs (train_v3.py:394), so a broken reload path fails
# the job instead of surfacing weeks later. Set POST_TRAIN_GRAPHS=null to skip.
POST_TRAIN_GRAPHS=${POST_TRAIN_GRAPHS:-$TEST_GRAPHS}

# wandb names: positional arg > env var > default. RUN_NAME also drives the
# checkpoint subdir (trainer.save_name) so the on-disk run stays identifiable.
RUN_NAME=${1:-${RUN_NAME:-e14_baseline_llm}}
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

echo "[$(date --iso-8601=seconds)] $RUN_NAME training (plain-LLM baseline for e14, tag=$TAG)"
echo "  arch: llm — no graph channel; fresh LoRA (init_lora_from unsupported for arch=llm)"
echo "  text_edge_list: $TEXT_EDGE_LIST"

srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    model.path="google/gemma-4-31B-it" \
    gnn.arch=llm \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.text_edge_list="$TEXT_EDGE_LIST" \
    data.icl_examples="$DATA_ICL_EXAMPLES" \
    trainer.freeze_lora=false \
    trainer.learning_rate=0.00025 trainer.epochs=3 \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$RUN_NAME" \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG" \
    eval.data="$TEST_GRAPHS" eval.num_graphs=-1 \
    eval.post_train_graphs="$POST_TRAIN_GRAPHS"

echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
