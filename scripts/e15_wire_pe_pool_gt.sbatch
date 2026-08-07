#!/bin/bash
#SBATCH --job-name=e15-wire-pe-pool-gt
#SBATCH --output=/vast/projects/aribeiro/alelab/jporras/GREP-PRISM/slurm-%x-%j.out
#SBATCH --time=24:00:00
#SBATCH --partition=dgx-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=224G
#
# e15 — WIRE (WireGraphLLM) trained END-TO-END on data/n_30, with the deferred-expectation
# positional encoder
#
#     Psi = E_q[ Phi( Phi(q; S, H), G, T ) ]        (gnn.pe_pool=gt)
#
# i.e. the M probe responses Phi(q^(s); S, H) are pushed through the GT blocks as one
# block-diagonal M-copy graph (probes independent, weights shared) and the probe
# expectation is taken LAST, instead of the historical
#
#     Psi = Phi( E_q[Phi(q; S, H)], G, T )          (gnn.pe_pool=pe, the default)
#
# which pools inside R-PEARL and lets the blocks see the first moment only.
# Both are Monte-Carlo estimators (E_q is symmetric in the probe set); they are not the
# same function of the same weights, which is exactly why pe_pool is now recorded in
# train_config.json (train_v3.py) — eval rebuilds Psi through gt.build_psi_producer and
# would otherwise fall back to "pe" with a clean, silent state_dict load.
#
# CONFIG LINEAGE — this MIRRORS experiments/e9_ms_stage1.yaml (= base_config.yaml +
# name/run_name + freeze_pe), reached through e9_base_config so the e9 wandb tagging is
# inherited. Everything not listed below is base_config as-is: gemma-4-31B-it, bf16
# (bit4=false), LoRA r16/alpha16/dropout0.2 on the 7 attention+MLP projections, lr 2e-4,
# 1 epoch, warmup 5, wd 0.05, per-device bs 1 x grad-accum 2, gradient checkpointing,
# loss_target=responses, text_edge_list=present, spine_tools=none, icl_examples=0,
# injection_scope=full_sequence, edge_weights=binary, eval every epoch on 5 graphs.
#
# FOUR deliberate departures from e9_ms_stage1, each stated because none is cosmetic:
#
#   1. gnn.arch=wire_llm (stage 1 is rpearl_llm). Psi enters as a ROTATION of q/k
#      (q_i -> R(theta_i) q_i, theta_n = omega_n . Psi_i) composed on top of Gemma's own
#      RoPE — never as an added signal, so there is no pe_proj/pe_norm here.
#      wire_llm consumes a BARE Psi producer (gt.build_psi_producer), which is the only
#      construction site that reads pe_pool — hence the arch this experiment needs.
#
#   2. gnn.pe_pool=gt — the point of the run (see above). Requires
#      pe_node_features=random (a deferred E_q needs a probe axis; wire_llm requires
#      random probes anyway) and pe_readout=mean, which build_psi_producer leaves at its
#      default.
#
#   3. trainer.freeze_pe=false — "END-TO-END". Stage 1 sets freeze_pe=true, which exists
#      to hold the PE inert while ONLY the LoRA fits the SPINE style; freezing it here
#      would train nothing but the adapter and leave Psi, omega and the gate at init, so
#      the pe_pool axis would be untested. Both the LoRA and the whole structural path
#      (GT/R-PEARL + omega + pe_gain) train in this run.
#
#   4. gnn.pe_gain_init=1.0, NOT base_config's 0.0. This is a CORRECTNESS fix, not a
#      tuning choice. WIRE's angle is BILINEAR in the gate and the frequencies:
#          r = Psi * tanh(pe_gain)          (gnn_llm.py, build_wire_signal)
#          theta = r @ omega^T              (gnn_llm.wire_cos_sin)
#      With wire_vanilla=true and wire_vanilla_omega_init=zero (both base defaults, and
#      both kept) omega starts at EXACTLY 0. Adding pe_gain_init=0.0 zeroes the other
#      factor too, and then BOTH gradients vanish at init:
#          d theta / d omega   = r        = 0     (gate is 0)
#          d theta / d pe_gain = omega... = 0     (omega is 0)
#      so omega, pe_gain and — through them — the entire Psi producer receive exactly
#      zero gradient and the model is pinned at the identity rotation for the whole run.
#      MEASURED on the tiny Gemma4 fixture, 3 SGD steps, both pe_pool arms: with 0.0 the
#      loss is bit-identical across steps and all three grad norms are exactly 0; with
#      1.0 step 0 moves omega alone (||g omega||=1.6e-2, the other two still 0 because
#      d theta/d r = omega = 0 there) and from step 1 the gate and the Psi producer pick
#      up gradient and the loss falls. So the identity cold start is still there — it is
#      now supplied ONCE, by omega=0, and costs one bootstrap step instead of the run.
#      base_config's 0.0 default is written for the ADDITIVE family, where Psi is added
#      rather than multiplied by omega and tanh'(0)=1 keeps the path alive.
#
# DELIBERATELY NOT SET — do not add without a reason:
#   * gnn.pe_gt_from / semantic_gt_from — Psi trains from scratch here, as in stage 1.
#     A navigator-pretrained Psi is a different arm (see e14_stage1to3_binary.sbatch).
#   * trainer.init_lora_from / init_pe_from — this is a first stage, not a continuation.
#   * gnn.structural_lr_mult — left at base 1.0; a two-group LR is a later A/B, not part
#     of a baseline.
#
# COST of pe_pool=gt, stated so it is not discovered on the node: the GT blocks run on
# M*N rows instead of N. MEASURED on three real data/n_30 test graphs at exactly the
# hyperparameters below (M = gnn.num_samples = 40, d_model = 1024): N = 29-33 nodes
# (24 regions + 5-9 objects), so the blocks see 1160-1320 rows = 4.8-5.4 MB fp32 per
# activation (2 blocks, activation-checkpointed during training) against 0.1 MB for
# pe_pool=pe, and ~40x nnz(A_k) on a k_gt=2 neighbourhood — negligible beside the 31B
# LoRA step. It scales LINEARLY in num_samples, so raising it (stage 2 uses 320) costs 8x.
#
# One measured SIDE EFFECT worth watching, not a defect: deferring E_q changes the Psi
# SCALE. On the same graphs ‖Psi_i‖ falls from 1.10*sqrt(d_model) (pe_pool=pe, the value
# base_config quotes) to 0.72*sqrt(d_model) (pe_pool=gt) — averaging M block outputs
# instead of pushing one averaged probe through them. WIRE angles are proportional to
# ‖Psi‖, so this arm starts at a ~35% smaller angle for the same omega. It is bounded
# either way by the unconditional wire_max_angle clamp; watch wire/psi_span,
# wire/angle_eff_max and wire/clamp_engaged rather than assuming the two arms sit at the
# same angle scale.
#
# The wandb names are overridable, in priority order:
#   positional arg  >  same-named env var  >  built-in default.
#   RUN_NAME (arg 1) -> wandb.run_name AND trainer.save_name (checkpoint subdir)
#   TAG      (arg 2) -> wandb.tag (run group)
#
# Paths are env-overridable: DATA_ROOT (the training corpus; TRAIN_FILES / VAL_FILES /
# TEST_GRAPHS derive from it and are individually overridable too).
#
# Submit:  sbatch scripts/e15_wire_pe_pool_gt.sbatch [RUN_NAME] [TAG]
#   env  :  RUN_NAME=... TAG=... DATA_ROOT=... PE_POOL=... sbatch scripts/e15_wire_pe_pool_gt.sbatch
# Local :  bash   scripts/e15_wire_pe_pool_gt.sbatch [RUN_NAME] [TAG]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# Shared drive root. Default matches ~/.bashrc on betty (and the #SBATCH --output
# path above); scripted submissions (ssh betty 'bash -lc "sbatch ..."') never see
# the .bashrc export because of its non-interactive guard.
export ALELAB_DRIVE="${ALELAB_DRIVE:-/vast/projects/aribeiro/alelab/jporras}"

ENV_NAME=${ENV_NAME:-$ALELAB_DRIVE/envs/GREP-PRISM-v3}
PROJ=$ALELAB_DRIVE/GREP-PRISM
CHECKPOINT_DIR=$PROJ/outputs/e15_wire_pe_pool_gt

# TRAINING DATA SOURCE. One root; train/val/test all derive from it, so switching
# corpora cannot leave the generation eval pointing at a different corpus than the SFT.
# This must be set explicitly: base_config.yaml still defaults train_files/val_files to
# data/revised/..., which does NOT exist in this repo — inheriting it makes the run die
# at datasets.load_dataset.
DATA_ROOT=${DATA_ROOT:-data/n_30/gen/nav100_n30_gemma_data/split}
TRAIN_FILES=${TRAIN_FILES:-$DATA_ROOT/formatted_all_new__train.json}
VAL_FILES=${VAL_FILES:-$DATA_ROOT/formatted_all_new__val.json}
TEST_GRAPHS=${TEST_GRAPHS:-$DATA_ROOT/test_graphs}

# THE AXIS UNDER TEST, exposed so the A/B arm is this same script with PE_POOL=pe and
# nothing else changed.
PE_POOL=${PE_POOL:-gt}
# ICL policy, stated explicitly so it cannot drift with the config defaults. e9_ms_stage1
# is ZERO-SHOT on both sides (base_config data.icl_examples=0 / eval.use_icl=false), and
# the two must agree — ICL also moves the scene graph out of the system message into the
# query turn. Kept at 0 because this run mirrors stage 1; it is NOT the e14 few-shot
# policy.
DATA_ICL_EXAMPLES=${DATA_ICL_EXAMPLES:-0}

# wandb names: positional arg > env var > default. RUN_NAME also drives the
# checkpoint subdir (trainer.save_name) so the on-disk run stays identifiable.
RUN_NAME=${1:-${RUN_NAME:-e15_wire_pe_pool_gt}}
TAG=${2:-${TAG:-e15_wire_pe_pool}}

export WANDB_ENTITY=${WANDB_ENTITY:-alelab}
# Home dir sits near its quota; keep the HF hub cache (base gemma-4-31B-it weights) on
# shared vast — downloaded once, reused across jobs.
export HF_HOME=${HF_HOME:-$ALELAB_DRIVE/hf_cache}
export HF_DATASETS_CACHE="${TMPDIR:-/tmp}/hf_datasets_${SLURM_JOB_ID:-manual}"
# SLURM 22.05+ stopped propagating --cpus-per-task to srun; pass it explicitly
# to avoid "CPU binding outside of job step allocation" on dgx-b200 nodes.
export SRUN_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-28}"

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[$(date --iso-8601=seconds)] $RUN_NAME — WIRE end-to-end, pe_pool=$PE_POOL (tag=$TAG)"
echo "  Psi:   $([ "$PE_POOL" = gt ] && echo 'E_q[Phi(Phi(q; S, H), G, T)]  (E_q after the GT blocks)' \
                                    || echo 'Phi(E_q[Phi(q; S, H)], G, T)  (E_q inside R-PEARL)')"
echo "  data:  $TRAIN_FILES"
echo "  eval:  $TEST_GRAPHS"

srun python -m prism.training.train_v3 \
    --config-name=e9_base_config \
    gnn.arch=wire_llm \
    gnn.pe_pool="$PE_POOL" \
    gnn.pe_node_features=random \
    gnn.pe_gain_init=1.0 \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.icl_examples="$DATA_ICL_EXAMPLES" \
    trainer.freeze_pe=false \
    +trainer.sft.save_total_limit=1 \
    trainer.checkpoint_dir="$CHECKPOINT_DIR" \
    trainer.save_name="$RUN_NAME" \
    wandb.run_name="$RUN_NAME" wandb.tag="$TAG" \
    eval.data="$TEST_GRAPHS"

echo "[$(date --iso-8601=seconds)] $RUN_NAME DONE"
