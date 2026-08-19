#!/usr/bin/env bash
# e17 end-to-end: train MagCompGraphLLM -> score it on N=100 -> train
# CompositeWireGraphLLM -> score that on N=100. Each stage waits $GPU_WAIT s (60) for
# free GPUs, same guard as run_opt_comp.sh.
#
#   bash run_n100_eval_then_wire.sh                  # all 4 stages, in order
#   STAGE=pipeline  bash run_n100_eval_then_wire.sh  # e17_pipeline.sh only
#   STAGE=eval      bash run_n100_eval_then_wire.sh  # RE-score the latest MagComp ckpt
#   STAGE=train     bash run_n100_eval_then_wire.sh  # WIRE training only
#   STAGE=wire-eval bash run_n100_eval_then_wire.sh  # RE-score the latest WIRE ckpt
#
# TWO runs by default. The N=100 eval is COLLAPSED into each training via
# eval.post_train_graphs, so it runs in-process and its results reach W&B directly:
#   1  scripts/e17_pipeline.sh  -> MagE-GT + Stage 3 MagCompGraphLLM   + N=100 post-eval
#   2  CompositeWireGraphLLM training (Suite 3 MagE-GT)                + N=100 post-eval
#      -- ARCHITECTURE SWITCHES HERE --
#
# WHY THAT IS THE SAME EVAL, NOT A LOOKALIKE. train_v3's post-train block reloads the
# just-saved checkpoint with checkpoint.load_checkpoint(output_dir, four_bit=
# trainer.bit4, device=trainer.device) — the SAME function scalability_evaluation
# aliases as _load_checkpoint (:69) — then calls evaluate.evaluate_model, which calls
# eval_model_multiple_graphs with include_edge_list/use_icl/edge_weights/
# injection_scope/permutation=None/client=None. Standalone, those values are RECOVERED
# from train_config.json; here they come from the live config that WROTE that file, so
# they are equal by construction. The only difference is on_graph_done (a progress
# printer vs None) which only prints. bit4=true and device=-1 reproduce --four-bit
# --device -1, and this script's global env gives the eval the same tool policy.
#
# Results land as wandb.run.summary["posteval/*"] (accuracy, per-graph acc, path
# metrics) plus <run_dir>/eval_logs/cross_eval/*.json and a figure. The namespace is
# posteval/, never eval/: eval/accuracy is the IN-TRAINING metric over config.eval.data,
# a DIFFERENT graph set, and one key cannot mean two measurements.
#
# The standalone scalability_evaluation stages are still there but OFF the default path
# (STAGE=eval / STAGE=wire-eval) — use them to re-score a checkpoint or with REPEATS>1.
#
# SPINE TOOLS ARE OFF EVERYWHERE. PRISM_DISABLE_SPINE_TOOLS=1 is exported once, below,
# so both TRAININGS (their in-run EvalCallback) and both POST-HOC evals see the same
# prompt. That is the parity property: a post-hoc number reproduces the training curve
# only if the tool policy matches. Note what tools-off actually does — it appends the
# _NO_TOOL_CALL_DIRECTIVE to the system prompt (evaluate.py:144), swaps the simulator to
# _NoToolsGraphSim, and cuts max_new_tokens 4x (2048*4 -> 2048, inference.py:37). It is a
# DIFFERENT measurement from the tools-on runs, not a cleaner one: on ocw1jjkj's own
# eval_logs 54 of 84 scored samples emitted a real `goto`, and the path validator parses
# str(plan) with those arguments in it.
#
# SHARED HYPERPARAMETERS (both architectures, so the arms are comparable):
#   trainer.learning_rate    2.5e-4   passed EXPLICITLY to both. e17_ms_stage3.yaml sets
#                                     it, but e17_composite_wire.yaml does NOT, so WIRE
#                                     would inherit base_config's 2.0e-4 — a 25% gap on
#                                     the base group. MEASURED by composing both configs.
#   gnn.structural_lr_mult   0.0012   matches ll9rhvb7 (the 87% LearnableGraphMaskLLM).
#                                     REVERTS e17_ms_stage3.yaml's 0.012 from commit
#                                     f4e74ec; 2.5e-4 * 0.0012 = 3.0e-7 on the structural
#                                     group, vs 3.0e-6 before.
#   data.text_edge_list      none     WIRE only; the mask config already ships it. See
#   data.max_seq_length      8192     the TEXT_EDGE_LIST block below.
#   gnn.mask_alpha           0.0      matches ll9rhvb7. INERT on MagCompGraphLLM —
#                                     architectures.py:146 passes it and the ctor
#                                     range-checks it, but MagCompGraphLLM.build_
#                                     structural_mask overrides the parent wholesale and
#                                     never reads _mask_alpha (the 2-term log form
#                                     dropped alpha). Set so the RECORDED config matches;
#                                     it changes no computation. base_config.yaml:173
#                                     still claims alpha is read under "log" — stale.
# Passed as CLI overrides, so experiments/*.yaml are untouched: run e17_pipeline.sh
# directly and you get its own 0.012 back.
#
# NOT DETERMINISTIC, BY CONSTRUCTION. fixed_seed_mode=false with num_samples=320 means
# R-PEARL redraws its probes every forward, so C = E_q[Phi' Phi'^T] - Psi Psi^T is a fresh
# Monte-Carlo estimate each time. Decoding is greedy (do_sample=False), but the logits it
# decodes are not fixed and nothing seeds torch before eval. Re-running will NOT reproduce
# the same accuracy. REPEATS>1 measures the spread. The WIRE arm carries MORE of it:
# wire_signal=cov_factor redraws the JL projection G every forward too, so r itself is
# resampled on top of the probes.
set -euo pipefail
# REPO ROOT, not the script's dir: this lives in scripts/ and every path below (data/,
# outputs/, scripts/e17_pipeline.sh) is repo-relative. Same anchor e17_pipeline.sh uses.
cd "$(dirname "$0")/.."
BASE="${BASE:-$PWD}"

# The N=100 TEST split, 10 graphs. Three candidates exist; this is the current one
# (…_old_gpt51_partial is stale, split/ holds only 2).
N100_GRAPHS="${N100_GRAPHS:-$BASE/data/n_100/gen/nav_n100_gemma_data/test_graphs}"
MAGGT="${MAGGT:-outputs/e17_mag_gt/suite3/mag_gt.pt}"
REPEATS="${REPEATS:-1}"
STAGE="${STAGE:-all}"
# In-training generation eval: 10 graphs, the WHOLE test set (the configs ship 3).
EVAL_GRAPHS="${EVAL_GRAPHS:-10}"
# Shared optimisation, applied to BOTH trainings. See the header.
# LR is passed EXPLICITLY to both and not left to the configs: e17_ms_stage3.yaml sets
# 2.5e-4 but e17_composite_wire.yaml never overrides it, so WIRE would otherwise inherit
# base_config's 2.0e-4 and the two arms would differ by 25% on the base group.
LR="${LR:-0.00025}"
STRUCT_LR_MULT="${STRUCT_LR_MULT:-0.0012}"
MASK_ALPHA="${MASK_ALPHA:-0.0}"
# The PROMPT the two arms face, forced identical. e17_ms_stage3.yaml already ships
# none/8192 (matching ll9rhvb7); e17_composite_wire.yaml ships present/2048, which would
# hand the WIRE arm the adjacency it is meant to infer from the graph channel and make
# the two N=100 numbers incomparable. Overridden for WIRE only — the mask side is
# already there, so passing it to the pipeline would be a no-op with a duplicate-key
# risk. Under `none` that config's own measurement is 393-528 post-block tokens, so
# wire_context_window=1024 covers the whole scope (it does NOT under `present`).
TEXT_EDGE_LIST="${TEXT_EDGE_LIST:-none}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"

# Where each training lands. ONE definition per arm: the training stage writes here and
# the eval stage discovers here, so the two cannot drift onto different runs.
# MASK_* mirrors scripts/e17_pipeline.sh's own layout (LORA_SUITE defaults to suite3).
LORA_SUITE="${E17_LORA_SUITE:-suite3}"
MASK_OUT="${MASK_OUT:-outputs/e17_magnetic_composite_graphs/$LORA_SUITE}"
MASK_NAME="${MASK_NAME:-e17_ms_stage3_$LORA_SUITE}"
WIRE_NAME="${WIRE_NAME:-e17_composite_wire_suite3}"
WIRE_OUT="${WIRE_OUT:-outputs/e17_composite_wire/suite3}"

# Fresh adapter by default: "a full training of CompositeWireGraphLLM" is a run in its
# own right, not a Stage-3 continuation. The mask arm's Stage 3 attaches its Stage-1
# adapter (e17_pipeline.sh does that itself); set this to carry one into WIRE instead.
E17W_INIT_LORA="${E17W_INIT_LORA:-}"

gpu_free() {
  command -v nvidia-smi >/dev/null || return 0
  local apps deadline=$((SECONDS + ${GPU_WAIT:-60}))
  while :; do
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader) ||
      { echo "nvidia-smi failed — cannot confirm the GPUs are free." >&2; exit 1; }
    [ -n "$apps" ] || return 0
    [ "$SECONDS" -lt "$deadline" ] || break
    sleep 5
  done
  printf 'GPUs busy — halting before %s:\n%s\n' "$1" "$apps" >&2; exit 1
}

# `conda` is an rc-file function, which a non-interactive bash never sources.
for c in "${CONDA_ROOT:-}" "$(conda info --base 2>/dev/null || true)" "$HOME"/{miniconda3,anaconda3,miniforge3}; do
  if [ -r "$c/etc/profile.d/conda.sh" ]; then CONDA_SH=$c/etc/profile.d/conda.sh; break; fi
done
# conda.sh trips `set -u`. The three statements are on separate lines because a
# lint directive binds to the NEXT COMMAND, and on a one-liner that would be `set +u`.
set +u
# SC1090: the path is discovered at runtime by design, so it cannot be constant.
# shellcheck source=/dev/null
. "${CONDA_SH:?conda.sh not found; set CONDA_ROOT}"
set -u
conda deactivate 2>/dev/null || true; conda activate GREP-PRISM

# Same allocator + visible devices as scripts/e17_pipeline.sh, so every stage sees the
# memory regime it was calibrated under. (The pipeline re-exports these with :- defaults,
# so setting them here is what it will use.)
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export WANDB_MODE="${E17_WANDB_MODE:-online}"

# Every env var the eval path reads, PINNED. Unsetting reproduces each one's CODE
# DEFAULT — what a clean shell gives — and cannot drift from those defaults the way
# hard-coded values would. Inheriting them from an interactive shell is a silent
# train/eval divergence; two of them default ON.
#   GREP_JUDGE=1                    (path_validator.py:221) AC judge LIVE
#   GREP_JUDGE_MODEL=gemma-4-E4B-it (path_validator.py:52)  which judge
#   GREP_PATH_RESCUE=1              (path_validator.py:330) rescue LIVE
#   GREP_GEMMA_REGRADE=0            (evaluate.py:126)       regrade OFF
#   PRISM_EVAL_BACKEND              (scalability_evaluation.py:111) sources --backend
unset GREP_JUDGE GREP_JUDGE_MODEL GREP_PATH_RESCUE GREP_GEMMA_REGRADE PRISM_EVAL_BACKEND
# The ONE eval env var set rather than unset: tools OFF for trainings and evals alike.
export PRISM_DISABLE_SPINE_TOOLS=1

check_graphs() {
  local dir="$1" name="$2"
  [ -d "$dir" ] || { echo "FATAL: $name graph dir not found: $dir" >&2; exit 1; }
  shopt -s nullglob; local g=("$dir"/*.json); shopt -u nullglob
  [ "${#g[@]}" -gt 0 ] || { echo "FATAL: no *.json graphs in $name dir: $dir" >&2; exit 1; }
  echo "$name test set : $dir (${#g[@]} graphs)"
}
check_graphs "$N100_GRAPHS" "N=100"
echo "shared optim       : lr=$LR  structural_lr_mult=$STRUCT_LR_MULT  mask_alpha=$MASK_ALPHA"
echo "in-training eval   : num_graphs=$EVAL_GRAPHS  use_icl=false  SPINE tools OFF"

# Newest run dir matching {save_name}_{wandb_id}, or empty. The W&B id is minted at
# runtime so the directory is only knowable by search; -t sorts by mtime, so this is the
# run that just finished. `|| true` is load-bearing: with `set -e` + `pipefail` an
# unmatched glob makes ls fail and the substitution inherits it, killing the script
# before the caller can report a useful error.
# SC2012: `ls -t` is used for its MTIME ORDERING, which find cannot do portably. These
# paths are ours ({save_name}_{wandb_id}), so the odd-filename hazard cannot arise.
latest_run() {
  local d
  # Directive binds to the NEXT COMMAND, so the assignment gets its own line.
  # shellcheck disable=SC2012
  d="$(ls -dt "$BASE/$1"/"$2"_*/ 2>/dev/null | head -1 || true)"
  echo "${d%/}"
}

# Refuse a checkpoint that is not the architecture this stage means to score. A config
# missing the keys that SELECT the architecture would reload as a different model with a
# clean load — the enforce_beta_bound / wire_signal failure mode.
assert_arch() {
  python3 - "$1" "$2" <<'PY' || exit 1
import json, sys, os
path, want = sys.argv[1], sys.argv[2]
p = os.path.join(path, "train_config.json")
if not os.path.exists(p):
    sys.exit(f"FATAL: {p} not found — not a usable checkpoint dir.")
tc = json.load(open(p)); g = tc.get("gnn") or {}
arch = tc.get("architecture")
if want == "magcomp":
    need, ok = ("mask_composite", "pe_pool", "directed", "learn_r"), arch == "learnable_graph_mask"
    ok = ok and g.get("mask_composite") is True
    what = "MagCompGraphLLM (learnable_graph_mask + mask_composite=true)"
else:
    need, ok = ("wire_composite", "wire_signal", "wire_vanilla", "pe_pool",
                "directed", "learn_r"), arch == "wire_llm"
    ok = ok and g.get("wire_composite") is True
    what = "CompositeWireGraphLLM (wire_llm + wire_composite=true)"
missing = [k for k in need if k not in g]
if missing:
    sys.exit(f"FATAL: {p} is missing {missing} — this checkpoint predates the key(s) and "
             "would reload as a different model. Refusing to score it.")
if not ok:
    sys.exit(f"FATAL: {p} records architecture={arch!r} mask_composite={g.get('mask_composite')} "
             f"wire_composite={g.get('wire_composite')}; expected {what}.")
print(f"[arch] {os.path.basename(path)} -> {what}")
PY
}

# One cross-eval, repeated REPEATS times.
#   $1 checkpoint   $2 graph dir   $3 output root   $4 label
# Every policy flag below is identical for both architectures; everything that DIFFERS
# between them (text_edge_list, edge_weights, injection_scope) is auto-recovered by the
# CLI from $1's own train_config.json, which is why none of those is passed.
run_eval() {
  local ckpt="$1" graphs="$2" root="$3" label="$4" i out icl use_icl
  [ -d "$ckpt" ] || { echo "FATAL: checkpoint not found: $ckpt" >&2; exit 1; }
  # DERIVED, not pinned. scalability_evaluation never calls resolve_prompt_policy, so its
  # --use-icl is a silent CLI default rather than a recovered value; recovering it here
  # makes every eval condition come from the run being scored. The REAL function is
  # called (not a reimplementation) so the two cannot drift — it reads the top level and
  # then the nested "gnn" block, which matters for older checkpoints.
  # NOTE spine_tools is deliberately NOT read from it: that is the TRAINING TARGET
  # policy, not the eval-time tool switch, which this script sets globally above.
  icl=$(uv run python -c '
import sys; sys.path.insert(0, "src")
from prism.eval.checkpoint import resolve_prompt_policy
print(resolve_prompt_policy(sys.argv[1])[1])' "$ckpt") || {
    echo "FATAL: could not recover the prompt policy from $ckpt" >&2; exit 1; }
  [ "$icl" -gt 0 ] 2>/dev/null && use_icl=true || use_icl=false
  echo "[$label] icl_examples=$icl -> --use-icl $use_icl"
  for i in $(seq 1 "$REPEATS"); do
    gpu_free "$label eval (repeat $i/$REPEATS)"
    out="$root"; [ "$REPEATS" -gt 1 ] && out="$root/rep_$i"
    echo "=== $label cross-eval, repeat $i/$REPEATS -> $out ==="
    uv run -m prism.eval.scalability_evaluation \
      --checkpoint "$ckpt" \
      --graphs "$graphs" \
      --four-bit \
      --device -1 \
      --use-icl "$use_icl" \
      --backend hf \
      --output "$out"
  done
}

# ------------------------------------------------- 1) e17_pipeline.sh (MagCompGraphLLM)
# E17_EVAL_GRAPHS is the pipeline's OWN knob for eval.num_graphs — using it (rather than
# passing eval.num_graphs in "$@") is what keeps Hydra from seeing the key twice, since
# the pipeline already emits it. The positional overrides reach STAGE 3 ONLY: that is
# where "$@" lands. Stage 1 is off by default and freezes the PE, so the structural LR
# does not apply to it.
if [ "$STAGE" = all ] || [ "$STAGE" = pipeline ]; then
  gpu_free "e17_pipeline.sh (MagCompGraphLLM)"
  echo "=== [1/4] scripts/e17_pipeline.sh — MagE-GT + Stage 3 MagCompGraphLLM ==="
  E17_EVAL_GRAPHS="$EVAL_GRAPHS" bash scripts/e17_pipeline.sh \
    trainer.learning_rate="$LR" \
    gnn.structural_lr_mult="$STRUCT_LR_MULT" \
    gnn.mask_alpha="$MASK_ALPHA" \
    eval.post_train_graphs="$N100_GRAPHS" \
    eval.use_icl=false
fi

# ------------------------------------------------------- 2) MagCompGraphLLM N=100 eval
if [ "$STAGE" = eval ]; then
  MASK_CKPT="${MASK_CKPT:-$(latest_run "$MASK_OUT" "$MASK_NAME")}"
  MASK_CKPT="${MASK_CKPT%/}"          # an explicit override may carry a trailing slash
  if [ -z "$MASK_CKPT" ]; then
    echo "FATAL: no MagCompGraphLLM checkpoint under $BASE/$MASK_OUT/${MASK_NAME}_*" >&2
    echo "       Run STAGE=pipeline first, or set MASK_CKPT=<run dir> explicitly." >&2
    exit 1
  fi
  assert_arch "$MASK_CKPT" magcomp
  echo "=== [2/4] MagCompGraphLLM checkpoint: $MASK_CKPT ==="
  run_eval "$MASK_CKPT" "$N100_GRAPHS" \
           "${MASK_OUT_BASE:-$BASE/results/$(basename "$MASK_CKPT")}_n100" "MagComp N100"
fi

# --------------------------------------------- 3) CompositeWireGraphLLM training
# wire_signal=cov_factor, pe_pool=gt, wire_vanilla=false and fixed_seed_mode=false are
# asserted at compose time by train_v3._validate_config, so a mis-set value fails before
# the 31B is loaded rather than at loss.
if [ "$STAGE" = all ] || [ "$STAGE" = train ]; then
  gpu_free "e17 composite WIRE training"
  echo "=== [3/4] CompositeWireGraphLLM (Suite 3 MagE-GT: $MAGGT) ==="
  echo "!! wire_max_angle is still 16.0, inherited from the wire_signal=psi arm."
  echo "!! The covariance factor's span is UNMEASURED — read wire/psi_span and"
  echo "!! wire/angle_eff_max off the first logged step and set the clamp from that"
  echo "!! measurement before treating this run as a baseline."
  [ -f "$BASE/$MAGGT" ] || { echo "FATAL: MagE-GT not found: $BASE/$MAGGT" >&2; exit 1; }
  uv run -m prism.training.train_v3 --config-name=e17_composite_wire \
    model.path=google/gemma-4-31B-it \
    trainer.device=-1 \
    gnn.pe_gt_from="$MAGGT" \
    gnn.semantic_gt_from=null \
    gnn.enforce_beta_bound=false \
    trainer.learning_rate="$LR" \
    gnn.structural_lr_mult="$STRUCT_LR_MULT" \
    data.text_edge_list="$TEXT_EDGE_LIST" \
    data.max_seq_length="$MAX_SEQ_LEN" \
    eval.num_graphs="$EVAL_GRAPHS" \
    eval.post_train_graphs="$N100_GRAPHS" \
    eval.use_icl=false \
    +trainer.sft.logging_steps=15 \
    trainer.epochs=3 \
    trainer.checkpoint_dir="$WIRE_OUT" \
    trainer.save_name="$WIRE_NAME" \
    wandb.run_name="$WIRE_NAME" \
    ${E17W_INIT_LORA:+trainer.init_lora_from="$E17W_INIT_LORA"}
fi

# ------------------------------------------------- 4) CompositeWireGraphLLM N=100 eval
if [ "$STAGE" = wire-eval ]; then
  WIRE_CKPT="${WIRE_CKPT:-$(latest_run "$WIRE_OUT" "$WIRE_NAME")}"
  WIRE_CKPT="${WIRE_CKPT%/}"          # an explicit override may carry a trailing slash
  if [ -z "$WIRE_CKPT" ]; then
    echo "FATAL: no WIRE checkpoint under $BASE/$WIRE_OUT/${WIRE_NAME}_*" >&2
    echo "       Run STAGE=train first, or set WIRE_CKPT=<run dir> explicitly." >&2
    exit 1
  fi
  assert_arch "$WIRE_CKPT" wire
  echo "=== [4/4] WIRE checkpoint: $WIRE_CKPT ==="
  run_eval "$WIRE_CKPT" "$N100_GRAPHS" \
           "${WIRE_OUT_BASE:-$BASE/results/$(basename "$WIRE_CKPT")}_n100" "WIRE N100"
fi
