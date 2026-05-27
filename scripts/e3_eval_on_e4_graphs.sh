#!/bin/bash
# Evaluate the three e3 model architectures on the e4 test graphs.
# Run each block in a separate terminal/tmux pane with the desired GPU.
#
# Results are written to:
#   outputs/e3_new_training_data/<run>/eval_logs/cross_eval/<graph>.json
#
# Usage: bash scripts/e3_eval_on_e4_graphs.sh [llm|rpearl|gt]
#   Omit the argument to print the commands for all three models.

E4_GRAPHS="data/training_data_20260428/aggregate_20260428/split_20260428/test_graphs"

LLM_CKPT="outputs/e3_new_training_data/e3_llm_llama-3.1-8b_r16_4bit_0sy9j5rz"
RPEARL_CKPT="outputs/e3_new_training_data/e3_rpearl_llm_llama-3.1-8b_r16_4bit_qmu8x2qu"
GT_CKPT="outputs/e3_new_training_data/e3_rpearl_gt_llm_llama-3.1-8b_r16_4bit_bzqt4zxt"

run_llm() {
    echo "=== e3 LLM (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto}) ==="
    python scripts/eval_checkpoint_on_graphs.py \
        --checkpoint "$LLM_CKPT" \
        --graphs "$E4_GRAPHS" \
        --text-edge-list present
}

run_rpearl() {
    echo "=== e3 RPEARL (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto}) ==="
    python scripts/eval_checkpoint_on_graphs.py \
        --checkpoint "$RPEARL_CKPT" \
        --graphs "$E4_GRAPHS" \
        --text-edge-list none
}

run_gt() {
    echo "=== e3 RPEARL-GT (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto}) ==="
    python scripts/eval_checkpoint_on_graphs.py \
        --checkpoint "$GT_CKPT" \
        --graphs "$E4_GRAPHS" \
        --text-edge-list none
}

case "${1:-}" in
    llm)    run_llm ;;
    rpearl) run_rpearl ;;
    gt)     run_gt ;;
    "")
        echo "Run each in a separate pane, e.g.:"
        echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/e3_eval_on_e4_graphs.sh llm"
        echo "  CUDA_VISIBLE_DEVICES=1 bash scripts/e3_eval_on_e4_graphs.sh rpearl"
        echo "  CUDA_VISIBLE_DEVICES=0 bash scripts/e3_eval_on_e4_graphs.sh gt"
        echo ""
        echo "Or run sequentially on one GPU:"
        run_llm && run_rpearl && run_gt
        ;;
    *)
        echo "Unknown argument: $1. Use llm, rpearl, or gt." >&2
        exit 1
        ;;
esac
