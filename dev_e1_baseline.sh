#!/bin/bash

################################################################################
# E1 Baseline Experiment: R-PEARL + LLM vs Pure LLM
#
# Runs training jobs with identical hyperparameters, differing only in the
# --architecture and --text_edge_list flags:
#   1. rpearl_llm  — GNN positional encodings via R-PEARL (strips scene graph text)
#   2. llm         — Pure LLM baseline (scene graph text stays in prompt)
#
# Flags:
#   --architecture    rpearl_llm | llm | both  (default: both)
#   --text_edge_list  present | none            (default: present)
#                       present — object_connections and region_connections are
#                                 kept in the scene graph text
#                       none    — both connection lists are stripped from the
#                                 scene graph text before training
#   --data            path to training JSON     (default: data/gen/spine_exp1/formatted.json)
#   --checkpoint_dir  path to checkpoint dir    (default: outputs/e1_baseline)
#
# Usage:
#   ./dev_e1_baseline.sh                                          # both archs, edges present
#   ./dev_e1_baseline.sh --architecture rpearl_llm                # only GNN-augmented
#   ./dev_e1_baseline.sh --architecture llm                       # only pure LLM
#   ./dev_e1_baseline.sh --text_edge_list none                    # strip edge lists, both archs
#   ./dev_e1_baseline.sh --architecture llm --text_edge_list none # ablation: LLM without edges
#   ./dev_e1_baseline.sh --data /path/to/data.json                # custom data path
#   ./dev_e1_baseline.sh --checkpoint_dir /path/to/ckpts          # custom checkpoint dir
################################################################################

set -e

# ============================================================================
# Resolve project root (directory containing this script)
# ============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ============================================================================
# Defaults
# ============================================================================

DATA_PATH="${PROJECT_ROOT}/data/gen/spine_exp1/formatted.json"
CHECKPOINT_DIR="${PROJECT_ROOT}/outputs/e1_baseline"
ARCHITECTURE="both"   # "rpearl_llm", "llm", or "both"
TEXT_EDGE_LIST="present"  # "present" or "none"

# ============================================================================
# Parse named arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case "$1" in
        --architecture)
            ARCHITECTURE="$2"
            shift 2
            ;;
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --text_edge_list)
            TEXT_EDGE_LIST="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 [--architecture rpearl_llm|llm|both] [--data PATH] [--checkpoint_dir PATH] [--text_edge_list present|none]"
            exit 1
            ;;
    esac
done

# Validate architecture
if [[ "$ARCHITECTURE" != "rpearl_llm" && "$ARCHITECTURE" != "llm" && "$ARCHITECTURE" != "both" ]]; then
    echo "Error: --architecture must be one of: rpearl_llm, llm, both (got: $ARCHITECTURE)"
    exit 1
fi

if [[ "$TEXT_EDGE_LIST" != "present" && "$TEXT_EDGE_LIST" != "none" ]]; then
    echo "Error: --text_edge_list must be 'present' or 'none' (got: $TEXT_EDGE_LIST)"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

echo "Data:           $DATA_PATH"
echo "Checkpoints:    $CHECKPOINT_DIR"
echo "Architecture:   $ARCHITECTURE"
echo "Text edge list: $TEXT_EDGE_LIST"

# ============================================================================
# Shared hyperparameters
# ============================================================================

EXPERIMENT_NAME="dev_e1"
WANDB_TAG="dev_e1"

LEARNING_RATE=2e-4
EPOCHS=2
R=16
D_MODEL=3072
PE_HIDDEN_CHANNELS=256
PE_NUM_LAYERS=5
NUM_SAMPLES=40
DROPOUT=0.1
K=3
USE_LAYER_NORM=True
FREEZE_LLM=False   
DEBUG=True
DATASET_PROPORTION=1.0
MAX_SEQ_LENGTH=2048

# ============================================================================
# Helper: run a single architecture
# ============================================================================

run_architecture() {
    local arch="$1"
    local run_label="$2"

    echo ""
    echo "###############################################################################"
    echo "# ${run_label}: ${arch}"
    echo "###############################################################################"
    echo ""

    # Build the base command
    local cmd=(
        python -m prism.training.train_v2
        --name "${EXPERIMENT_NAME}"
        --architecture "${arch}"
        --checkpoint_dir "$CHECKPOINT_DIR"
        --data "$DATA_PATH"
        --base_model meta-llama/Llama-3.2-3B-Instruct
        --r $R
        --learning_rate $LEARNING_RATE
        --epochs $EPOCHS
        --debug $DEBUG
        --dataset_proportion $DATASET_PROPORTION
        --max_seq_length $MAX_SEQ_LENGTH
        --text_edge_list "$TEXT_EDGE_LIST"
        --wandb_project GREP-PRISM
        --wandb_run_name "${EXPERIMENT_NAME}_${arch}"
        --wandb_tag "$WANDB_TAG"
    )

    # rpearl_llm needs extra GNN-specific flags
    if [[ "$arch" == "rpearl_llm" ]]; then
        cmd+=(
            --d_model $D_MODEL
            --pe_hidden_channels $PE_HIDDEN_CHANNELS
            --pe_num_layers $PE_NUM_LAYERS
            --num_samples $NUM_SAMPLES
            --dropout $DROPOUT
            --k $K
            --use_layer_norm $USE_LAYER_NORM
            --freeze_llm $FREEZE_LLM
        )
    fi

    "${cmd[@]}"

    echo ""
    echo "${run_label} (${arch}) completed."
    echo ""
}

# ============================================================================
# Execute
# ============================================================================

cd "${PROJECT_ROOT}/src"

if [[ "$ARCHITECTURE" == "rpearl_llm" || "$ARCHITECTURE" == "both" ]]; then
    run_architecture "rpearl_llm" "RUN: rpearl_llm (GNN-augmented)"
fi

if [[ "$ARCHITECTURE" == "llm" || "$ARCHITECTURE" == "both" ]]; then
    run_architecture "llm" "RUN: llm (pure LLM baseline)"
fi

cd "${PROJECT_ROOT}"

# ============================================================================
# Summary
# ============================================================================

echo "###############################################################################"
echo "# E1 BASELINE EXPERIMENT COMPLETE"
echo "###############################################################################"
echo ""
echo "Both runs finished. Compare on W&B under tag: ${WANDB_TAG}"
echo "  1. ${EXPERIMENT_NAME}_rpearl_llm"
echo "  2. ${EXPERIMENT_NAME}_llm"
echo ""
