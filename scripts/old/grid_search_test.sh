#!/bin/bash

################################################################################
# Quick Test Script for R-PEARL Grid Search
# 
# This script runs a minimal set of configurations to test your setup before
# launching the full grid search. Use this to:
# - Verify your data path is correct
# - Test that training runs without errors
# - Estimate runtime per configuration
# - Debug any issues before committing to full grid search
################################################################################

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Activate virtual environment
VENV_PATH="/Users/cyberlives/Documents/GitHub/GREP-PRISM/.venv2"
source "${VENV_PATH}/bin/activate"

# Script paths
PROJECT_ROOT="/Users/cyberlives/Documents/GitHub/GREP-PRISM"
TRAIN_SCRIPT="prism/training/train_v2.py"

# Fixed hyperparameters
LEARNING_RATE=3e-3
FREEZE_LLM=True
DATASET_PROPORTION=0.05  # Use even less data for quick testing
EPOCHS=1  # Just 1 epoch for speed
D_MODEL=3072

# Required arguments
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <data_path> <checkpoint_dir> <experiment_name>"
    echo ""
    echo "This script runs 3 quick test configurations to validate your setup."
    echo ""
    echo "Example:"
    echo "  $0 /path/to/data.json ./test_checkpoints test_run"
    echo ""
    exit 1
fi

DATA_PATH="$1"
CHECKPOINT_DIR="$2"
EXPERIMENT_NAME="$3"

# Verify data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

echo ""
echo "###############################################################################"
echo "# GRID SEARCH TEST RUN"
echo "# Running 3 quick configurations to validate setup"
echo "###############################################################################"
echo ""
echo "Configuration:"
echo "  Data: $DATA_PATH"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Experiment: $EXPERIMENT_NAME"
echo "  Dataset proportion: $DATASET_PROPORTION (5% for quick test)"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LEARNING_RATE"
echo "  Freeze LLM: $FREEZE_LLM"
echo ""

# ============================================================================
# Test Configurations
# ============================================================================

# Test 1: Small model, no regularization (Pass 1 style)
echo "=========================================================================="
echo "TEST 1/3: Small model, no regularization"
echo "=========================================================================="
cd "${PROJECT_ROOT}/src"
python "$TRAIN_SCRIPT" \
    --name "${EXPERIMENT_NAME}_test1_small" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data "$DATA_PATH" \
    --learning_rate $LEARNING_RATE \
    --freeze_llm $FREEZE_LLM \
    --dataset_proportion $DATASET_PROPORTION \
    --pe_num_layers 2 \
    --k 2 \
    --d_model $D_MODEL \
    --pe_hidden_channels 64 \
    --num_samples 10 \
    --dropout 0.0 \
    --use_layer_norm False \
    --epochs $EPOCHS \
    --wandb_run_name "${EXPERIMENT_NAME}_test1" \
    --wandb_tag "grid_search_test" \
    --debug False

cd "${PROJECT_ROOT}"  # Return to project root

echo ""
echo "✓ Test 1 completed successfully"
echo ""

# Test 2: Medium model, with regularization (Pass 2 style)
echo "=========================================================================="
echo "TEST 2/3: Medium model, with regularization"
echo "=========================================================================="
cd "${PROJECT_ROOT}/src"
python "$TRAIN_SCRIPT" \
    --name "${EXPERIMENT_NAME}_test2_medium" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data "$DATA_PATH" \
    --learning_rate $LEARNING_RATE \
    --freeze_llm $FREEZE_LLM \
    --dataset_proportion $DATASET_PROPORTION \
    --pe_num_layers 3 \
    --k 3 \
    --d_model $D_MODEL \
    --pe_hidden_channels 128 \
    --num_samples 30 \
    --dropout 0.05 \
    --use_layer_norm True \
    --epochs $EPOCHS \
    --wandb_run_name "${EXPERIMENT_NAME}_test2" \
    --wandb_tag "grid_search_test" \
    --debug False

cd "${PROJECT_ROOT}"  # Return to project root

echo ""
echo "✓ Test 2 completed successfully"
echo ""

# Test 3: Larger model (Pass 3 style with more data)
echo "=========================================================================="
echo "TEST 3/3: Larger model, more data"
echo "=========================================================================="
cd "${PROJECT_ROOT}/src"
python "$TRAIN_SCRIPT" \
    --name "${EXPERIMENT_NAME}_test3_large" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --data "$DATA_PATH" \
    --learning_rate $LEARNING_RATE \
    --freeze_llm $FREEZE_LLM \
    --dataset_proportion 0.10 \
    --pe_num_layers 4 \
    --k 4 \
    --d_model $D_MODEL \
    --pe_hidden_channels 256 \
    --num_samples 40 \
    --dropout 0.05 \
    --use_layer_norm True \
    --epochs $EPOCHS \
    --wandb_run_name "${EXPERIMENT_NAME}_test3" \
    --wandb_tag "grid_search_test" \
    --debug False

cd "${PROJECT_ROOT}"  # Return to project root

echo ""
echo "✓ Test 3 completed successfully"
echo ""

# ============================================================================
# Summary
# ============================================================================

echo "###############################################################################"
echo "# TEST RUN COMPLETE!"
echo "###############################################################################"
echo ""
echo "All test configurations ran successfully!"
echo ""
echo "Next steps:"
echo "  1. Review the test outputs and logs"
echo "  2. Check W&B dashboard for test runs"
echo "  3. Verify checkpoints were saved to: $CHECKPOINT_DIR"
echo "  4. Estimate full grid search time:"
echo "     - Pass 1: ~450 runs (150x the time of one test run)"
echo "     - Pass 2: ~3 runs"
echo "     - Pass 3+: ~5-10 runs (with larger dataset)"
echo ""
echo "If everything looks good, run the full grid search:"
echo "  ./grid_search_auto.sh \"$DATA_PATH\" \"$CHECKPOINT_DIR\" \"$EXPERIMENT_NAME\" 5"
echo ""
echo "Or use the interactive version:"
echo "  ./grid_search.sh \"$DATA_PATH\" \"$CHECKPOINT_DIR\" \"$EXPERIMENT_NAME\""
echo ""
