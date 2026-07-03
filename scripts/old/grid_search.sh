#!/bin/bash

################################################################################
# Grid Search Script for R-PEARL Model Training
# 
# This script performs a multi-pass grid search to:
# 1. Pass 1: Find minimal model that overfits (dropout=0)
# 2. Pass 2: Add regularization (dropout + layer_norm)
# 3. Pass 3+: Scale up dataset size
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
BASE_DATASET_PROPORTION=0.10
EPOCHS=2
D_MODEL=3072  # Must match Llama-3.2-3B embedding dimension

# Required arguments (must be provided by user)
if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Usage: $0 <data_path> <checkpoint_dir> <experiment_name>"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/data.json ./checkpoints r_pearl_search"
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

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Results log
RESULTS_LOG="${CHECKPOINT_DIR}/grid_search_results.txt"
echo "Grid Search Results - Started at $(date)" > "$RESULTS_LOG"
echo "Data: $DATA_PATH" >> "$RESULTS_LOG"
echo "=" >> "$RESULTS_LOG"
echo "" >> "$RESULTS_LOG"

# ============================================================================
# Helper Function
# ============================================================================

run_training() {
    local pass_num=$1
    local run_name=$2
    local pe_num_layers=$3
    local k=$4
    local pe_hidden_channels=$5
    local num_samples=$6
    local dropout=$7
    local use_layer_norm=$8
    local dataset_proportion=$9
    
    echo ""
    echo "========================================================================"
    echo "Pass $pass_num: $run_name"
    echo "========================================================================"
    echo "Parameters:"
    echo "  pe_num_layers: $pe_num_layers"
    echo "  k: $k"
    echo "  pe_hidden_channels: $pe_hidden_channels"
    echo "  num_samples: $num_samples"
    echo "  dropout: $dropout"
    echo "  use_layer_norm: $use_layer_norm"
    echo "  dataset_proportion: $dataset_proportion"
    echo "------------------------------------------------------------------------"
    
    # Log to results file
    echo "[Pass $pass_num] $run_name" >> "$RESULTS_LOG"
    echo "  Config: layers=$pe_num_layers k=$k hidden=$pe_hidden_channels samples=$num_samples dropout=$dropout layer_norm=$use_layer_norm data=$dataset_proportion" >> "$RESULTS_LOG"
    
    # Run training (from src directory)
    cd "${PROJECT_ROOT}/src"
    python "$TRAIN_SCRIPT" \
        --name "${EXPERIMENT_NAME}_${run_name}" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --data "$DATA_PATH" \
        --learning_rate $LEARNING_RATE \
        --freeze_llm $FREEZE_LLM \
        --dataset_proportion $dataset_proportion \
        --pe_num_layers $pe_num_layers \
        --k $k \
        --d_model $D_MODEL \
        --pe_hidden_channels $pe_hidden_channels \
        --num_samples $num_samples \
        --dropout $dropout \
        --use_layer_norm $use_layer_norm \
        --epochs $EPOCHS \
        --wandb_run_name "${EXPERIMENT_NAME}_${run_name}" \
        --wandb_tag "grid_search_pass${pass_num}" \
        --debug False
    
    local exit_code=$?
    cd "${PROJECT_ROOT}"  # Return to project root
    if [ $exit_code -eq 0 ]; then
        echo "  Status: SUCCESS" >> "$RESULTS_LOG"
    else
        echo "  Status: FAILED (exit code: $exit_code)" >> "$RESULTS_LOG"
    fi
    echo "" >> "$RESULTS_LOG"
    
    echo "Completed: $run_name"
}

# ============================================================================
# PASS 1: Overfitting Search (Find minimal model that achieves near-zero training error)
# Goal: Smallest model that can overfit (dropout=0, no layer_norm)
# ============================================================================

echo ""
echo "###############################################################################"
echo "# PASS 1: OVERFITTING SEARCH"
echo "# Goal: Find minimal architecture that can achieve near-zero training error"
echo "###############################################################################"

PASS=1
DATASET_PROPORTION=$BASE_DATASET_PROPORTION

# Grid search parameters for Pass 1
PE_NUM_LAYERS_OPTIONS=(2 3 4)
K_OPTIONS=(1 2 3 4 5)
PE_HIDDEN_CHANNELS_OPTIONS=(64 128 256)
NUM_SAMPLES_OPTIONS=(50 40 30 20 10)

# Fixed for Pass 1
DROPOUT=0.0
USE_LAYER_NORM=False

run_counter=0

# Iterate through grid
for pe_num_layers in "${PE_NUM_LAYERS_OPTIONS[@]}"; do
    for k in "${K_OPTIONS[@]}"; do
        for pe_hidden_channels in "${PE_HIDDEN_CHANNELS_OPTIONS[@]}"; do
            for num_samples in "${NUM_SAMPLES_OPTIONS[@]}"; do
                run_counter=$((run_counter + 1))
                run_name="p1_run${run_counter}_l${pe_num_layers}_k${k}_h${pe_hidden_channels}_s${num_samples}"
                
                run_training \
                    $PASS \
                    "$run_name" \
                    $pe_num_layers \
                    $k \
                    $pe_hidden_channels \
                    $num_samples \
                    $DROPOUT \
                    $USE_LAYER_NORM \
                    $DATASET_PROPORTION
            done
        done
    done
done

echo ""
echo "###############################################################################"
echo "# PASS 1 COMPLETE"
echo "# Total runs: $run_counter"
echo "###############################################################################"
echo ""
echo "Please review the results and identify the best configuration from Pass 1."
echo "Edit the script to set BEST_* variables below for Pass 2."
echo ""
read -p "Enter BEST pe_num_layers from Pass 1: " BEST_PE_NUM_LAYERS
read -p "Enter BEST k from Pass 1: " BEST_K
read -p "Enter BEST pe_hidden_channels from Pass 1: " BEST_PE_HIDDEN_CHANNELS
read -p "Enter BEST num_samples from Pass 1: " BEST_NUM_SAMPLES

# ============================================================================
# PASS 2: Regularization Search (Add dropout and layer_norm)
# Goal: Prevent overfitting while maintaining performance
# ============================================================================

echo ""
echo "###############################################################################"
echo "# PASS 2: REGULARIZATION SEARCH"
echo "# Using best config from Pass 1 with added regularization"
echo "###############################################################################"

PASS=2
DATASET_PROPORTION=$BASE_DATASET_PROPORTION

# Regularization options
DROPOUT_OPTIONS=(0.1 0.05 0.01)
USE_LAYER_NORM=True

run_counter=0

for dropout in "${DROPOUT_OPTIONS[@]}"; do
    run_counter=$((run_counter + 1))
    run_name="p2_run${run_counter}_dropout${dropout}_ln"
    
    run_training \
        $PASS \
        "$run_name" \
        $BEST_PE_NUM_LAYERS \
        $BEST_K \
        $BEST_PE_HIDDEN_CHANNELS \
        $BEST_NUM_SAMPLES \
        $dropout \
        $USE_LAYER_NORM \
        $DATASET_PROPORTION
done

echo ""
echo "###############################################################################"
echo "# PASS 2 COMPLETE"
echo "# Total runs: $run_counter"
echo "###############################################################################"
echo ""
echo "Please review the results and select the best dropout value for Pass 3+."
echo ""
read -p "Enter BEST dropout from Pass 2: " BEST_DROPOUT

# ============================================================================
# PASS 3+: Scale up dataset (increase dataset_proportion)
# Goal: Validate model generalizes with more data
# ============================================================================

echo ""
echo "###############################################################################"
echo "# PASS 3+: DATASET SCALING"
echo "# Using best config from Pass 2 with increasing dataset sizes"
echo "###############################################################################"

# Number of scaling passes (3, 4, 5, ...)
read -p "How many additional passes with increasing data? (e.g., 5 for passes 3-7): " NUM_SCALING_PASSES

for i in $(seq 1 $NUM_SCALING_PASSES); do
    PASS=$((i + 2))  # Pass 3, 4, 5, ...
    DATASET_PROPORTION=$(echo "scale=2; 0.1 * ($PASS - 1)" | bc)
    
    # Cap at 1.0 (100% of data)
    if (( $(echo "$DATASET_PROPORTION > 1.0" | bc -l) )); then
        DATASET_PROPORTION=1.0
    fi
    
    echo ""
    echo "###############################################################################"
    echo "# PASS $PASS: DATASET PROPORTION = ${DATASET_PROPORTION}"
    echo "###############################################################################"
    
    run_name="p${PASS}_data${DATASET_PROPORTION}"
    
    run_training \
        $PASS \
        "$run_name" \
        $BEST_PE_NUM_LAYERS \
        $BEST_K \
        $BEST_PE_HIDDEN_CHANNELS \
        $BEST_NUM_SAMPLES \
        $BEST_DROPOUT \
        $USE_LAYER_NORM \
        $DATASET_PROPORTION
done

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "###############################################################################"
echo "# GRID SEARCH COMPLETE!"
echo "###############################################################################"
echo ""
echo "Results logged to: $RESULTS_LOG"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo ""
echo "Final Best Configuration:"
echo "  pe_num_layers: $BEST_PE_NUM_LAYERS"
echo "  k: $BEST_K"
echo "  pe_hidden_channels: $BEST_PE_HIDDEN_CHANNELS"
echo "  num_samples: $BEST_NUM_SAMPLES"
echo "  dropout: $BEST_DROPOUT"
echo "  use_layer_norm: True"
echo ""
echo "Grid search completed at $(date)"
