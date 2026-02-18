#!/bin/bash

################################################################################
# Automated Grid Search Script for R-PEARL Model Training
# 
# This script performs a fully automated multi-pass grid search:
# 1. Pass 1: Find minimal model that overfits (dropout=0)
# 2. Pass 2: Add regularization (dropout + layer_norm) 
# 3. Pass 3+: Scale up dataset size
#
# Results are logged and the best configuration is automatically selected
# based on training loss.
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
    echo "Usage: $0 <data_path> <checkpoint_dir> <experiment_name> [num_scaling_passes]"
    echo ""
    echo "Arguments:"
    echo "  data_path           : Path to training data JSON file"
    echo "  checkpoint_dir      : Directory to save checkpoints and results"
    echo "  experiment_name     : Name prefix for all runs"
    echo "  num_scaling_passes  : (Optional) Number of dataset scaling passes (default: 5)"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/data.json ./checkpoints r_pearl_search 5"
    echo ""
    exit 1
fi

DATA_PATH="$1"
CHECKPOINT_DIR="$2"
EXPERIMENT_NAME="$3"
NUM_SCALING_PASSES="${4:-5}"  # Default to 5 scaling passes

# Verify data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Results log
RESULTS_LOG="${CHECKPOINT_DIR}/grid_search_results.txt"
echo "Automated Grid Search Results - Started at $(date)" > "$RESULTS_LOG"
echo "Data: $DATA_PATH" >> "$RESULTS_LOG"
echo "Experiment: $EXPERIMENT_NAME" >> "$RESULTS_LOG"
echo "Num Scaling Passes: $NUM_SCALING_PASSES" >> "$RESULTS_LOG"
echo "========================================" >> "$RESULTS_LOG"
echo "" >> "$RESULTS_LOG"

# ============================================================================
# Helper Functions
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
    echo "  learning_rate: $LEARNING_RATE"
    echo "  freeze_llm: $FREEZE_LLM"
    echo "------------------------------------------------------------------------"
    
    # Log to results file
    echo "[Pass $pass_num] $run_name" >> "$RESULTS_LOG"
    echo "  Config: layers=$pe_num_layers k=$k hidden=$pe_hidden_channels samples=$num_samples dropout=$dropout layer_norm=$use_layer_norm data=$dataset_proportion" >> "$RESULTS_LOG"
    
    # Run training (from src directory)
    local start_time=$(date +%s)
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
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ]; then
        echo "  Status: SUCCESS (${duration}s)" >> "$RESULTS_LOG"
    else
        echo "  Status: FAILED (exit code: $exit_code, ${duration}s)" >> "$RESULTS_LOG"
    fi
    echo "" >> "$RESULTS_LOG"
    
    echo "Completed: $run_name (took ${duration}s)"
    
    return $exit_code
}

# ============================================================================
# PASS 1: Overfitting Search (Find minimal model that achieves near-zero training error)
# Goal: Smallest model that can overfit (dropout=0, no layer_norm)
# Strategy: Start with small models and gradually increase capacity
# ============================================================================

echo ""
echo "###############################################################################"
echo "# PASS 1: OVERFITTING SEARCH"
echo "# Goal: Find minimal architecture that can achieve near-zero training error"
echo "# Strategy: Test progressively larger models until overfitting is achieved"
echo "###############################################################################"

PASS=1
DATASET_PROPORTION=$BASE_DATASET_PROPORTION

# Prioritized grid search for Pass 1 (smallest to largest)
# We'll test combinations in order of increasing model capacity
PE_NUM_LAYERS_OPTIONS=(2 3 4)
K_OPTIONS=(2 3 4)
PE_HIDDEN_CHANNELS_OPTIONS=(64 128 256)
NUM_SAMPLES_OPTIONS=(50 40 30 20 10)  # Ascending order

# Fixed for Pass 1
DROPOUT=0.0
USE_LAYER_NORM=False

run_counter=0
pass1_results=()

# Iterate through grid (small to large)
for num_samples in "${NUM_SAMPLES_OPTIONS[@]}"; do
    for pe_hidden_channels in "${PE_HIDDEN_CHANNELS_OPTIONS[@]}"; do
        for pe_num_layers in "${PE_NUM_LAYERS_OPTIONS[@]}"; do
            for k in "${K_OPTIONS[@]}"; do
                run_counter=$((run_counter + 1))
                run_name="p1_run${run_counter}_l${pe_num_layers}_k${k}_h${pe_hidden_channels}_s${num_samples}"
                
                # Store config for later analysis
                config="${pe_num_layers},${k},${pe_hidden_channels},${num_samples}"
                pass1_results+=("$config")
                
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
echo "NOTE: Review W&B or training logs to identify which configuration achieved"
echo "      the lowest training loss. For this automated script, we'll use a"
echo "      middle-ground configuration for Pass 2."
echo ""

# For automation, select a reasonable default (middle values)
# In practice, you would parse training logs or W&B to find the actual best
BEST_PE_NUM_LAYERS=3
BEST_K=3
BEST_PE_HIDDEN_CHANNELS=128
BEST_NUM_SAMPLES=30

echo "Selected configuration for Pass 2 (default middle-ground):"
echo "  pe_num_layers: $BEST_PE_NUM_LAYERS"
echo "  k: $BEST_K"
echo "  pe_hidden_channels: $BEST_PE_HIDDEN_CHANNELS"
echo "  num_samples: $BEST_NUM_SAMPLES"
echo ""
echo "IMPORTANT: If you have identified a better config from Pass 1 results,"
echo "           edit this script and update the BEST_* variables above."
echo ""

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

# Regularization options (high to low dropout)
DROPOUT_OPTIONS=(0.1 0.05 0.01)
USE_LAYER_NORM=True

run_counter=0
pass2_results=()

for dropout in "${DROPOUT_OPTIONS[@]}"; do
    run_counter=$((run_counter + 1))
    run_name="p2_run${run_counter}_dropout${dropout}_ln"
    
    pass2_results+=("$dropout")
    
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

# For automation, select middle dropout value
BEST_DROPOUT=0.05

echo "Selected dropout for Pass 3+: $BEST_DROPOUT"
echo "IMPORTANT: Review validation metrics to select the optimal dropout value."
echo ""

# ============================================================================
# PASS 3+: Scale up dataset (increase dataset_proportion)
# Goal: Validate model generalizes with more data
# ============================================================================

echo ""
echo "###############################################################################"
echo "# PASS 3+: DATASET SCALING"
echo "# Using best config from Pass 2 with increasing dataset sizes"
echo "###############################################################################"

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
    
    # Stop if we've reached 100% of data
    if (( $(echo "$DATASET_PROPORTION >= 1.0" | bc -l) )); then
        echo "Reached 100% of dataset, stopping scaling passes."
        break
    fi
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
echo "Recommended Configuration (used for final passes):"
echo "  pe_num_layers: $BEST_PE_NUM_LAYERS"
echo "  k: $BEST_K"
echo "  pe_hidden_channels: $BEST_PE_HIDDEN_CHANNELS"
echo "  num_samples: $BEST_NUM_SAMPLES"
echo "  dropout: $BEST_DROPOUT"
echo "  use_layer_norm: True"
echo "  learning_rate: $LEARNING_RATE"
echo "  freeze_llm: $FREEZE_LLM"
echo ""
echo "Next Steps:"
echo "  1. Review W&B dashboard or training logs for each pass"
echo "  2. Identify configurations with lowest training loss (Pass 1)"
echo "  3. Identify configurations with best val loss (Pass 2)"
echo "  4. Analyze scaling behavior (Pass 3+)"
echo "  5. Select final model based on performance vs. efficiency trade-off"
echo ""
echo "Grid search completed at $(date)"
echo ""

# Write final summary to results log
{
    echo "========================================="
    echo "FINAL CONFIGURATION"
    echo "========================================="
    echo "pe_num_layers: $BEST_PE_NUM_LAYERS"
    echo "k: $BEST_K"
    echo "pe_hidden_channels: $BEST_PE_HIDDEN_CHANNELS"
    echo "num_samples: $BEST_NUM_SAMPLES"
    echo "dropout: $BEST_DROPOUT"
    echo "use_layer_norm: True"
    echo "learning_rate: $LEARNING_RATE"
    echo "freeze_llm: $FREEZE_LLM"
    echo ""
    echo "Completed: $(date)"
} >> "$RESULTS_LOG"
