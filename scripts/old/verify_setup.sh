#!/bin/bash

################################################################################
# Setup Verification Script
# 
# Run this script to verify your environment is ready for grid search
################################################################################

set +e  # Don't exit on error (we're checking)

echo "###############################################################################"
echo "# Grid Search Setup Verification"
echo "###############################################################################"
echo ""

ERRORS=0
WARNINGS=0

# ============================================================================
# Check 1: Virtual Environment
# ============================================================================
echo "[1/8] Checking virtual environment..."
if [ -d ".venv2" ]; then
    echo "  ✓ .venv2 directory exists"
    if [ -f ".venv2/bin/activate" ]; then
        echo "  ✓ Activation script found"
        source .venv2/bin/activate
        echo "  ✓ Virtual environment activated"
    else
        echo "  ✗ .venv2/bin/activate not found"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  ✗ .venv2 directory not found"
    echo "    Run: python3.12 -m venv .venv2 && source .venv2/bin/activate && pip install -e ."
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ============================================================================
# Check 2: Python Version
# ============================================================================
echo "[2/8] Checking Python version..."
if command -v python &> /dev/null; then
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    echo "  ✓ Python found: $PYTHON_VERSION"
    
    # Check if version is 3.8+
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 8 ]; then
        echo "  ✓ Python version is 3.8 or higher"
    else
        echo "  ⚠ Python version may be too old (need 3.8+)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ✗ Python not found in PATH"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ============================================================================
# Check 3: Required Python Packages
# ============================================================================
echo "[3/8] Checking required Python packages..."
REQUIRED_PACKAGES=("torch" "transformers" "peft" "trl" "datasets" "torch_geometric" "wandb")

for pkg in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $pkg" 2>/dev/null; then
        VERSION=$(python -c "import $pkg; print($pkg.__version__)" 2>/dev/null || echo "unknown")
        echo "  ✓ $pkg ($VERSION)"
    else
        echo "  ✗ $pkg not found"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# ============================================================================
# Check 4: Training Script
# ============================================================================
echo "[4/8] Checking training script..."
if [ -f "src/prism/training/train_v2.py" ]; then
    echo "  ✓ train_v2.py exists"
    
    # Check if it's executable via Python
    if python src/prism/training/train_v2.py --help &>/dev/null; then
        echo "  ✓ train_v2.py can be executed"
    else
        echo "  ⚠ train_v2.py may have syntax errors (or missing --help)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ✗ src/prism/training/train_v2.py not found"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# ============================================================================
# Check 5: Model Files
# ============================================================================
echo "[5/8] Checking model files..."
MODEL_FILES=("src/prism/models/r_pearl.py" "src/prism/models/gcn.py" "src/prism/models/gnn_llm.py")

for file in "${MODEL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✓ $file exists"
    else
        echo "  ✗ $file not found"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# ============================================================================
# Check 6: Grid Search Scripts
# ============================================================================
echo "[6/8] Checking grid search scripts..."
GRID_SCRIPTS=("grid_search.sh" "grid_search_auto.sh" "grid_search_test.sh")

for script in "${GRID_SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            echo "  ✓ $script (executable)"
        else
            echo "  ⚠ $script exists but not executable"
            echo "    Run: chmod +x $script"
            WARNINGS=$((WARNINGS + 1))
        fi
    else
        echo "  ✗ $script not found"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# ============================================================================
# Check 7: GPU Availability
# ============================================================================
echo "[7/8] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    echo "  ✓ nvidia-smi found"
    
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo "  ✓ $GPU_COUNT GPU(s) detected"
        nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader | while IFS=, read -r idx name total free; do
            echo "    GPU $idx: $name (Free: $free / Total: $total)"
        done
    else
        echo "  ⚠ No GPUs detected"
        WARNINGS=$((WARNINGS + 1))
    fi
    
    # Check CUDA availability in PyTorch
    if python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null; then
        CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null)
        echo "  ✓ PyTorch can access CUDA ($CUDA_VERSION)"
    else
        echo "  ✗ PyTorch cannot access CUDA"
        echo "    Training will be very slow on CPU!"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "  ⚠ nvidia-smi not found (are you on a GPU machine?)"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# ============================================================================
# Check 8: W&B Configuration
# ============================================================================
echo "[8/8] Checking Weights & Biases..."
if python -c "import wandb" 2>/dev/null; then
    echo "  ✓ wandb package installed"
    
    # Check if logged in
    if python -c "import wandb; wandb.api.api_key" &>/dev/null; then
        echo "  ✓ W&B API key configured"
    else
        echo "  ⚠ W&B API key not found"
        echo "    Run: wandb login"
        echo "    (Optional: use --report_to none to disable W&B)"
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo "  ⚠ wandb not installed"
    echo "    Install: pip install wandb"
    WARNINGS=$((WARNINGS + 1))
fi
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "###############################################################################"
echo "# Verification Summary"
echo "###############################################################################"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo "✓ All checks passed! You're ready to run the grid search."
    echo ""
    echo "Quick start:"
    echo "  1. Test: ./grid_search_test.sh /path/to/data.json ./test_checkpoints test"
    echo "  2. Full: ./grid_search_auto.sh /path/to/data.json ./checkpoints experiment 5"
    echo ""
    EXIT_CODE=0
elif [ $ERRORS -eq 0 ]; then
    echo "⚠ Setup is mostly ready, but there are $WARNINGS warning(s)."
    echo "  Review the warnings above before proceeding."
    echo ""
    EXIT_CODE=0
else
    echo "✗ Setup has $ERRORS error(s) and $WARNINGS warning(s)."
    echo "  Please fix the errors above before running the grid search."
    echo ""
    EXIT_CODE=1
fi

echo "For more information, see:"
echo "  - GRID_SEARCH_QUICKSTART.md"
echo "  - GRID_SEARCH_README.md"
echo ""

exit $EXIT_CODE
