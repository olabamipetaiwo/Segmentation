#!/usr/bin/env bash
# setup.sh
# Creates a Python 3.10 virtual environment, installs PyTorch, clones and
# installs MobileSAM, then installs all remaining project dependencies.
#
# Usage:
#   bash shell/setup.sh
#
# Run this once from the project root before any other script.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/venv"
PYTHON_BIN="python3.10"

echo "============================================================"
echo " MobileSAM-LoRA Environment Setup"
echo " Project root: $PROJECT_ROOT"
echo "============================================================"

# ------------------------------------------------------------
# 1. Create virtual environment
# ------------------------------------------------------------
if [ -d "$VENV_DIR" ]; then
    echo "[1/5] Virtual environment already exists at $VENV_DIR -- skipping creation."
else
    echo "[1/5] Creating virtual environment with $PYTHON_BIN ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo "      Created: $VENV_DIR"
fi

# Activate
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
echo "      Activated: $VENV_DIR"

# Upgrade pip
pip install --upgrade pip --quiet

# ------------------------------------------------------------
# 2. Install PyTorch
# ------------------------------------------------------------
echo "[2/5] Installing PyTorch ..."
# Detect CUDA availability to pick the right index URL
if command -v nvcc &>/dev/null; then
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
    echo "      Detected CUDA $CUDA_VERSION"
    # Map to PyTorch index URL (adjust if your CUDA version differs)
    case "$CUDA_VERSION" in
        12.*) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
        11.8) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
        11.*) TORCH_INDEX="https://download.pytorch.org/whl/cu117" ;;
        *)    TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    esac
    pip install torch torchvision --index-url "$TORCH_INDEX" --quiet
else
    echo "      No CUDA detected -- installing CPU-only PyTorch."
    pip install torch torchvision --quiet
fi

# ------------------------------------------------------------
# 3. Clone and install MobileSAM
# ------------------------------------------------------------
MOBILE_SAM_DIR="$PROJECT_ROOT/MobileSAM"

echo "[3/5] Installing MobileSAM ..."
if [ -d "$MOBILE_SAM_DIR" ]; then
    echo "      MobileSAM directory already exists -- skipping clone."
else
    git clone https://github.com/ChaoningZhang/MobileSAM "$MOBILE_SAM_DIR"
fi

pip install -e "$MOBILE_SAM_DIR" --quiet
echo "      MobileSAM installed in editable mode."

# ------------------------------------------------------------
# 4. Install project dependencies
# ------------------------------------------------------------
echo "[4/5] Installing project dependencies from requirements.txt ..."
pip install -r "$PROJECT_ROOT/requirements.txt" --quiet

# ------------------------------------------------------------
# 5. Remind user to download checkpoint
# ------------------------------------------------------------
CHECKPOINT="$PROJECT_ROOT/mobile_sam.pt"

echo "[5/5] Checking for MobileSAM checkpoint ..."
if [ -f "$CHECKPOINT" ]; then
    echo "      Found: $CHECKPOINT"
else
    echo ""
    echo "  ACTION REQUIRED: mobile_sam.pt not found."
    echo "  Download it from:"
    echo "    https://github.com/ChaoningZhang/MobileSAM (Releases section)"
    echo "  and place it at:"
    echo "    $CHECKPOINT"
fi

echo ""
echo "============================================================"
echo " Setup complete."
echo " Activate the environment with:"
echo "   source venv/bin/activate"
echo "============================================================"
