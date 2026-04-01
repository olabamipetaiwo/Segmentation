#!/usr/bin/env bash
# train.sh
# Runs five-fold cross-validation training of MobileSAM-LoRA on NuInsSeg.
#
# Usage:
#   bash shell/train.sh [OPTIONS]
#
# Options:
#   --checkpoint PATH    Path to mobile_sam.pt  (default: mobile_sam.pt)
#   --data_root   PATH   Root of NuInsSeg data   (default: data/nuinsseg)
#   --fold        INT    Train only this fold (0-indexed). Omit for all folds.
#
# Examples:
#   bash shell/train.sh
#   bash shell/train.sh --fold 0
#   bash shell/train.sh --checkpoint /path/to/mobile_sam.pt --data_root /data/nuinsseg

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="mobile_sam.pt"
DATA_ROOT="data/nuinsseg"
FOLD_ARG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --fold)
            FOLD_ARG="--fold $2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash shell/train.sh [--checkpoint PATH] [--data_root PATH] [--fold INT]"
            exit 1
            ;;
    esac
done

# Activate virtual environment
VENV_DIR="$PROJECT_ROOT/venv"
if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment not found. Run bash shell/setup.sh first."
    exit 1
fi

# Pre-flight checks
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: MobileSAM checkpoint not found: $CHECKPOINT"
    echo "Download mobile_sam.pt from the MobileSAM GitHub and place it in:"
    echo "  $PROJECT_ROOT"
    exit 1
fi

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data root not found: $DATA_ROOT"
    echo "Run bash shell/preprocess.sh first."
    exit 1
fi

echo "============================================================"
echo " MobileSAM-LoRA Training"
echo " Checkpoint : $CHECKPOINT"
echo " Data root  : $DATA_ROOT"
echo " Fold arg   : ${FOLD_ARG:-(all folds)}"
echo " Device     : $(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
echo "============================================================"

# Create output directories
mkdir -p checkpoints logs

# Run training
# shellcheck disable=SC2086
python train.py \
    --checkpoint "$CHECKPOINT" \
    --data_root  "$DATA_ROOT" \
    $FOLD_ARG

echo ""
echo "Training complete."
echo "Checkpoints saved to: checkpoints/"
echo "Logs saved to:        logs/"
