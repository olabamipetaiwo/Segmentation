#!/usr/bin/env bash
# visualize.sh
# Generates side-by-side comparison figures (H&E image, GT mask, prediction)
# for a sample of validation images from the given fold.
#
# Usage:
#   bash shell/visualize.sh [OPTIONS]
#
# Options:
#   --checkpoint    PATH   Trained .pt file (default: checkpoints/fold_1/best_checkpoint.pt)
#   --fold_index    INT    Zero-based fold index (default: 0)
#   --data_root     PATH   Root of NuInsSeg data (default: data/nuinsseg)
#   --figures_dir   PATH   Output directory for PNG figures (default: figures)
#   --num_examples  INT    Number of figures to generate (default: 6)
#   --grid_step     INT    Grid spacing for candidate prompts (default: 64)
#   --iou_threshold FLOAT  Minimum predicted IoU to keep a mask (default: 0.5)

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="checkpoints/fold_1/best_checkpoint.pt"
FOLD_INDEX=0
DATA_ROOT="data/nuinsseg"
FIGURES_DIR="figures"
NUM_EXAMPLES=6
GRID_STEP=64
IOU_THRESHOLD=0.5

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)    CHECKPOINT="$2";    shift 2 ;;
        --fold_index)    FOLD_INDEX="$2";    shift 2 ;;
        --data_root)     DATA_ROOT="$2";     shift 2 ;;
        --figures_dir)   FIGURES_DIR="$2";   shift 2 ;;
        --num_examples)  NUM_EXAMPLES="$2";  shift 2 ;;
        --grid_step)     GRID_STEP="$2";     shift 2 ;;
        --iou_threshold) IOU_THRESHOLD="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

VENV_DIR="$PROJECT_ROOT/venv"
if [ -d "$VENV_DIR" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment not found. Run bash shell/setup.sh first."
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Run bash shell/train.sh first."
    exit 1
fi

mkdir -p "$FIGURES_DIR"

echo "============================================================"
echo " MobileSAM-LoRA Visualisation"
echo " Checkpoint    : $CHECKPOINT"
echo " Fold index    : $FOLD_INDEX"
echo " Data root     : $DATA_ROOT"
echo " Output dir    : $FIGURES_DIR"
echo " Num examples  : $NUM_EXAMPLES"
echo "============================================================"

python visualize.py \
    --checkpoint    "$CHECKPOINT" \
    --fold_index    "$FOLD_INDEX" \
    --data_root     "$DATA_ROOT" \
    --figures_dir   "$FIGURES_DIR" \
    --num_examples  "$NUM_EXAMPLES" \
    --grid_step     "$GRID_STEP" \
    --iou_threshold "$IOU_THRESHOLD"

echo ""
echo "Figures saved to: $FIGURES_DIR/"
