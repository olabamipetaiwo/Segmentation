#!/usr/bin/env bash
# evaluate.sh
# Runs inference on the held-out fold of a trained checkpoint and reports
# Dice, AJI, and PQ metrics. Saves per-image results to a CSV.
#
# Usage:
#   bash shell/evaluate.sh [OPTIONS]
#
# Options:
#   --checkpoint    PATH   Trained .pt file (default: checkpoints/fold_1/best_checkpoint.pt)
#   --fold_index    INT    Zero-based fold index matching the checkpoint (default: 0)
#   --data_root     PATH   Root of NuInsSeg data (default: data/nuinsseg)
#   --grid_step     INT    Grid spacing for candidate prompts in pixels (default: 64)
#   --iou_threshold FLOAT  Minimum predicted IoU to keep a mask (default: 0.5)
#   --output_csv    PATH   Where to write the per-image results (default: results.csv)
#
# Example (evaluate all five folds):
#   for i in 0 1 2 3 4; do
#     bash shell/evaluate.sh \
#       --checkpoint "checkpoints/fold_$((i+1))/best_checkpoint.pt" \
#       --fold_index $i \
#       --output_csv "results_fold_$((i+1)).csv"
#   done

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CHECKPOINT="checkpoints/fold_1/best_checkpoint.pt"
FOLD_INDEX=0
DATA_ROOT="data/nuinsseg"
GRID_STEP=64
IOU_THRESHOLD=0.5
OUTPUT_CSV="results.csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)    CHECKPOINT="$2";    shift 2 ;;
        --fold_index)    FOLD_INDEX="$2";    shift 2 ;;
        --data_root)     DATA_ROOT="$2";     shift 2 ;;
        --grid_step)     GRID_STEP="$2";     shift 2 ;;
        --iou_threshold) IOU_THRESHOLD="$2"; shift 2 ;;
        --output_csv)    OUTPUT_CSV="$2";    shift 2 ;;
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

echo "============================================================"
echo " MobileSAM-LoRA Evaluation"
echo " Checkpoint    : $CHECKPOINT"
echo " Fold index    : $FOLD_INDEX"
echo " Data root     : $DATA_ROOT"
echo " Grid step     : $GRID_STEP px"
echo " IoU threshold : $IOU_THRESHOLD"
echo " Output CSV    : $OUTPUT_CSV"
echo "============================================================"

python evaluate.py \
    --checkpoint    "$CHECKPOINT" \
    --fold_index    "$FOLD_INDEX" \
    --data_root     "$DATA_ROOT" \
    --grid_step     "$GRID_STEP" \
    --iou_threshold "$IOU_THRESHOLD" \
    --output_csv    "$OUTPUT_CSV"

echo ""
echo "Evaluation complete. Results saved to: $OUTPUT_CSV"
