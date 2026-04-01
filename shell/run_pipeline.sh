#!/usr/bin/env bash
# run_pipeline.sh
# Runs the full end-to-end pipeline in sequence:
#   setup -> preprocess -> train (all folds) -> evaluate (all folds) -> visualize
#
# Usage:
#   bash shell/run_pipeline.sh [OPTIONS]
#
# Options:
#   --checkpoint PATH   Path to mobile_sam.pt (default: mobile_sam.pt)
#   --data_root   PATH  Root of NuInsSeg data  (default: data/nuinsseg)
#   --skip_setup        Skip environment setup (if already done)
#   --skip_preprocess   Skip mask generation   (if already done)
#
# This script is most useful for a clean first run. For iterative
# experimentation, call the individual scripts in shell/ directly.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHELL_DIR="$PROJECT_ROOT/shell"
cd "$PROJECT_ROOT"

CHECKPOINT="mobile_sam.pt"
DATA_ROOT="data/nuinsseg"
SKIP_SETUP=false
SKIP_PREPROCESS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)     CHECKPOINT="$2";  shift 2 ;;
        --data_root)      DATA_ROOT="$2";   shift 2 ;;
        --skip_setup)     SKIP_SETUP=true;  shift ;;
        --skip_preprocess) SKIP_PREPROCESS=true; shift ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo " MobileSAM-LoRA Full Pipeline"
echo " Checkpoint  : $CHECKPOINT"
echo " Data root   : $DATA_ROOT"
echo "============================================================"
echo ""

# Step 1: Environment setup
if [ "$SKIP_SETUP" = false ]; then
    echo ">>> Step 1/5: Environment setup"
    bash "$SHELL_DIR/setup.sh"
    echo ""
else
    echo ">>> Step 1/5: Environment setup [SKIPPED]"
fi

# Step 2: Preprocessing
if [ "$SKIP_PREPROCESS" = false ]; then
    echo ">>> Step 2/5: Mask generation"
    bash "$SHELL_DIR/preprocess.sh" --data_root "$DATA_ROOT"
    echo ""
else
    echo ">>> Step 2/5: Mask generation [SKIPPED]"
fi

# Step 3: Training (all 5 folds)
echo ">>> Step 3/5: Training (all folds)"
bash "$SHELL_DIR/train.sh" \
    --checkpoint "$CHECKPOINT" \
    --data_root  "$DATA_ROOT"
echo ""

# Step 4: Evaluation (all 5 folds)
echo ">>> Step 4/5: Evaluation (all folds)"
for FOLD_IDX in 0 1 2 3 4; do
    FOLD_NUM=$((FOLD_IDX + 1))
    CKPT="checkpoints/fold_${FOLD_NUM}/best_checkpoint.pt"
    if [ -f "$CKPT" ]; then
        bash "$SHELL_DIR/evaluate.sh" \
            --checkpoint "$CKPT" \
            --fold_index "$FOLD_IDX" \
            --data_root  "$DATA_ROOT" \
            --output_csv "results_fold_${FOLD_NUM}.csv"
    else
        echo "  Checkpoint not found for fold $FOLD_NUM -- skipping: $CKPT"
    fi
done
echo ""

# Aggregate results across folds
echo ">>> Cross-validation summary"
python - <<'PYEOF'
import csv, glob, numpy as np
from pathlib import Path

csv_files = sorted(glob.glob("results_fold_*.csv"))
if not csv_files:
    print("No result CSV files found.")
    exit()

metric_accum = {"dice": [], "aji": [], "pq": []}
for csv_path in csv_files:
    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            for metric in metric_accum:
                metric_accum[metric].append(float(row[metric]))

print(f"{'Metric':6s}  {'Mean':>8s}  {'Std':>8s}")
print("-" * 28)
for metric, values in metric_accum.items():
    print(f"{metric.upper():6s}  {np.mean(values):8.4f}  {np.std(values):8.4f}")
PYEOF
echo ""

# Step 5: Visualisation (fold 1 as representative example)
echo ">>> Step 5/5: Visualisation (fold 1)"
FIRST_CKPT="checkpoints/fold_1/best_checkpoint.pt"
if [ -f "$FIRST_CKPT" ]; then
    bash "$SHELL_DIR/visualize.sh" \
        --checkpoint   "$FIRST_CKPT" \
        --fold_index   0 \
        --data_root    "$DATA_ROOT" \
        --num_examples 6
else
    echo "  Checkpoint not found -- skipping visualisation."
fi

echo ""
echo "============================================================"
echo " Pipeline complete."
echo " Checkpoints : checkpoints/"
echo " Logs        : logs/"
echo " Results     : results_fold_*.csv"
echo " Figures     : figures/"
echo "============================================================"
