#!/usr/bin/env bash
# preprocess.sh
# Generates binary masks, instance label maps, and distance maps from the
# NuInsSeg XML annotations. Must be run before training.
#
# Usage:
#   bash shell/preprocess.sh [--data_root PATH]
#
# Default data root: data/nuinsseg

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="data/nuinsseg"

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash shell/preprocess.sh [--data_root PATH]"
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

echo "============================================================"
echo " NuInsSeg Mask Generation"
echo " Data root: $DATA_ROOT"
echo "============================================================"

if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data root not found: $DATA_ROOT"
    echo "Download the NuInsSeg dataset from:"
    echo "  https://zenodo.org/records/10518968"
    echo "and extract it to: $DATA_ROOT"
    exit 1
fi

python preprocessing/generate_masks.py --data_root "$DATA_ROOT"

echo ""
echo "Preprocessing complete. You can now run bash shell/train.sh"
