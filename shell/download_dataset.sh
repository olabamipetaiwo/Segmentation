#!/usr/bin/env bash
# download_dataset.sh
# Downloads the NuInsSeg dataset from Zenodo and extracts it to data/nuinsseg/.
#
# Usage:
#   bash shell/download_dataset.sh [--data_root PATH]
#
# Options:
#   --data_root PATH   Destination directory (default: data/nuinsseg)
#
# Requires: curl or wget, unzip

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT="data/nuinsseg"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_root)
            DATA_ROOT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash shell/download_dataset.sh [--data_root PATH]"
            exit 1
            ;;
    esac
done

ZENODO_URL="https://zenodo.org/api/records/10518968/files/NuInsSeg.zip/content"
ZIP_PATH="$PROJECT_ROOT/data/NuInsSeg.zip"

echo "============================================================"
echo " NuInsSeg Dataset Download"
echo " Source : Zenodo record 10518968"
echo " Dest   : $DATA_ROOT"
echo " Size   : ~1.6 GB"
echo "============================================================"

mkdir -p "$PROJECT_ROOT/data"
mkdir -p "$DATA_ROOT"

# Download
if [ -f "$ZIP_PATH" ]; then
    echo "Zip already exists at $ZIP_PATH -- skipping download."
else
    echo "Downloading NuInsSeg.zip ..."
    if command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$ZIP_PATH" "$ZENODO_URL"
    elif command -v wget &>/dev/null; then
        wget --show-progress -O "$ZIP_PATH" "$ZENODO_URL"
    else
        echo "ERROR: neither curl nor wget is available."
        exit 1
    fi
    echo "Download complete: $ZIP_PATH"
fi

# Extract
echo "Extracting to $DATA_ROOT ..."
if command -v unzip &>/dev/null; then
    unzip -q "$ZIP_PATH" -d "$PROJECT_ROOT/data/nuinsseg_raw"
else
    echo "ERROR: unzip is not available."
    exit 1
fi

# The zip contains a top-level NuInsSeg/ folder -- move its contents to DATA_ROOT
EXTRACTED_DIR="$PROJECT_ROOT/data/nuinsseg_raw"
INNER_DIR=$(find "$EXTRACTED_DIR" -mindepth 1 -maxdepth 1 -type d | head -1)

if [ -n "$INNER_DIR" ]; then
    echo "Moving contents of $(basename "$INNER_DIR") -> $DATA_ROOT ..."
    cp -r "$INNER_DIR"/. "$DATA_ROOT/"
    rm -rf "$EXTRACTED_DIR"
else
    # Flat zip: contents already in nuinsseg_raw, just rename
    cp -r "$EXTRACTED_DIR"/. "$DATA_ROOT/"
    rm -rf "$EXTRACTED_DIR"
fi

# Verify
IMAGE_COUNT=$(find "$DATA_ROOT" -name "*.png" | wc -l | tr -d ' ')
echo ""
echo "============================================================"
echo " Extraction complete."
echo " Images found : $IMAGE_COUNT PNG files"
echo " Location     : $DATA_ROOT"
echo ""
echo " Next step: bash shell/preprocess.sh"
echo "============================================================"
