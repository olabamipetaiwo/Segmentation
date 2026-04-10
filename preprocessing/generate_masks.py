"""
Generate binary masks, instance label maps, and distance maps from NuInsSeg TIF label masks.

The NuInsSeg dataset stores pre-generated instance label maps as TIF files
(uint16, each unique non-zero value = one nucleus instance).

This script walks the dataset directory, reads each TIF label mask,
and saves three output arrays per image:

    - masks/       binary foreground mask (uint8, 0/255)
    - instances/   instance label map (int32, unique ID per nucleus)
    - distances/   per-nucleus distance transform map (float32)

Usage:
    python preprocessing/generate_masks.py --data_root data/nuinsseg

NuInsSeg GitHub reference: https://github.com/masih4/NuInsSeg
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tifffile
from scipy.ndimage import distance_transform_edt


def generate_masks_from_tif(
    tif_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate binary mask, instance label map, and distance transform map from a TIF label mask.

    Inputs:
        tif_path: path to the TIF instance label map (uint16, 0 = background)
    Outputs:
        binary_mask: uint8 (H, W) with 255 for nucleus pixels, 0 for background
        instance_map: int32 (H, W) with a unique positive integer per nucleus
        distance_map: float32 (H, W) with per-nucleus EDT values peaking at each nucleus centre
    """
    instance_map = tifffile.imread(tif_path).astype(np.int32)

    binary_mask = (instance_map > 0).astype(np.uint8) * 255

    distance_map = np.zeros(instance_map.shape, dtype=np.float32)
    for nucleus_id in np.unique(instance_map):
        if nucleus_id == 0:
            continue
        nucleus_mask = instance_map == nucleus_id
        distance_map += distance_transform_edt(nucleus_mask).astype(np.float32)

    return binary_mask, instance_map, distance_map


def process_dataset(data_root: str) -> None:
    """
    Walk the NuInsSeg dataset directory and generate masks for all images.

    Handles two layouts:
      1. Subdirectory layout (full multi-tissue dataset):
           <data_root>/<tissue>/tissue images/*.png
           <data_root>/<tissue>/label masks/*.tif
         Outputs saved under each tissue subdirectory.

      2. Flat layout (single-tissue, legacy):
           <data_root>/tissue images/*.png
           <data_root>/label masks/*.tif
         Outputs saved under data_root.

    Skips images whose instance .npy already exists (resumable).

    Inputs:
        data_root: path to the root NuInsSeg directory
    Outputs:
        None (writes NPY files to disk)
    """
    data_root_path = Path(data_root)
    if not data_root_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Collect (img_path, tif_path, output_dir) triples
    triples: List[Tuple[Path, Path, Path]] = []

    # Layout 1: subdirectories per tissue
    for tissue_dir in sorted(data_root_path.iterdir()):
        if not tissue_dir.is_dir():
            continue
        sub_image_dir = tissue_dir / "tissue images"
        sub_label_dir = tissue_dir / "label masks"
        if sub_image_dir.exists() and sub_label_dir.exists():
            for tif_path in sorted(sub_label_dir.glob("*.tif")):
                img_path = sub_image_dir / (tif_path.stem + ".png")
                if img_path.exists():
                    triples.append((img_path, tif_path, tissue_dir))

    # Layout 2: flat root-level (fallback for legacy single-tissue setup)
    root_image_dir = data_root_path / "tissue images"
    root_label_dir = data_root_path / "label masks"
    if root_image_dir.exists() and root_label_dir.exists():
        for tif_path in sorted(root_label_dir.glob("*.tif")):
            img_path = root_image_dir / (tif_path.stem + ".png")
            if img_path.exists():
                triples.append((img_path, tif_path, data_root_path))

    if not triples:
        print("No image/TIF pairs found. Verify the data_root layout.")
        return

    print(f"Found {len(triples)} image/annotation pairs. Generating masks...")

    skipped = 0
    for img_path, tif_path, out_root in triples:
        instances_dir = out_root / "instances"
        instance_npy = instances_dir / f"{img_path.stem}.npy"
        if instance_npy.exists():
            skipped += 1
            continue

        masks_dir = out_root / "masks"
        distances_dir = out_root / "distances"
        for directory in (masks_dir, instances_dir, distances_dir):
            directory.mkdir(parents=True, exist_ok=True)

        try:
            binary_mask, instance_map, distance_map = generate_masks_from_tif(str(tif_path))
        except Exception as exc:
            print(f"[ERROR] Skipping {tif_path.name}: {exc}")
            continue

        stem = img_path.stem
        np.save(str(masks_dir / f"{stem}.npy"), binary_mask)
        np.save(str(instances_dir / f"{stem}.npy"), instance_map)
        np.save(str(distances_dir / f"{stem}.npy"), distance_map)

        num_nuclei = int(instance_map.max())
        print(f"  Processed: {img_path.name} ({num_nuclei} nuclei)")

    if skipped:
        print(f"  Skipped {skipped} already-processed images.")
    print("Mask generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate NuInsSeg masks from TIF label maps."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/nuinsseg",
        help="Root directory of the NuInsSeg dataset",
    )
    args = parser.parse_args()
    process_dataset(args.data_root)
