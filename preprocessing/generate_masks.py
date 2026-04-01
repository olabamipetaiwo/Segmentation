"""
Generate binary masks, instance label maps, and distance maps from NuInsSeg XML annotations.

The NuInsSeg dataset stores polygon annotations in XML files (one per image).
This script walks the dataset directory, parses each XML annotation file,
rasterises the nucleus polygons, and saves three output arrays per image:

    - masks/       binary foreground mask (uint8, 0/255)
    - instances/   instance label map (int32, unique ID per nucleus)
    - distances/   per-nucleus distance transform map (float32)

Usage:
    python preprocessing/generate_masks.py --data_root data/nuinsseg

NuInsSeg GitHub reference: https://github.com/masih4/NuInsSeg
"""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


# ---------------------------------------------------------------------------
# XML parsing
# ---------------------------------------------------------------------------

def parse_xml_annotations(xml_path: str) -> List[np.ndarray]:
    """
    Parse a NuInsSeg XML annotation file and return nucleus polygon coordinates.

    The XML format contains one or more Annotation elements, each holding
    Region elements with Vertex children that define polygon outlines.

    Inputs:
        xml_path: path to the XML annotation file
    Outputs:
        list of float32 numpy arrays of shape (N_vertices, 2) in (X, Y) order,
        one array per nucleus polygon
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    polygons: List[np.ndarray] = []

    # Support both flat <Annotation> and nested <Annotations><Annotation> layouts
    for region in root.iter("Region"):
        vertices: List[List[float]] = []
        for vertex in region.iter("Vertex"):
            x_val = float(vertex.get("X", 0))
            y_val = float(vertex.get("Y", 0))
            vertices.append([x_val, y_val])
        if len(vertices) >= 3:
            polygons.append(np.array(vertices, dtype=np.float32))

    return polygons


# ---------------------------------------------------------------------------
# Mask generation
# ---------------------------------------------------------------------------

def generate_masks_for_image(
    image_path: str,
    xml_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate binary mask, instance label map, and distance transform map for one image.

    Inputs:
        image_path: path to the source PNG image
        xml_path: path to the corresponding XML annotation file
    Outputs:
        binary_mask: uint8 numpy array (H, W) with 255 for nucleus pixels, 0 for background
        instance_map: int32 numpy array (H, W) with a unique positive integer per nucleus
        distance_map: float32 numpy array (H, W) with per-nucleus EDT values peaking
                      at each nucleus centre
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    H, W = image.shape[:2]

    polygons = parse_xml_annotations(xml_path)

    binary_mask = np.zeros((H, W), dtype=np.uint8)
    instance_map = np.zeros((H, W), dtype=np.int32)
    distance_map = np.zeros((H, W), dtype=np.float32)

    for nucleus_id, polygon in enumerate(polygons, start=1):
        # Round and reshape polygon for OpenCV (N, 1, 2)
        poly_int = np.round(polygon).astype(np.int32).reshape(-1, 1, 2)

        # Draw filled polygon for binary mask
        cv2.fillPoly(binary_mask, [poly_int], color=255)

        # Draw filled polygon for instance map with unique ID
        nucleus_binary = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(nucleus_binary, [poly_int], color=1)
        instance_map[nucleus_binary == 1] = nucleus_id

        # Per-nucleus distance transform (peaks at centre of each nucleus)
        nucleus_float = nucleus_binary.astype(bool)
        nucleus_distance = distance_transform_edt(nucleus_float)
        distance_map += nucleus_distance.astype(np.float32)

    return binary_mask, instance_map, distance_map


# ---------------------------------------------------------------------------
# Directory traversal
# ---------------------------------------------------------------------------

def process_dataset(data_root: str) -> None:
    """
    Walk the NuInsSeg dataset directory and generate masks for all images.

    Expected input layout (per tissue type):
        <data_root>/<tissue_type>/tissue images/*.png
        <data_root>/<tissue_type>/mask/*.xml

    Alternatively, if images and XML files share a directory, they are
    matched by stem name.

    Output directories are created under each tissue type folder:
        <data_root>/<tissue_type>/masks/       binary mask NPY files
        <data_root>/<tissue_type>/instances/   instance map NPY files
        <data_root>/<tissue_type>/distances/   distance map NPY files

    Inputs:
        data_root: path to the root NuInsSeg directory
    Outputs:
        None (writes NPY files to disk)
    """
    data_root_path = Path(data_root)
    if not data_root_path.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Collect (image_path, xml_path) pairs
    pairs: List[Tuple[Path, Path]] = []

    for tissue_dir in sorted(data_root_path.iterdir()):
        if not tissue_dir.is_dir():
            continue

        # Layout 1: tissue images/ and mask/ subdirectories
        image_dir = tissue_dir / "tissue images"
        xml_dir = tissue_dir / "mask"

        if image_dir.exists() and xml_dir.exists():
            for img_path in sorted(image_dir.glob("*.png")):
                xml_path = xml_dir / (img_path.stem + ".xml")
                if xml_path.exists():
                    pairs.append((img_path, xml_path))
                else:
                    print(f"[WARN] No XML found for: {img_path}")

        else:
            # Layout 2: images and XML in the same directory
            for img_path in sorted(tissue_dir.rglob("*.png")):
                xml_path = img_path.with_suffix(".xml")
                if xml_path.exists():
                    pairs.append((img_path, xml_path))

    if not pairs:
        print("No image/XML pairs found. Verify the data_root layout.")
        return

    print(f"Found {len(pairs)} image/annotation pairs. Generating masks...")

    for img_path, xml_path in pairs:
        tissue_dir = img_path.parent.parent if "tissue" in img_path.parent.name else img_path.parent

        masks_dir = tissue_dir / "masks"
        instances_dir = tissue_dir / "instances"
        distances_dir = tissue_dir / "distances"

        for directory in (masks_dir, instances_dir, distances_dir):
            directory.mkdir(parents=True, exist_ok=True)

        try:
            binary_mask, instance_map, distance_map = generate_masks_for_image(
                str(img_path), str(xml_path)
            )
        except Exception as exc:
            print(f"[ERROR] Skipping {img_path.name}: {exc}")
            continue

        stem = img_path.stem
        np.save(str(masks_dir / f"{stem}.npy"), binary_mask)
        np.save(str(instances_dir / f"{stem}.npy"), instance_map)
        np.save(str(distances_dir / f"{stem}.npy"), distance_map)

        num_nuclei = int(instance_map.max())
        print(f"  Processed: {img_path.name} ({num_nuclei} nuclei)")

    print("Mask generation complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate NuInsSeg masks from XML annotations."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/nuinsseg",
        help="Root directory of the NuInsSeg dataset",
    )
    args = parser.parse_args()
    process_dataset(args.data_root)
