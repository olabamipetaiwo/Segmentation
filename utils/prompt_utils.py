"""
Utility functions for constructing SAM point and box prompts from instance label maps.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from skimage.measure import regionprops


def extract_centroids_from_instances(
    instance_map: np.ndarray,
) -> List[Tuple[float, float]]:
    """
    Extract nucleus centroid coordinates from an instance label map.

    Uses skimage regionprops to find the centroid of each uniquely labelled nucleus.

    Inputs:
        instance_map: integer numpy array of shape (H, W) where each nucleus
                      occupies pixels with a unique positive integer ID and
                      background is 0
    Outputs:
        list of (row, col) float tuples, one per nucleus, in image pixel coordinates
    """
    props = regionprops(instance_map)
    centroids = [(float(prop.centroid[0]), float(prop.centroid[1])) for prop in props]
    return centroids


def build_point_prompts(
    centroids: List[Tuple[float, float]],
    neg_centroids: Optional[List[Tuple[float, float]]] = None,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build point prompt tensors in the format expected by MobileSAM's prompt encoder.

    SAM uses (x, y) = (col, row) coordinate ordering.

    Inputs:
        centroids: list of (row, col) float tuples representing nucleus centres
        neg_centroids: optional list of (row, col) background points, one per
                       centroid. When provided, each prompt gets 2 points:
                       a foreground centre (label 1) and a background point
                       (label 0), which sharpens boundary delineation.
        device: torch device string on which to place the output tensors
    Outputs:
        point_coords: float32 tensor of shape (N, P, 2) where P=1 (pos only)
                      or P=2 (pos + neg), with coordinates in (col, row) order
        point_labels: int32 tensor of shape (N, P)
    """
    if len(centroids) == 0:
        point_coords = torch.zeros(0, 1, 2, dtype=torch.float32, device=device)
        point_labels = torch.zeros(0, 1, dtype=torch.int32, device=device)
        return point_coords, point_labels

    # Convert (row, col) to SAM's expected (x, y) = (col, row)
    pos_coords = [[col, row] for row, col in centroids]

    if neg_centroids is not None:
        neg_coords = [[col, row] for row, col in neg_centroids]
        # (N, 2, 2): each prompt has [positive_point, negative_point]
        point_coords = torch.tensor(
            [[p, n] for p, n in zip(pos_coords, neg_coords)],
            dtype=torch.float32,
            device=device,
        )  # (N, 2, 2)
        point_labels = torch.tensor(
            [[1, 0]] * len(centroids), dtype=torch.int32, device=device
        )  # (N, 2)
    else:
        point_coords = (
            torch.tensor(pos_coords, dtype=torch.float32, device=device).unsqueeze(1)
        )  # (N, 1, 2)
        point_labels = torch.ones(
            len(centroids), 1, dtype=torch.int32, device=device
        )  # (N, 1)

    return point_coords, point_labels


def extract_boxes_from_instances(
    instance_map: np.ndarray,
) -> Tuple[List[Tuple[float, float, float, float]], List[int]]:
    """
    Extract axis-aligned bounding boxes and instance IDs from an instance label map.

    Inputs:
        instance_map: integer numpy array of shape (H, W) where each nucleus
                      occupies pixels with a unique positive integer ID and
                      background is 0
    Outputs:
        boxes: list of (row_min, col_min, row_max, col_max) float tuples
        instance_ids: list of integer nucleus IDs, one per box
    """
    props = regionprops(instance_map)
    boxes = [
        (float(p.bbox[0]), float(p.bbox[1]), float(p.bbox[2]), float(p.bbox[3]))
        for p in props
    ]
    instance_ids = [p.label for p in props]
    return boxes, instance_ids


def build_box_prompts(
    boxes: List[Tuple[float, float, float, float]],
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a box prompt tensor in the format expected by SAM's prompt encoder.

    SAM uses (x, y) = (col, row) coordinate ordering and expects boxes as
    [x1, y1, x2, y2] = [col_min, row_min, col_max, row_max].

    Inputs:
        boxes: list of (row_min, col_min, row_max, col_max) bounding boxes
        device: torch device string on which to place the output tensor
    Outputs:
        float32 tensor of shape (N, 4) in SAM [x1, y1, x2, y2] order
    """
    if len(boxes) == 0:
        return torch.zeros(0, 4, dtype=torch.float32, device=device)
    sam_boxes = [
        [col_min, row_min, col_max, row_max]
        for row_min, col_min, row_max, col_max in boxes
    ]
    return torch.tensor(sam_boxes, dtype=torch.float32, device=device)


def generate_grid_centroids(
    image_size: int = 1024,
    grid_step: int = 64,
) -> List[Tuple[float, float]]:
    """
    Generate a uniform grid of candidate point prompts for inference-time use.

    At inference time, ground-truth centroids are unavailable, so a regular grid
    of candidate points is used to cover the image.

    Inputs:
        image_size: side length of the square image in pixels
        grid_step: spacing between adjacent grid points in pixels
    Outputs:
        list of (row, col) float tuples covering the image in a uniform grid
    """
    half_step = grid_step // 2
    rows = np.arange(half_step, image_size, grid_step)
    cols = np.arange(half_step, image_size, grid_step)
    grid_rows, grid_cols = np.meshgrid(rows, cols, indexing="ij")
    centroids = [
        (float(r), float(c))
        for r, c in zip(grid_rows.flatten(), grid_cols.flatten())
    ]
    return centroids
