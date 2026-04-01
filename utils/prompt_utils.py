"""
Utility functions for constructing SAM point prompts from instance label maps.
"""

from typing import List, Tuple

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
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build point prompt tensors in the format expected by MobileSAM's prompt encoder.

    SAM uses (x, y) = (col, row) coordinate ordering.

    Inputs:
        centroids: list of (row, col) float tuples representing nucleus centres
        device: torch device string on which to place the output tensors
    Outputs:
        point_coords: float32 tensor of shape (N, 1, 2) where N = len(centroids),
                      with coordinates in (x, y) = (col, row) order
        point_labels: int32 tensor of shape (N, 1) with all values 1 (foreground)
    """
    if len(centroids) == 0:
        point_coords = torch.zeros(0, 1, 2, dtype=torch.float32, device=device)
        point_labels = torch.zeros(0, 1, dtype=torch.int32, device=device)
        return point_coords, point_labels

    # Convert (row, col) to SAM's expected (x, y) = (col, row)
    coords = [[col, row] for row, col in centroids]
    point_coords = (
        torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(1)
    )  # (N, 1, 2)
    point_labels = torch.ones(
        len(centroids), 1, dtype=torch.int32, device=device
    )  # (N, 1)
    return point_coords, point_labels


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
