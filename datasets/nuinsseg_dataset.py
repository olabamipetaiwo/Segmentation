"""
PyTorch Dataset class for the NuInsSeg nuclei instance segmentation dataset.

Each sample returns a preprocessed image tensor, binary mask, instance label map,
and centroid coordinates extracted from the (possibly augmented) instance map.

Images are resized to 1024x1024 and normalised with SAM's pixel mean and std
so they are ready for direct input to the MobileSAM image encoder.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.prompt_utils import extract_centroids_from_instances

# SAM normalisation constants
SAM_PIXEL_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
SAM_PIXEL_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


# 
# Augmentation pipelines
# 

def get_train_transform(image_size: int = 1024) -> A.Compose:
    """
    Build the albumentations augmentation pipeline for training.

    Applies spatial augmentations jointly to the image, binary mask, and
    instance map. ColorJitter is applied to the image only.

    Inputs:
        image_size: target side length for resizing (both H and W)
    Outputs:
        albumentations Compose object with 'instance_map' as an additional target
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
            # Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, interpolation=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_REFLECT_101, p=0.5),
            # Intensity / stain variation (H&E-specific)
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1, p=0.5),
            # Microscope artefacts
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ],
        additional_targets={"instance_map": "mask"},
    )


def get_val_transform(image_size: int = 1024) -> A.Compose:
    """
    Build the albumentations transform for validation (resize only, no augmentation).

    Inputs:
        image_size: target side length for resizing (both H and W)
    Outputs:
        albumentations Compose object with 'instance_map' as an additional target
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size, interpolation=cv2.INTER_LINEAR),
        ],
        additional_targets={"instance_map": "mask"},
    )


# 
# Dataset
# 

class NuInsSegDataset(Dataset):
    """
    PyTorch Dataset for NuInsSeg nuclei instance segmentation.

    For each image, loads the RGB image, binary mask, and instance label map,
    applies augmentations, normalises with SAM constants, and extracts centroids
    from the transformed instance map.

    Returns a dict with keys:
        image      - float32 tensor (3, H, W) SAM-normalised
        mask       - float32 tensor (1, H, W) binary foreground mask in [0, 1]
        instances  - int32 tensor (1, H, W) instance label map
        centroids  - list of (row, col) float tuples
        image_path - str path to the source image for bookkeeping
    """

    def __init__(
        self,
        image_paths: List[str],
        mode: str = "train",
        image_size: int = 1024,
    ) -> None:
        """
        Inputs:
            image_paths: list of absolute paths to source PNG images; corresponding
                         mask/instance/distance arrays must exist alongside them
            mode: 'train' applies full augmentations; 'val' applies resize only
            image_size: target spatial resolution for the model input
        """
        if mode not in ("train", "val"):
            raise ValueError(f"mode must be 'train' or 'val', got: {mode!r}")

        self.image_paths = image_paths
        self.mode = mode
        self.image_size = image_size

        self.transform = (
            get_train_transform(image_size) if mode == "train"
            else get_val_transform(image_size)
        )

    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self.image_paths)

    def _get_mask_path(self, image_path: Path, subdir: str, ext: str = ".npy") -> Path:
        """
        Resolve the path to a derived array (mask, instance map, distance map).

        Inputs:
            image_path: path to the source image
            subdir: subdirectory name ('masks', 'instances', 'distances')
            ext: file extension of the derived array file
        Outputs:
            Path to the derived array file
        """
        tissue_dir = image_path.parent.parent
        return tissue_dir / subdir / (image_path.stem + ext)

    def __getitem__(self, index: int) -> Dict:
        """
        Load, transform, and return one sample.

        Inputs:
            index: integer index into image_paths
        Outputs:
            dict with keys: image, mask, instances, centroids, image_path
        """
        image_path = Path(self.image_paths[index])

        # Load source image (BGR -> RGB)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # (H, W, 3) uint8

        # Load binary mask
        mask_path = self._get_mask_path(image_path, "masks")
        if mask_path.exists():
            binary_mask = np.load(str(mask_path))
        else:
            # Fall back: derive from instance map
            binary_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        # Load instance label map
        instance_path = self._get_mask_path(image_path, "instances")
        if instance_path.exists():
            instance_map = np.load(str(instance_path)).astype(np.int32)
        else:
            instance_map = np.zeros(image_rgb.shape[:2], dtype=np.int32)

        # Ensure binary mask is uint8 in {0, 1} for albumentations
        binary_mask = (binary_mask > 0).astype(np.uint8)

        # Apply spatial augmentations jointly to image, mask, and instance map
        transformed = self.transform(
            image=image_rgb,
            mask=binary_mask,
            instance_map=instance_map.astype(np.int32),
        )

        aug_image = transformed["image"]              # (H, W, 3) uint8
        aug_binary = transformed["mask"]              # (H, W) uint8
        aug_instances = transformed["instance_map"]   # (H, W) int32

        # Normalise image with SAM pixel statistics
        image_float = aug_image.astype(np.float32)
        image_norm = (image_float - SAM_PIXEL_MEAN) / SAM_PIXEL_STD  # (H, W, 3)

        # Extract centroids from the augmented instance map
        centroids = extract_centroids_from_instances(aug_instances)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_norm).permute(2, 0, 1)  # (3, H, W)
        mask_tensor = torch.from_numpy(aug_binary.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        instance_tensor = torch.from_numpy(aug_instances).unsqueeze(0)  # (1, H, W)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "instances": instance_tensor,
            "centroids": centroids,
            "image_path": str(image_path),
        }


# 
# Collate function for DataLoader
# 

def nuinsseg_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function that handles variable-length centroid lists.

    Inputs:
        batch: list of sample dicts returned by NuInsSegDataset.__getitem__
    Outputs:
        dict with stacked tensors for 'image', 'mask', 'instances', and a list
        of centroid lists under 'centroids', and list of paths under 'image_path'
    """
    images = torch.stack([item["image"] for item in batch])
    masks = torch.stack([item["mask"] for item in batch])
    instances = torch.stack([item["instances"] for item in batch])
    centroids = [item["centroids"] for item in batch]
    image_paths = [item["image_path"] for item in batch]

    return {
        "image": images,
        "mask": masks,
        "instances": instances,
        "centroids": centroids,
        "image_path": image_paths,
    }
