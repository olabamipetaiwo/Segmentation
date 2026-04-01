"""
Qualitative visualisation of MobileSAM-LoRA predictions vs ground truth.

Produces side-by-side three-panel figures (original image, GT instance mask,
predicted instance mask) for a set of validation images. Selects examples with
strong and weaker AJI to show a range of performance. Figures are saved to
the figures/ directory.

Usage:
    python visualize.py --checkpoint checkpoints/fold_1/best_checkpoint.pt
                        --fold_index 0
                        --data_root data/nuinsseg
                        --num_examples 6
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.nuinsseg_dataset import NuInsSegDataset, nuinsseg_collate_fn
from evaluate import infer_image
from models.mobile_sam_lora import MobileSAMLoRA
from utils.metrics import aggregated_jaccard_index


# ---------------------------------------------------------------------------
# Colour-coded instance mask rendering
# ---------------------------------------------------------------------------

def colorise_instance_map(instance_map: np.ndarray) -> np.ndarray:
    """
    Render an integer instance label map as an RGB image with distinct colours.

    Background (label 0) is rendered as black. Each unique positive label
    receives a distinct colour drawn from the HSV colour space.

    Inputs:
        instance_map: integer numpy array (H, W) with unique positive ID per instance
    Outputs:
        float32 numpy array (H, W, 3) with RGB values in [0, 1]
    """
    H, W = instance_map.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    instance_ids = np.unique(instance_map)
    instance_ids = instance_ids[instance_ids > 0]

    if len(instance_ids) == 0:
        return rgb

    # Spread hues evenly across the HSV colour wheel
    for rank, inst_id in enumerate(instance_ids):
        hue = (rank / len(instance_ids)) % 1.0
        rgb_colour = mcolors.hsv_to_rgb([hue, 0.7, 0.9])
        pixels = instance_map == inst_id
        rgb[pixels] = rgb_colour

    return rgb


def denormalise_image(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Reverse SAM normalisation to recover a displayable uint8 RGB image.

    Inputs:
        image_tensor: float32 tensor (3, H, W) normalised with SAM pixel stats
    Outputs:
        uint8 numpy array (H, W, 3) with values clipped to [0, 255]
    """
    pixel_mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    pixel_std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    img_np = image_tensor.cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
    img_np = img_np * pixel_std + pixel_mean
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return img_np


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------

def save_comparison_figure(
    image_rgb: np.ndarray,
    gt_instance_map: np.ndarray,
    pred_instance_map: np.ndarray,
    aji_score: float,
    save_path: str,
    image_name: str,
) -> None:
    """
    Save a three-panel side-by-side comparison figure.

    Panels: original H&E image | GT instance mask | predicted instance mask.

    Inputs:
        image_rgb: uint8 numpy array (H, W, 3), the original RGB image
        gt_instance_map: integer numpy array (H, W), ground truth instance labels
        pred_instance_map: integer numpy array (H, W), predicted instance labels
        aji_score: AJI score for this image (shown in title)
        save_path: output file path (.png)
        image_name: short name for the figure title
    Outputs:
        None (writes PNG to disk)
    """
    gt_colour = colorise_instance_map(gt_instance_map)
    pred_colour = colorise_instance_map(pred_instance_map)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title("H&E Image", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(gt_colour)
    num_gt = int((np.unique(gt_instance_map) > 0).sum())
    axes[1].set_title(f"Ground Truth ({num_gt} nuclei)", fontsize=14)
    axes[1].axis("off")

    axes[2].imshow(pred_colour)
    num_pred = int((np.unique(pred_instance_map) > 0).sum())
    axes[2].set_title(f"Prediction ({num_pred} nuclei)", fontsize=14)
    axes[2].axis("off")

    fig.suptitle(f"{image_name}   AJI = {aji_score:.3f}", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main visualisation loop
# ---------------------------------------------------------------------------

def visualise(
    checkpoint_path: str,
    fold_index: int,
    data_root: str,
    figures_dir: str = "figures",
    num_examples: int = 6,
    grid_step: int = 64,
    iou_threshold: float = 0.5,
    random_seed: int = 42,
) -> None:
    """
    Run inference on a sample of validation images and save comparison figures.

    Selects the top half of examples by AJI (strong predictions) and the bottom
    half (weaker predictions) to illustrate a range of model performance.

    Inputs:
        checkpoint_path: path to the trained .pt checkpoint
        fold_index: zero-based index of the fold to visualise
        data_root: root directory of the NuInsSeg dataset
        figures_dir: directory where PNG figures are saved
        num_examples: total number of example figures to generate
        grid_step: grid spacing for inference-time candidate prompts
        iou_threshold: minimum predicted IoU to keep a candidate mask
        random_seed: seed for reproducible example sampling
    Outputs:
        None (writes PNG files to figures_dir)
    """
    random.seed(random_seed)
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    lora_rank = config.get("lora_rank", 4)
    lora_alpha = config.get("lora_alpha", 1.0)
    image_size = config.get("image_size", 1024)

    sam_checkpoint_candidates = list(Path(".").glob("mobile_sam.pt"))
    if not sam_checkpoint_candidates:
        raise FileNotFoundError(
            "mobile_sam.pt not found. Place it in the project root."
        )
    sam_checkpoint_path = str(sam_checkpoint_candidates[0])

    model = MobileSAMLoRA(
        checkpoint_path=sam_checkpoint_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load fold
    fold_json_path = Path(data_root) / "folds.json"
    with open(str(fold_json_path)) as fh:
        folds = json.load(fh)

    val_paths = folds[fold_index]["val"]
    # Sample a subset for visualisation
    sampled_paths = random.sample(val_paths, min(len(val_paths), max(num_examples * 3, 20)))

    val_dataset = NuInsSegDataset(sampled_paths, mode="val", image_size=image_size)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=nuinsseg_collate_fn
    )

    # Collect per-image results
    results: List[Dict] = []
    print("Running inference for visualisation...")

    for batch in tqdm(val_loader):
        image_tensor = batch["image"][0:1].to(device)
        gt_instances = batch["instances"][0, 0].numpy()
        image_path_str = batch["image_path"][0]

        pred_instance_map, _ = infer_image(
            model=model,
            image_tensor=image_tensor,
            device=device,
            image_size=image_size,
            grid_step=grid_step,
            iou_threshold=iou_threshold,
        )

        aji = aggregated_jaccard_index(pred_instance_map, gt_instances)

        results.append({
            "image_tensor": batch["image"][0],  # keep on CPU
            "gt_instances": gt_instances,
            "pred_instance_map": pred_instance_map,
            "aji": aji,
            "image_path": image_path_str,
        })

    if not results:
        print("No results to visualise.")
        return

    # Sort by AJI and select strong and weak examples
    results_sorted = sorted(results, key=lambda x: x["aji"], reverse=True)
    num_strong = num_examples // 2
    num_weak = num_examples - num_strong

    strong_examples = results_sorted[:num_strong]
    weak_examples = results_sorted[-num_weak:] if num_weak > 0 else []
    selected = strong_examples + weak_examples

    print(f"Saving {len(selected)} comparison figures to: {figures_dir}/")

    for rank, result in enumerate(selected):
        category = "strong" if rank < num_strong else "weak"
        image_name = Path(result["image_path"]).stem
        save_name = f"{category}_{rank + 1:02d}_{image_name}_aji{result['aji']:.3f}.png"
        save_path = str(figures_path / save_name)

        image_rgb = denormalise_image(result["image_tensor"])
        save_comparison_figure(
            image_rgb=image_rgb,
            gt_instance_map=result["gt_instances"],
            pred_instance_map=result["pred_instance_map"],
            aji_score=result["aji"],
            save_path=save_path,
            image_name=image_name,
        )
        print(f"  Saved [{category}] AJI={result['aji']:.3f}: {save_name}")

    print("Visualisation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate qualitative comparison figures for MobileSAM-LoRA."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint .pt file")
    parser.add_argument("--fold_index", type=int, default=0,
                        help="Zero-based fold index to visualise")
    parser.add_argument("--data_root", type=str, default="data/nuinsseg",
                        help="Root directory of the NuInsSeg dataset")
    parser.add_argument("--figures_dir", type=str, default="figures",
                        help="Directory to save output PNG figures")
    parser.add_argument("--num_examples", type=int, default=6,
                        help="Number of comparison figures to generate")
    parser.add_argument("--grid_step", type=int, default=64,
                        help="Grid spacing for inference-time candidate prompts")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Minimum predicted IoU to keep a candidate mask")
    args = parser.parse_args()

    visualise(
        checkpoint_path=args.checkpoint,
        fold_index=args.fold_index,
        data_root=args.data_root,
        figures_dir=args.figures_dir,
        num_examples=args.num_examples,
        grid_step=args.grid_step,
        iou_threshold=args.iou_threshold,
    )
