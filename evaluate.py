"""
Evaluation script for MobileSAM-LoRA on the NuInsSeg held-out fold.

At inference time, ground-truth centroids are unavailable, so candidate point
prompts are drawn from a uniform grid. Predicted masks are filtered by IoU score
and de-duplicated with mask-IoU NMS to produce the final instance segmentation.
Results are printed and saved as a per-image CSV.

Usage:
    python evaluate.py --checkpoint checkpoints/fold_1/best_checkpoint.pt
                       --fold_index 0
                       --data_root data/nuinsseg
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets.nuinsseg_dataset import NuInsSegDataset, nuinsseg_collate_fn
from models.mobile_sam_lora import MobileSAMLoRA
from torch.utils.data import DataLoader
from utils.metrics import compute_all_metrics
from utils.prompt_utils import build_point_prompts, generate_grid_centroids


# 
# NMS on binary masks
# 

def mask_nms(
    binary_masks: List[np.ndarray],
    scores: List[float],
    overlap_threshold: float = 0.5,
) -> List[int]:
    """
    Non-maximum suppression on binary masks based on pairwise mask IoU.

    Masks are sorted by score (descending). A mask is suppressed if its IoU
    with any previously kept mask exceeds overlap_threshold.

    Inputs:
        binary_masks: list of boolean numpy arrays (H, W), one per candidate mask
        scores: list of floats (confidence scores), same length as binary_masks
        overlap_threshold: masks with IoU above this are suppressed (default 0.5)
    Outputs:
        list of integer indices (into binary_masks) of the kept masks
    """
    if not binary_masks:
        return []

    order = np.argsort(-np.array(scores))
    keep: List[int] = []
    suppressed = set()

    for current_idx in order:
        if current_idx in suppressed:
            continue
        keep.append(int(current_idx))
        current_mask = binary_masks[current_idx]

        for other_idx in order:
            if other_idx == current_idx or other_idx in suppressed:
                continue
            other_mask = binary_masks[other_idx]
            intersection = int(np.logical_and(current_mask, other_mask).sum())
            if intersection == 0:
                continue
            union = int(np.logical_or(current_mask, other_mask).sum())
            iou = intersection / union if union > 0 else 0.0
            if iou > overlap_threshold:
                suppressed.add(other_idx)

    return keep


# 
# Instance map assembly
# 

def build_instance_map(
    binary_masks: List[np.ndarray],
    scores: List[float],
    image_size: int = 1024,
) -> np.ndarray:
    """
    Assemble an integer instance label map from a set of binary masks.

    Masks are processed in descending score order. Each mask fills only pixels
    that have not yet been assigned, so higher-confidence masks take priority.

    Inputs:
        binary_masks: list of boolean numpy arrays (H, W)
        scores: list of float confidence scores, same length as binary_masks
        image_size: side length of the output square map
    Outputs:
        int32 numpy array (image_size, image_size) with unique positive ID per
        nucleus and 0 for background
    """
    instance_map = np.zeros((image_size, image_size), dtype=np.int32)
    order = np.argsort(-np.array(scores))
    instance_id = 1

    for idx in order:
        mask = binary_masks[idx]
        unassigned = (instance_map == 0) & mask
        if unassigned.sum() >= 10:   # ignore tiny fragments
            instance_map[unassigned] = instance_id
            instance_id += 1

    return instance_map


# 
# Single-image inference
# 

def infer_image(
    model: MobileSAMLoRA,
    image_tensor: torch.Tensor,
    device: str,
    image_size: int = 1024,
    grid_step: int = 64,
    iou_threshold: float = 0.5,
    min_mask_area: int = 50,
    nms_overlap_threshold: float = 0.5,
    inference_batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on one image using a grid of candidate point prompts.

    Steps:
      1. Generate a uniform grid of candidate prompts.
      2. Process prompts in mini-batches through the mask decoder.
      3. Filter by predicted IoU score.
      4. Apply mask-IoU NMS.
      5. Assemble the final instance map.

    Inputs:
        model: MobileSAMLoRA model in eval mode
        image_tensor: float tensor (1, 3, H, W) normalised for SAM
        device: torch device string
        image_size: spatial resolution of the image
        grid_step: spacing between adjacent grid points in pixels
        iou_threshold: minimum predicted IoU to keep a candidate mask
        min_mask_area: minimum area (pixels) for a valid mask
        nms_overlap_threshold: mask-IoU threshold for NMS suppression
        inference_batch_size: number of prompts processed per decoder call
    Outputs:
        pred_instance_map: int32 numpy array (H, W) with unique nucleus IDs
        pred_binary_mask: uint8 numpy array (H, W) binary foreground mask
    """
    grid_centroids = generate_grid_centroids(image_size=image_size, grid_step=grid_step)

    all_binary_masks: List[np.ndarray] = []
    all_scores: List[float] = []

    model.eval()
    with torch.no_grad():
        for batch_start in range(0, len(grid_centroids), inference_batch_size):
            batch_centroids = grid_centroids[batch_start : batch_start + inference_batch_size]
            point_coords, point_labels = build_point_prompts(batch_centroids, device=device)

            low_res_masks, iou_preds = model(image_tensor, point_coords, point_labels)
            full_masks = model.upsample_masks(low_res_masks, (image_size, image_size))

            prob_masks = torch.sigmoid(full_masks).squeeze(1).cpu().numpy()  # (K, H, W)
            scores = iou_preds.squeeze(1).cpu().numpy()                      # (K,)

            for mask_np, score in zip(prob_masks, scores):
                binary = mask_np > 0.5
                if score < iou_threshold:
                    continue
                if binary.sum() < min_mask_area:
                    continue
                all_binary_masks.append(binary)
                all_scores.append(float(score))

    if not all_binary_masks:
        empty = np.zeros((image_size, image_size), dtype=np.int32)
        return empty, empty.astype(np.uint8)

    # NMS
    keep_indices = mask_nms(all_binary_masks, all_scores, overlap_threshold=nms_overlap_threshold)
    kept_masks = [all_binary_masks[i] for i in keep_indices]
    kept_scores = [all_scores[i] for i in keep_indices]

    pred_instance_map = build_instance_map(kept_masks, kept_scores, image_size=image_size)
    pred_binary_mask = (pred_instance_map > 0).astype(np.uint8)

    return pred_instance_map, pred_binary_mask


# 
# Main evaluation
# 

def evaluate(
    checkpoint_path: str,
    fold_index: int,
    data_root: str,
    grid_step: int = 64,
    iou_threshold: float = 0.5,
    output_csv: str = "results.csv",
) -> None:
    """
    Load a trained checkpoint, run inference on the held-out fold, compute and
    print metrics, and save per-image results to CSV.

    Inputs:
        checkpoint_path: path to the .pt checkpoint file
        fold_index: zero-based index of the fold to evaluate
        data_root: root directory of the NuInsSeg dataset
        grid_step: grid spacing for candidate prompts (smaller = more prompts)
        iou_threshold: minimum predicted IoU to keep a candidate mask
        output_csv: path for the per-image results CSV
    Outputs:
        None (prints summary and writes CSV)
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    lora_rank = config.get("lora_rank", 4)
    lora_alpha = config.get("lora_alpha", 1.0)
    image_size = config.get("image_size", 1024)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine SAM checkpoint path (expected alongside the training checkpoint)
    sam_checkpoint_candidates = list(Path(".").glob("mobile_sam.pt"))
    if not sam_checkpoint_candidates:
        raise FileNotFoundError(
            "mobile_sam.pt not found in the current directory. "
            "Download it from the MobileSAM GitHub and place it here."
        )
    sam_checkpoint_path = str(sam_checkpoint_candidates[0])

    # Build model and load weights
    model = MobileSAMLoRA(
        checkpoint_path=sam_checkpoint_path,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from: {checkpoint_path}")

    # Load fold splits
    fold_json_path = Path(data_root) / "folds.json"
    if not fold_json_path.exists():
        raise FileNotFoundError(
            f"Fold file not found: {fold_json_path}. Run train.py first."
        )
    with open(str(fold_json_path)) as fh:
        folds = json.load(fh)

    val_paths = folds[fold_index]["val"]
    print(f"Evaluating fold {fold_index + 1}: {len(val_paths)} images")

    # Dataset
    val_dataset = NuInsSegDataset(val_paths, mode="val", image_size=image_size)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=nuinsseg_collate_fn
    )

    # Evaluate
    all_metrics: Dict[str, List[float]] = {
        "dice": [], "aji": [], "pq": [], "sq": [], "dq": []
    }

    csv_rows: List[Dict] = []

    for batch in tqdm(val_loader, desc="Evaluating"):
        image_tensor = batch["image"][0:1].to(device)   # (1, 3, H, W)
        gt_instances = batch["instances"][0, 0].numpy()  # (H, W)
        image_path_str = batch["image_path"][0]

        pred_instance_map, pred_binary = infer_image(
            model=model,
            image_tensor=image_tensor,
            device=device,
            image_size=image_size,
            grid_step=grid_step,
            iou_threshold=iou_threshold,
        )

        gt_binary = (gt_instances > 0).astype(np.uint8)
        metrics = compute_all_metrics(
            pred_binary, gt_binary, pred_instance_map, gt_instances
        )

        for key, value in metrics.items():
            all_metrics[key].append(value)

        csv_rows.append({
            "image_path": image_path_str,
            "dice": f"{metrics['dice']:.6f}",
            "aji": f"{metrics['aji']:.6f}",
            "pq": f"{metrics['pq']:.6f}",
            "sq": f"{metrics['sq']:.6f}",
            "dq": f"{metrics['dq']:.6f}",
        })

    # Print summary
    print("\n" + "=" * 60)
    print(f"Fold {fold_index + 1} evaluation results")
    print("=" * 60)
    for metric_name in ("dice", "aji", "pq"):
        values = all_metrics[metric_name]
        print(
            f"  {metric_name.upper():5s}: "
            f"{np.mean(values):.4f} +/- {np.std(values):.4f}"
        )

    # Save CSV
    with open(output_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nPer-image results saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate MobileSAM-LoRA on a NuInsSeg fold."
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained checkpoint .pt file")
    parser.add_argument("--fold_index", type=int, default=0,
                        help="Zero-based index of the fold to evaluate")
    parser.add_argument("--data_root", type=str, default="data/nuinsseg",
                        help="Root directory of the NuInsSeg dataset")
    parser.add_argument("--grid_step", type=int, default=64,
                        help="Grid spacing for candidate prompts (pixels)")
    parser.add_argument("--iou_threshold", type=float, default=0.5,
                        help="Minimum predicted IoU to keep a candidate mask")
    parser.add_argument("--output_csv", type=str, default="results.csv",
                        help="Path for per-image results CSV")
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        fold_index=args.fold_index,
        data_root=args.data_root,
        grid_step=args.grid_step,
        iou_threshold=args.iou_threshold,
        output_csv=args.output_csv,
    )
