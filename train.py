"""
Five-fold cross-validation training script for MobileSAM-LoRA nuclei instance segmentation.

Generates reproducible fold splits, trains one model per fold using AdamW with
cosine annealing, logs per-epoch metrics to CSV, and saves best and periodic
checkpoints. Reports mean and standard deviation across folds on completion.

Usage:
    python train.py --checkpoint path/to/mobile_sam.pt --data_root data/nuinsseg
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.nuinsseg_dataset import NuInsSegDataset, nuinsseg_collate_fn
from models.mobile_sam_lora import MobileSAMLoRA
from utils.losses import CombinedLoss
from utils.metrics import compute_all_metrics
from utils.prompt_utils import build_point_prompts, extract_centroids_from_instances

# 
# Configuration
# 

CONFIG: Dict = {
    "lora_rank": 8,            # increased from 4 for more adaptation capacity
    "lora_alpha": 1.0,
    "learning_rate": 1e-4,
    "batch_size": 1,           # number of images per gradient update
    "num_epochs": 120,         # extended: loss still decreasing at epoch 75
    "warmup_epochs": 3,        # linear LR warmup before cosine decay
    "image_size": 1024,
    "num_folds": 5,
    "random_state": 42,
    "checkpoint_dir": "checkpoints",
    "log_dir": "logs",
    "max_prompts_per_image": 6,    # kept at 6 (GPU memory constrained: ~2.5 GB free)
    "iou_filter_threshold": 0.35,  # lowered from 0.5: IoU predictor tends to underestimate
    "iou_loss_weight": 0.5,        # weight for auxiliary IoU prediction loss
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# 
# Reproducibility
# 

def set_all_seeds(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch for reproducibility.

    Inputs:
        seed: integer seed value
    Outputs:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 
# Fold generation
# 

def collect_image_paths(data_root: str) -> List[str]:
    """
    Recursively collect all PNG image paths from the NuInsSeg dataset directory.

    Inputs:
        data_root: path to the root NuInsSeg directory
    Outputs:
        sorted list of absolute path strings to PNG images found under
        'tissue images/' subdirectories
    """
    data_root_path = Path(data_root)
    # Search inside 'tissue images/' subdirectories
    image_paths = sorted(data_root_path.rglob("tissue images/*.png"))
    if not image_paths:
        # Fallback: any PNG file under the data root (excluding mask directories)
        image_paths = [
            p for p in sorted(data_root_path.rglob("*.png"))
            if "mask" not in str(p).lower() and "masks" not in str(p).lower()
        ]
    return [str(p) for p in image_paths]


def generate_and_save_folds(
    image_paths: List[str],
    fold_json_path: str,
    num_folds: int = 5,
    random_state: int = 42,
) -> List[Dict]:
    """
    Build stratified five-fold splits and save them to a JSON file.

    Inputs:
        image_paths: list of image path strings to split
        fold_json_path: path where the fold JSON will be written
        num_folds: number of cross-validation folds
        random_state: random seed for reproducible shuffling
    Outputs:
        list of dicts, each with 'train' and 'val' keys holding path lists
    """
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    folds: List[Dict] = []

    for train_indices, val_indices in kf.split(image_paths):
        folds.append({
            "train": [image_paths[i] for i in train_indices],
            "val": [image_paths[i] for i in val_indices],
        })

    Path(fold_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(fold_json_path, "w") as fh:
        json.dump(folds, fh, indent=2)

    print(f"Saved {num_folds}-fold splits to: {fold_json_path}")
    return folds


# 
# Per-step training helpers
# 

def get_nucleus_target_mask(
    instance_map_np: np.ndarray,
    centroid_row: float,
    centroid_col: float,
) -> Tuple[np.ndarray, int]:
    """
    Retrieve the binary target mask for the nucleus at a given centroid location.

    Inputs:
        instance_map_np: int32 numpy array (H, W) with unique nucleus IDs
        centroid_row: row coordinate of the nucleus centroid
        centroid_col: column coordinate of the nucleus centroid
    Outputs:
        nucleus_mask: float32 numpy array (H, W) with 1 where the nucleus is
        instance_id: integer ID of the nucleus (0 means no nucleus found)
    """
    H, W = instance_map_np.shape
    row_int = int(round(centroid_row))
    col_int = int(round(centroid_col))
    row_int = max(0, min(row_int, H - 1))
    col_int = max(0, min(col_int, W - 1))

    instance_id = int(instance_map_np[row_int, col_int])
    if instance_id == 0:
        return np.zeros((H, W), dtype=np.float32), 0

    nucleus_mask = (instance_map_np == instance_id).astype(np.float32)
    return nucleus_mask, instance_id


# 
# Validation
# 

def run_validation(
    model: MobileSAMLoRA,
    val_loader: DataLoader,
    device: str,
    image_size: int,
    iou_filter_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate the model on a validation DataLoader using all GT centroids as prompts.

    All centroids in each validation image are used (no sampling cap) so that AJI
    and PQ are computed on the full instance set, not a random subset.

    Inputs:
        model: MobileSAMLoRA model
        val_loader: DataLoader yielding validation batches
        device: torch device string
        image_size: spatial resolution of the images
        iou_filter_threshold: predicted IoU score below which a mask is discarded
    Outputs:
        dict with mean 'dice', 'aji', 'pq', 'sq', 'dq' across all validation images
    """
    model.eval()
    metric_accumulator: Dict[str, List[float]] = {
        "dice": [], "aji": [], "pq": [], "sq": [], "dq": []
    }

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            instances_batch = batch["instances"]   # stays on CPU for metric computation
            centroids_batch = batch["centroids"]

            for img_idx in range(len(images)):
                image = images[img_idx : img_idx + 1]  # (1, 3, H, W)
                gt_instance_map = instances_batch[img_idx, 0].numpy()  # (H, W)
                centroids = centroids_batch[img_idx]

                if not centroids:
                    continue

                # Process prompts in chunks to avoid OOM on images with many nuclei
                val_chunk = 16
                all_masks_np: List[np.ndarray] = []
                all_iou_np: List[float] = []
                for chunk_start in range(0, len(centroids), val_chunk):
                    chunk = centroids[chunk_start : chunk_start + val_chunk]
                    pc, pl = build_point_prompts(chunk, device=device)
                    lrm, iou_c = model(image, pc, pl)
                    pm = model.upsample_masks(lrm, (image_size, image_size))
                    all_masks_np.append(
                        (torch.sigmoid(pm).squeeze(1).cpu().numpy() > 0.5)
                    )
                    all_iou_np.append(iou_c.squeeze(1).cpu().numpy())

                pred_binary_np = np.concatenate(all_masks_np, axis=0)
                iou_scores_np = np.concatenate(all_iou_np, axis=0)

                # Build predicted instance map: highest-confidence masks placed first;
                # masks below the IoU threshold are discarded as false positives
                pred_instance_map = np.zeros((image_size, image_size), dtype=np.int32)
                sorted_order = np.argsort(-iou_scores_np)

                for rank_idx in sorted_order:
                    if iou_scores_np[rank_idx] < iou_filter_threshold:
                        continue
                    nucleus_mask = pred_binary_np[rank_idx]
                    free_pixels = (pred_instance_map == 0) & nucleus_mask
                    if free_pixels.sum() >= 10:
                        pred_instance_map[free_pixels] = int(rank_idx) + 1

                gt_binary = (gt_instance_map > 0).astype(np.uint8)
                pred_binary = (pred_instance_map > 0).astype(np.uint8)

                metrics = compute_all_metrics(
                    pred_binary, gt_binary, pred_instance_map, gt_instance_map
                )
                for key, value in metrics.items():
                    metric_accumulator[key].append(value)

    model.train()
    return {key: float(np.mean(vals)) if vals else 0.0
            for key, vals in metric_accumulator.items()}


# 
# Main training loop
# 

def train_one_fold(
    fold_index: int,
    train_paths: List[str],
    val_paths: List[str],
    config: Dict,
    sam_checkpoint: str,
    resume_checkpoint: str = None,
) -> Dict[str, float]:
    """
    Train MobileSAMLoRA for one cross-validation fold.

    Inputs:
        fold_index: zero-based fold number (used for naming checkpoints and logs)
        train_paths: list of image paths for the training split
        val_paths: list of image paths for the validation split
        config: hyperparameter dictionary
        sam_checkpoint: path to the pretrained MobileSAM .pt checkpoint
    Outputs:
        dict of best validation metrics for this fold
    """
    device = config["device"]
    print(f"\n{'='*60}")
    print(f"Fold {fold_index + 1}/{config['num_folds']}")
    print(f"  Train images: {len(train_paths)}  |  Val images: {len(val_paths)}")
    print(f"{'='*60}")

    # Datasets and loaders
    train_dataset = NuInsSegDataset(train_paths, mode="train", image_size=config["image_size"])
    val_dataset = NuInsSegDataset(val_paths, mode="val", image_size=config["image_size"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
        collate_fn=nuinsseg_collate_fn,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=nuinsseg_collate_fn,
    )

    # Model
    model = MobileSAMLoRA(
        checkpoint_path=sam_checkpoint,
        lora_rank=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
    ).to(device)

    # Optimizer and scheduler (linear warmup then cosine decay)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"])
    warmup_epochs = config.get("warmup_epochs", 3)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["num_epochs"] - warmup_epochs, eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    criterion = CombinedLoss()

    # Checkpoint and log directories
    checkpoint_dir = Path(config["checkpoint_dir"]) / f"fold_{fold_index + 1}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv_path = log_dir / f"fold_{fold_index + 1}_metrics.csv"

    csv_fields = ["epoch", "train_loss", "val_dice", "val_aji", "val_pq", "val_sq", "val_dq"]

    # Resume from checkpoint if requested
    start_epoch = 1
    best_val_aji = -1.0
    best_metrics: Dict[str, float] = {}

    if resume_checkpoint is not None:
        print(f"  Resuming from: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        if "val_metrics" in ckpt:
            best_val_aji = ckpt["val_metrics"].get("aji", -1.0)
            best_metrics = dict(ckpt["val_metrics"])
        # Fast-forward scheduler to the correct state
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"  Resuming from epoch {start_epoch}, best AJI so far: {best_val_aji:.4f}")

    # Open CSV: append if resuming, write fresh otherwise
    csv_file = open(str(log_csv_path), "a" if resume_checkpoint is not None else "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if resume_checkpoint is None:
        csv_writer.writeheader()

    max_prompts = config["max_prompts_per_image"]

    for epoch in range(start_epoch, config["num_epochs"] + 1):
        model.train()
        epoch_loss_total = 0.0
        epoch_loss_count = 0

        pbar = tqdm(train_loader, desc=f"Fold {fold_index+1} Epoch {epoch}/{config['num_epochs']}")

        for batch in pbar:
            images = batch["image"].to(device)            # (B, 3, H, W)
            instances_batch = batch["instances"]          # (B, 1, H, W) on CPU
            centroids_batch = batch["centroids"]          # list of B lists

            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=False)
            num_valid_steps = 0

            for img_idx in range(len(images)):
                image = images[img_idx : img_idx + 1]     # (1, 3, H, W)
                instance_map_np = instances_batch[img_idx, 0].numpy()
                centroids = centroids_batch[img_idx]

                if not centroids:
                    continue

                # Sample a subset of nuclei for this step
                sampled_centroids = (
                    random.sample(centroids, max_prompts)
                    if len(centroids) > max_prompts
                    else centroids
                )

                # Build target masks and valid prompt list
                valid_centroids: List[Tuple[float, float]] = []
                target_mask_list: List[np.ndarray] = []

                for centroid_row, centroid_col in sampled_centroids:
                    nucleus_mask, instance_id = get_nucleus_target_mask(
                        instance_map_np, centroid_row, centroid_col
                    )
                    if instance_id == 0:
                        continue
                    valid_centroids.append((centroid_row, centroid_col))
                    target_mask_list.append(nucleus_mask)

                if not valid_centroids:
                    continue

                # Sample one background (negative) point per nucleus.
                # Picking a point from the background region improves boundary
                # delineation and is standard practice when fine-tuning SAM.
                bg_pixels = np.argwhere(instance_map_np == 0)
                neg_centroids: List[Tuple[float, float]] = []
                for _ in valid_centroids:
                    if len(bg_pixels) > 0:
                        idx = random.randrange(len(bg_pixels))
                        neg_centroids.append(
                            (float(bg_pixels[idx, 0]), float(bg_pixels[idx, 1]))
                        )
                    else:
                        neg_centroids.append((0.0, 0.0))

                point_coords, point_labels = build_point_prompts(
                    valid_centroids, neg_centroids=neg_centroids, device=device
                )

                # Stack targets: (K, 1, H, W)
                target_masks = torch.from_numpy(
                    np.stack(target_mask_list, axis=0)[:, np.newaxis, :, :]
                ).to(device)

                # Forward pass with AMP
                with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    low_res_masks, iou_preds = model(image, point_coords, point_labels)

                    # Upsample predictions to full resolution for loss
                    pred_masks = model.upsample_masks(
                        low_res_masks, (config["image_size"], config["image_size"])
                    )

                    seg_loss = criterion(pred_masks, target_masks)

                    # Auxiliary IoU prediction loss: MSE(predicted_iou, actual_iou)
                    with torch.no_grad():
                        pred_binary = (torch.sigmoid(pred_masks.detach()) > 0.5).float()
                        inter = (pred_binary * target_masks).sum(dim=(-2, -1))
                        union = ((pred_binary + target_masks) >= 1).float().sum(dim=(-2, -1))
                        actual_iou = (inter / (union + 1e-6)).squeeze(1)  # (N,)
                    iou_loss = F.mse_loss(iou_preds[:, 0], actual_iou)

                    step_loss = seg_loss + config.get("iou_loss_weight", 0.5) * iou_loss

                batch_loss = batch_loss + step_loss
                num_valid_steps += 1

            if num_valid_steps > 0:
                avg_batch_loss = batch_loss / num_valid_steps
                scaler.scale(avg_batch_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss_total += avg_batch_loss.item()
                epoch_loss_count += 1
                pbar.set_postfix(loss=f"{avg_batch_loss.item():.4f}")

        scheduler.step()
        avg_train_loss = epoch_loss_total / max(epoch_loss_count, 1)

        # Validate
        val_metrics = run_validation(
            model, val_loader, device, config["image_size"],
            iou_filter_threshold=config["iou_filter_threshold"],
        )

        print(
            f"  Epoch {epoch:3d} | Loss: {avg_train_loss:.4f} | "
            f"Dice: {val_metrics['dice']:.4f} | AJI: {val_metrics['aji']:.4f} | "
            f"PQ: {val_metrics['pq']:.4f}"
        )

        # Log to CSV
        csv_writer.writerow({
            "epoch": epoch,
            "train_loss": f"{avg_train_loss:.6f}",
            "val_dice": f"{val_metrics['dice']:.6f}",
            "val_aji": f"{val_metrics['aji']:.6f}",
            "val_pq": f"{val_metrics['pq']:.6f}",
            "val_sq": f"{val_metrics['sq']:.6f}",
            "val_dq": f"{val_metrics['dq']:.6f}",
        })
        csv_file.flush()

        # Save best checkpoint
        if val_metrics["aji"] > best_val_aji:
            best_val_aji = val_metrics["aji"]
            best_metrics = dict(val_metrics)
            best_ckpt_path = checkpoint_dir / "best_checkpoint.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                },
                str(best_ckpt_path),
            )

        # Periodic intermediate checkpoint every 10 epochs
        if epoch % 10 == 0:
            periodic_ckpt_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_metrics,
                    "config": config,
                },
                str(periodic_ckpt_path),
            )

    csv_file.close()
    print(f"Fold {fold_index + 1} best AJI: {best_val_aji:.4f}")
    return best_metrics


def main() -> None:
    """
    Entry point: parse arguments, set seeds, generate folds, and run training.
    """
    parser = argparse.ArgumentParser(description="Train MobileSAM-LoRA on NuInsSeg.")
    parser.add_argument("--checkpoint", type=str, default="mobile_sam.pt",
                        help="Path to pretrained MobileSAM checkpoint")
    parser.add_argument("--data_root", type=str, default="data/nuinsseg",
                        help="Root directory of the NuInsSeg dataset")
    parser.add_argument("--fold", type=int, default=None,
                        help="Train only this specific fold (0-indexed). "
                             "Omit to train all folds.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint (.pt) to resume training from.")
    args = parser.parse_args()

    set_all_seeds(CONFIG["random_state"])

    # Collect image paths and build folds
    image_paths = collect_image_paths(args.data_root)
    if not image_paths:
        print(f"No images found under: {args.data_root}")
        print("Run preprocessing/generate_masks.py first.")
        sys.exit(1)

    print(f"Total images found: {len(image_paths)}")

    fold_json_path = Path(args.data_root) / "folds.json"
    if fold_json_path.exists():
        with open(str(fold_json_path)) as fh:
            folds = json.load(fh)
        print(f"Loaded existing folds from: {fold_json_path}")
    else:
        folds = generate_and_save_folds(
            image_paths,
            str(fold_json_path),
            num_folds=CONFIG["num_folds"],
            random_state=CONFIG["random_state"],
        )

    # Train folds
    fold_indices = [args.fold] if args.fold is not None else list(range(len(folds)))
    all_fold_metrics: List[Dict[str, float]] = []

    for fold_idx in fold_indices:
        fold_data = folds[fold_idx]
        best_metrics = train_one_fold(
            fold_index=fold_idx,
            train_paths=fold_data["train"],
            val_paths=fold_data["val"],
            config=CONFIG,
            sam_checkpoint=args.checkpoint,
            resume_checkpoint=args.resume,
        )
        all_fold_metrics.append(best_metrics)

    # Summarise results
    if len(all_fold_metrics) > 1:
        print("\n" + "=" * 60)
        print("Cross-validation summary")
        print("=" * 60)
        for metric_name in ("dice", "aji", "pq"):
            values = [m[metric_name] for m in all_fold_metrics]
            print(
                f"  {metric_name.upper():5s}: "
                f"{np.mean(values):.4f} +/- {np.std(values):.4f}"
            )


if __name__ == "__main__":
    main()
