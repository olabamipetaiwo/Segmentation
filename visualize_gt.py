"""
Qualitative visualisation using GT bounding-box prompts (matches training validation).

Produces side-by-side three-panel figures (H&E image | GT instance map | predicted
instance map) ranked by AJI. Saves strong and weak examples per fold.

Usage:
    python visualize_gt.py --fold 1 --data_root data/nuinsseg --num_examples 4
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.nuinsseg_dataset import NuInsSegDataset, nuinsseg_collate_fn
from models.mobile_sam_lora import MobileSAMLoRA
from utils.metrics import aggregated_jaccard_index
from utils.prompt_utils import build_box_prompts, extract_boxes_from_instances
from visualize import colorise_instance_map, denormalise_image, save_comparison_figure

MIN_MASK_PIXELS = 10


def infer_with_gt_boxes(
    model: MobileSAMLoRA,
    image_tensor: torch.Tensor,
    gt_instance_map: np.ndarray,
    device: str,
    image_size: int,
    iou_threshold: float = 0.35,
    chunk: int = 16,
) -> np.ndarray:
    boxes, _ = extract_boxes_from_instances(gt_instance_map)
    if not boxes:
        return np.zeros((image_size, image_size), dtype=np.int32)

    all_masks, all_ious = [], []
    with torch.no_grad():
        for s in range(0, len(boxes), chunk):
            box_t = build_box_prompts(boxes[s : s + chunk], device=device)
            lrm, iou_c = model(image_tensor, boxes=box_t)
            pm = model.upsample_masks(lrm, (image_size, image_size))
            all_masks.append((torch.sigmoid(pm).squeeze(1).cpu().numpy() > 0.5))
            all_ious.append(iou_c.squeeze(1).cpu().numpy())

    masks_np = np.concatenate(all_masks, axis=0)
    ious_np  = np.concatenate(all_ious,  axis=0)

    pred_map = np.zeros((image_size, image_size), dtype=np.int32)
    for rank_idx in np.argsort(-ious_np):
        if ious_np[rank_idx] < iou_threshold:
            continue
        free = (pred_map == 0) & masks_np[rank_idx]
        if free.sum() >= MIN_MASK_PIXELS:
            pred_map[free] = int(rank_idx) + 1
    return pred_map


def run(fold: int, data_root: str, num_examples: int, seed: int) -> None:
    random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_size = 1024
    fold_index = fold - 1

    ckpt_path = f"checkpoints/fold_{fold}/best_checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg  = ckpt.get("config", {})
    model = MobileSAMLoRA(
        checkpoint_path="mobile_sam.pt",
        lora_rank=cfg.get("lora_rank", 8),
        lora_alpha=cfg.get("lora_alpha", 1.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded fold {fold} checkpoint (epoch {ckpt.get('epoch', '?')})")

    with open(Path(data_root) / "folds.json") as f:
        folds = json.load(f)
    val_paths = folds[fold_index]["val"]

    # Sample a pool larger than needed for diversity in strong/weak selection
    pool = random.sample(val_paths, min(len(val_paths), max(num_examples * 4, 30)))
    dataset = NuInsSegDataset(pool, mode="val", image_size=image_size)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False,
                         num_workers=2, collate_fn=nuinsseg_collate_fn)

    results: List[Dict] = []
    print("Running GT-box inference...")
    for batch in tqdm(loader):
        img_t    = batch["image"][0:1].to(device)
        gt_map   = batch["instances"][0, 0].numpy()
        img_path = batch["image_path"][0]

        pred_map = infer_with_gt_boxes(model, img_t, gt_map, device, image_size)
        aji = aggregated_jaccard_index(pred_map, gt_map)

        results.append({
            "image_rgb": denormalise_image(batch["image"][0]),  # uint8, not tensor
            "gt_map":    gt_map,
            "pred_map":  pred_map,
            "aji":       aji,
            "name":      Path(img_path).stem,
        })

    results.sort(key=lambda x: x["aji"], reverse=True)
    n_strong = num_examples // 2
    n_weak   = num_examples - n_strong
    selected = results[:n_strong] + results[-n_weak:]

    out_dir = Path("figures") / f"fold_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    for rank, r in enumerate(selected):
        tag   = "strong" if rank < n_strong else "weak"
        fname = f"{tag}_{rank + 1:02d}_{r['name']}_aji{r['aji']:.3f}.png"
        save_comparison_figure(
            image_rgb=r["image_rgb"],
            gt_instance_map=r["gt_map"],
            pred_instance_map=r["pred_map"],
            aji_score=r["aji"],
            save_path=str(out_dir / fname),
            image_name=r["name"],
        )
        print(f"  Saved [{tag}] AJI={r['aji']:.3f}: {fname}")

    print(f"Done — {len(selected)} figures saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold",         type=int, default=1)
    parser.add_argument("--data_root",    type=str, default="data/nuinsseg")
    parser.add_argument("--num_examples", type=int, default=4)
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    run(fold=args.fold, data_root=args.data_root,
        num_examples=args.num_examples, seed=args.seed)
