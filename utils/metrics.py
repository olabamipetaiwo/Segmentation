"""
Segmentation metrics for nuclei instance segmentation evaluation.
Implements Dice, AJI, and PQ following the NuInsSeg evaluation protocol.
Adapted from https://github.com/masih4/NuInsSeg
"""

from typing import Dict

import numpy as np


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute pixel-level Dice coefficient between a predicted and ground truth binary mask.

    Inputs:
        pred_mask: numpy array of any shape, treated as binary (truthy = foreground)
        gt_mask: numpy array of same shape, treated as binary (truthy = foreground)
    Outputs:
        float Dice coefficient in [0, 1]; returns 1.0 when both masks are empty
    """
    pred_bool = pred_mask.astype(bool).flatten()
    gt_bool = gt_mask.astype(bool).flatten()

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    total = pred_bool.sum() + gt_bool.sum()

    if total == 0:
        return 1.0

    return float(2.0 * intersection / total)


def aggregated_jaccard_index(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
) -> float:
    """
    Compute the Aggregated Jaccard Index (AJI) for nuclei instance segmentation.

    For each ground-truth nucleus, finds the predicted nucleus with maximum
    intersection, accumulates intersection and union, and penalises unmatched
    predicted nuclei by adding their area to the denominator.
    Adapted from https://github.com/masih4/NuInsSeg

    Inputs:
        pred_instances: integer numpy array (H, W), unique positive ID per predicted
                        nucleus, 0 = background
        gt_instances: integer numpy array (H, W), unique positive ID per ground-truth
                      nucleus, 0 = background
    Outputs:
        float AJI score in [0, 1]; returns 1.0 when both maps are empty, 0.0 when
        one is empty and the other is not
    """
    gt_ids = np.unique(gt_instances)
    gt_ids = gt_ids[gt_ids > 0]

    pred_ids = np.unique(pred_instances)
    pred_ids = pred_ids[pred_ids > 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return 1.0
    if len(gt_ids) == 0 or len(pred_ids) == 0:
        return 0.0

    matched_pred_ids: set = set()
    total_intersection = 0
    total_union = 0

    for gt_id in gt_ids:
        gt_mask = gt_instances == gt_id
        gt_area = int(gt_mask.sum())

        # Find predicted instances that overlap with this GT nucleus
        overlapping_pred_ids = np.unique(pred_instances[gt_mask])
        overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]

        best_intersection = 0
        best_pred_id = -1

        for pred_id in overlapping_pred_ids:
            pred_mask = pred_instances == pred_id
            current_intersection = int(np.logical_and(gt_mask, pred_mask).sum())
            if current_intersection > best_intersection:
                best_intersection = current_intersection
                best_pred_id = pred_id

        if best_pred_id >= 0:
            best_pred_mask = pred_instances == best_pred_id
            union = int(np.logical_or(gt_mask, best_pred_mask).sum())
            total_intersection += best_intersection
            total_union += union
            matched_pred_ids.add(best_pred_id)
        else:
            # No predicted nucleus overlaps this GT nucleus; add GT area to union
            total_union += gt_area

    # Penalise unmatched predicted nuclei
    for pred_id in pred_ids:
        if pred_id not in matched_pred_ids:
            pred_area = int((pred_instances == pred_id).sum())
            total_union += pred_area

    if total_union == 0:
        return 0.0

    return float(total_intersection / total_union)


def panoptic_quality(
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Panoptic Quality (PQ) for nuclei instance segmentation.

    A ground-truth and predicted nucleus are matched when their IoU exceeds
    iou_threshold. SQ is the mean IoU of matched pairs; DQ is the detection
    F1 score; PQ = SQ * DQ.

    Inputs:
        pred_instances: integer numpy array (H, W), unique positive ID per predicted
                        nucleus, 0 = background
        gt_instances: integer numpy array (H, W), unique positive ID per ground-truth
                      nucleus, 0 = background
        iou_threshold: minimum IoU required for a true positive match (default 0.5)
    Outputs:
        dict with keys 'pq', 'sq', 'dq', all floats in [0, 1]
    """
    gt_ids = np.unique(gt_instances)
    gt_ids = gt_ids[gt_ids > 0]

    pred_ids = np.unique(pred_instances)
    pred_ids = pred_ids[pred_ids > 0]

    if len(gt_ids) == 0 and len(pred_ids) == 0:
        return {"pq": 1.0, "sq": 1.0, "dq": 1.0}
    if len(gt_ids) == 0:
        return {"pq": 0.0, "sq": 0.0, "dq": 0.0}
    if len(pred_ids) == 0:
        return {"pq": 0.0, "sq": 0.0, "dq": 0.0}

    matched_iou_scores = []
    matched_gt_ids: set = set()
    matched_pred_ids: set = set()

    for gt_id in gt_ids:
        gt_mask = gt_instances == gt_id

        overlapping_pred_ids = np.unique(pred_instances[gt_mask])
        overlapping_pred_ids = overlapping_pred_ids[overlapping_pred_ids > 0]

        best_iou = iou_threshold  # Must exceed threshold to count as TP
        best_pred_id = -1

        for pred_id in overlapping_pred_ids:
            if pred_id in matched_pred_ids:
                continue
            pred_mask = pred_instances == pred_id
            intersection = int(np.logical_and(gt_mask, pred_mask).sum())
            union = int(np.logical_or(gt_mask, pred_mask).sum())
            iou = intersection / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_pred_id = pred_id

        if best_pred_id >= 0:
            matched_iou_scores.append(best_iou)
            matched_gt_ids.add(gt_id)
            matched_pred_ids.add(best_pred_id)

    tp = len(matched_iou_scores)
    fp = len(pred_ids) - len(matched_pred_ids)
    fn = len(gt_ids) - len(matched_gt_ids)

    sq = float(np.mean(matched_iou_scores)) if tp > 0 else 0.0
    denominator = tp + 0.5 * fp + 0.5 * fn
    dq = float(tp / denominator) if denominator > 0 else 0.0
    pq = sq * dq

    return {"pq": pq, "sq": sq, "dq": dq}


def compute_all_metrics(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
    pred_instances: np.ndarray,
    gt_instances: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all segmentation metrics: Dice, AJI, PQ, SQ, DQ.

    Inputs:
        pred_binary: numpy array (H, W), binary predicted foreground mask
        gt_binary: numpy array (H, W), binary ground-truth foreground mask
        pred_instances: numpy array (H, W), integer predicted instance label map
        gt_instances: numpy array (H, W), integer ground-truth instance label map
    Outputs:
        dict with keys 'dice', 'aji', 'pq', 'sq', 'dq', all floats in [0, 1]
    """
    dice = dice_coefficient(pred_binary, gt_binary)
    aji = aggregated_jaccard_index(pred_instances, gt_instances)
    pq_dict = panoptic_quality(pred_instances, gt_instances)

    return {
        "dice": dice,
        "aji": aji,
        "pq": pq_dict["pq"],
        "sq": pq_dict["sq"],
        "dq": pq_dict["dq"],
    }
