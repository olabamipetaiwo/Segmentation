"""
Loss functions for nuclei instance segmentation training.
Dice loss, Focal loss, and their combination.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Binary Dice loss operating on sigmoid-activated predictions.

    Inputs to forward:
        predictions: float tensor of any shape, raw logits
        targets: float tensor of the same shape, binary labels in [0, 1]
    Outputs:
        scalar loss value in [0, 1]
    """

    def __init__(self, smooth: float = 1.0) -> None:
        """
        Inputs:
            smooth: additive smoothing constant to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute per-instance Dice loss averaged across the batch.

        Computing Dice per nucleus (rather than globally flattening all N masks)
        gives each instance equal gradient weight regardless of its area.

        Inputs:
            predictions: float tensor (N, 1, H, W) or any shape, raw logits
            targets: float tensor of same shape, binary ground truth in [0, 1]
        Outputs:
            scalar Dice loss
        """
        activated = torch.sigmoid(predictions)
        N = predictions.shape[0]
        pred_flat = activated.view(N, -1)     # (N, H*W)
        target_flat = targets.view(N, -1)     # (N, H*W)

        intersection = (pred_flat * target_flat).sum(dim=1)   # (N,)
        dice_per = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return (1.0 - dice_per).mean()


class FocalLoss(nn.Module):
    """
    Binary focal loss for handling class imbalance in segmentation.

    Inputs to forward:
        predictions: float tensor, raw logits
        targets: float tensor, binary labels in [0, 1]
    Outputs:
        scalar focal loss value
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        """
        Inputs:
            gamma: focusing exponent, reduces loss contribution from easy examples
            alpha: weighting factor for the positive class
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute binary focal loss.

        Inputs:
            predictions: float tensor of any shape, raw logits
            targets: float tensor of same shape, binary ground truth in [0, 1]
        Outputs:
            scalar focal loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction="none"
        )
        probabilities = torch.sigmoid(predictions)
        pt = torch.where(targets >= 0.5, probabilities, 1.0 - probabilities)
        alpha_t = torch.where(targets >= 0.5,
                              torch.full_like(targets, self.alpha),
                              torch.full_like(targets, 1.0 - self.alpha))
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        return (focal_weight * bce_loss).mean()


class CombinedLoss(nn.Module):
    """
    Equal-weighted sum of Dice loss and Focal loss.

    Inputs to forward:
        predictions: float tensor, raw logits
        targets: float tensor, binary labels in [0, 1]
    Outputs:
        scalar combined loss value
    """

    def __init__(
        self,
        dice_smooth: float = 1.0,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ) -> None:
        """
        Inputs:
            dice_smooth: smoothing constant for Dice loss
            focal_gamma: focusing exponent for Focal loss
            focal_alpha: positive-class weight for Focal loss
        """
        super().__init__()
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.focal_loss = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined Dice + Focal loss with equal weighting.

        Inputs:
            predictions: float tensor of any shape, raw logits
            targets: float tensor of same shape, binary ground truth in [0, 1]
        Outputs:
            scalar combined loss
        """
        return self.dice_loss(predictions, targets) + self.focal_loss(predictions, targets)
