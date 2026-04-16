"""
MobileSAM model wrapped with LoRA adapters in the image encoder.

Loads pretrained MobileSAM weights, injects LoRA into every attention block
of the TinyViT image encoder, keeps the mask decoder fully trainable for
domain adaptation, and freezes all other parameters.

MobileSAM GitHub: https://github.com/ChaoningZhang/MobileSAM
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mobile_sam import sam_model_registry
except ImportError as exc:
    raise ImportError(
        "MobileSAM not found. Install it with:\n"
        "  git clone https://github.com/ChaoningZhang/MobileSAM\n"
        "  cd MobileSAM && pip install -e ."
    ) from exc

from models.lora import inject_lora


class MobileSAMLoRA(nn.Module):
    """
    MobileSAM with LoRA adapters injected into the image encoder.

    The image encoder's QKV projections gain trainable low-rank A and B matrices
    while all other image encoder parameters remain frozen. The mask decoder is
    kept fully trainable to allow domain adaptation to the nuclei segmentation task.

    Inputs to __init__:
        checkpoint_path: path to the pretrained MobileSAM .pt checkpoint
        lora_rank: rank for LoRA decomposition (default 4)
        lora_alpha: scaling factor for LoRA updates (default 1.0)
    """

    def __init__(
        self,
        checkpoint_path: str,
        lora_rank: int = 4,
        lora_alpha: float = 1.0,
    ) -> None:
        super().__init__()

        # Load pretrained MobileSAM (TinyViT image encoder + SAM mask decoder)
        self.sam = sam_model_registry["vit_t"](checkpoint=checkpoint_path)

        # Inject LoRA into image encoder; freezes everything except LoRA params
        self.num_lora_params = inject_lora(
            self.sam, rank=lora_rank, alpha=lora_alpha
        )

        # Unfreeze mask decoder for task-specific adaptation
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True

        num_decoder_params = sum(
            p.numel() for p in self.sam.mask_decoder.parameters()
        )
        total_trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        print(
            f"Mask decoder parameters (unfrozen): {num_decoder_params:,}\n"
            f"Total trainable parameters: {total_trainable:,}"
        )

    def forward(
        self,
        image: torch.Tensor,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MobileSAM with either box or point prompts on a single preprocessed image.

        Exactly one of ``boxes`` or (``point_coords``, ``point_labels``) must be
        provided. Box prompts are preferred: each box tightly encodes the spatial
        extent of a nucleus, giving the decoder stronger positional signal than a
        single centroid point.

        The image encoder is run once; its embedding is shared across all N prompts
        by SAM's internal repeat_interleave mechanism in the mask decoder.

        Inputs:
            image: float32 tensor (1, 3, 1024, 1024) normalised with SAM pixel stats
            point_coords: float32 tensor (N, P, 2) in (x, y) = (col, row) order
            point_labels: int32 tensor (N, P)
            boxes: float32 tensor (N, 4) in SAM [x1, y1, x2, y2] = [col_min, row_min,
                   col_max, row_max] order; when provided, point prompts are ignored
        Outputs:
            low_res_masks: float32 tensor (N, 1, 256, 256), raw mask logits
            iou_predictions: float32 tensor (N, 1), predicted IoU scores
        """
        device = image.device

        # Determine number of prompts and early-exit on empty input
        if boxes is not None:
            num_prompts = boxes.shape[0]
        elif point_coords is not None:
            num_prompts = point_coords.shape[0]
        else:
            num_prompts = 0

        if num_prompts == 0:
            return (
                torch.zeros(0, 1, 256, 256, device=device),
                torch.zeros(0, 1, device=device),
            )

        # Encode image once: (1, 256, 64, 64)
        image_embedding = self.sam.image_encoder(image)

        # Dense positional encoding: (1, 256, 64, 64)
        # SAM's mask decoder calls repeat_interleave internally to expand from
        # batch_size=1 to batch_size=N (number of prompts).
        image_pe = self.sam.prompt_encoder.get_dense_pe()

        # Encode prompts — use boxes when available, fall back to points
        if boxes is not None:
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=None,
                boxes=boxes,    # (N, 4) → encodes top-left & bottom-right corners
                masks=None,
            )
        else:
            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )

        # Decode masks; mask decoder expands image_embedding from (1,...) to (N,...)
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=image_embedding,   # (1, 256, 64, 64) - SAM expands internally
            image_pe=image_pe,                  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        return low_res_masks, iou_predictions

    def upsample_masks(
        self,
        low_res_masks: torch.Tensor,
        target_size: Tuple[int, int] = (1024, 1024),
    ) -> torch.Tensor:
        """
        Bilinearly upsample low-resolution mask logits to the target image size.

        Inputs:
            low_res_masks: float tensor (N, 1, 256, 256), raw logits
            target_size: (H, W) tuple for the output spatial dimensions
        Outputs:
            float tensor (N, 1, H, W) at the requested resolution
        """
        return F.interpolate(
            low_res_masks,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
