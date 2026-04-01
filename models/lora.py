"""
LoRA (Low-Rank Adaptation) layer and injection utilities for MobileSAM.

Implements LoRALayer as a wrapper around nn.Linear, QKVLoRAWrapper to replace
fused QKV projections in TinyViT attention blocks, and inject_lora to apply
LoRA to all attention blocks in the image encoder.

References:
    LoRA paper: https://arxiv.org/abs/2106.09685
    LoRA-ViT reference: https://github.com/JamesQFreeman/LoRA-ViT
    finetuneSAM reference: https://github.com/mazurowski-lab/finetuneSAM
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    LoRA wrapper around an existing nn.Linear layer.

    Freezes the original weight and bias, then adds two trainable low-rank
    matrices A (rank x d_in) and B (d_out x rank). The forward pass computes:
        output = original_linear(x) + scaling * (x @ A.T @ B.T)
    where scaling = alpha / rank.

    Inputs to __init__:
        linear_layer: the existing nn.Linear to wrap (its parameters are frozen)
        rank: rank of the low-rank decomposition (default 4)
        alpha: scaling factor for the LoRA update (default 1.0)
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.original_linear = linear_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = linear_layer.in_features
        out_features = linear_layer.out_features

        # Freeze the original weights so only lora_A and lora_B train
        for param in self.original_linear.parameters():
            param.requires_grad = False

        # Low-rank trainable matrices: A initialised Kaiming uniform, B zeros
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the LoRA-adapted linear transformation.

        Inputs:
            x: float tensor of shape (..., in_features)
        Outputs:
            float tensor of shape (..., out_features)
        """
        original_out = self.original_linear(x)
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return original_out + self.scaling * lora_out


class QKVLoRAWrapper(nn.Module):
    """
    Replacement for a fused QKV linear that applies LoRA to Q and V while
    keeping K frozen.

    The fused QKV linear of shape (dim -> 3*dim) is split into separate
    Q (LoRALayer), K (frozen nn.Linear), and V (LoRALayer) projections.
    The forward method concatenates their outputs to reproduce the original
    (... -> 3*dim) output shape expected by the rest of the attention block.

    Inputs to __init__:
        q_layer: LoRALayer wrapping the Q projection
        k_layer: frozen nn.Linear for the K projection
        v_layer: LoRALayer wrapping the V projection
    """

    def __init__(
        self,
        q_layer: LoRALayer,
        k_layer: nn.Linear,
        v_layer: LoRALayer,
    ) -> None:
        super().__init__()
        self.q_layer = q_layer
        self.k_layer = k_layer
        self.v_layer = v_layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project input to Q, K, V and concatenate to match the original fused output.

        Inputs:
            x: float tensor of shape (..., dim)
        Outputs:
            float tensor of shape (..., 3 * dim)
        """
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)
        return torch.cat([q, k, v], dim=-1)


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
) -> int:
    """
    Inject LoRA adapters into every attention block in the MobileSAM image encoder.

    MobileSAM's TinyViT uses fused QKV projections (nn.Linear, out = 3 * in).
    This function:
      1. Finds every such fused QKV linear in the image encoder.
      2. Splits its weights into separate Q, K, V nn.Linear layers.
      3. Wraps Q and V with LoRALayer; K remains frozen.
      4. Replaces the original qkv attribute with a QKVLoRAWrapper.
      5. Freezes all parameters in the model.
      6. Unfreezes only the lora_A and lora_B parameters in every LoRALayer.

    Note: the mask decoder is NOT touched here; callers should unfreeze it
    separately if domain adaptation is desired.

    Inputs:
        model: MobileSAM model (expects model.image_encoder to contain TinyViT blocks)
        rank: rank for LoRA low-rank matrices
        alpha: LoRA scaling factor
    Outputs:
        num_trainable: total count of trainable parameters after injection
                       (counts only LoRA parameters, not mask decoder)
    """
    # Pass 1: replace fused qkv with QKVLoRAWrapper in every attention block
    for module in model.image_encoder.modules():
        if not (hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear)):
            continue

        qkv_linear = module.qkv
        in_features = qkv_linear.in_features
        out_features = qkv_linear.out_features
        has_bias = qkv_linear.bias is not None

        # Only handle equal-split QKV (standard TinyViT window attention)
        if out_features != 3 * in_features:
            continue

        dim = in_features
        weight = qkv_linear.weight.data  # (3*dim, dim)

        # Split into Q, K, V weight chunks
        q_weight = weight[:dim, :].clone()
        k_weight = weight[dim : 2 * dim, :].clone()
        v_weight = weight[2 * dim :, :].clone()

        q_linear = nn.Linear(dim, dim, bias=has_bias)
        k_linear = nn.Linear(dim, dim, bias=has_bias)
        v_linear = nn.Linear(dim, dim, bias=has_bias)

        q_linear.weight.data = q_weight
        k_linear.weight.data = k_weight
        v_linear.weight.data = v_weight

        if has_bias:
            bias = qkv_linear.bias.data  # (3*dim,)
            q_linear.bias.data = bias[:dim].clone()
            k_linear.bias.data = bias[dim : 2 * dim].clone()
            v_linear.bias.data = bias[2 * dim :].clone()

        # Freeze K
        for param in k_linear.parameters():
            param.requires_grad = False

        # Wrap Q and V with LoRA
        q_lora = LoRALayer(q_linear, rank=rank, alpha=alpha)
        v_lora = LoRALayer(v_linear, rank=rank, alpha=alpha)

        # Replace the fused qkv with the wrapper
        module.qkv = QKVLoRAWrapper(q_lora, k_linear, v_lora)

    # Pass 2: freeze every parameter in the entire model
    for param in model.parameters():
        param.requires_grad = False

    # Pass 3: unfreeze only the LoRA matrices
    for module in model.modules():
        if isinstance(module, LoRALayer):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA injection complete. Tunable LoRA parameters: {num_trainable:,}")
    return num_trainable
