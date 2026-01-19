# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rmsnorm_ascend(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    epsilon: float,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using Ascend NPU.

    Args:
        x: Input tensor
        residual: Optional residual tensor to add before normalization
        weight: Normalization weight
        epsilon: Small constant for numerical stability

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    orig_dtype = x.dtype
    x = x.float()

    if residual is not None:
        residual = residual.float()
        x = x + residual

    # Compute RMS normalization
    # RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)

    # Apply weight
    x = x * weight

    x = x.to(orig_dtype)

    if residual is not None:
        return x, residual.to(orig_dtype)
    return x
