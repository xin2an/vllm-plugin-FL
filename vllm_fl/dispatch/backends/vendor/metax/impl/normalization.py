# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX normalization operator implementations.
"""

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_metax(
    instance,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using METAX/MACA.

    Args:
        instance: The calling instance (e.g., RMSNorm layer)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual is provided
    """
    # Get weight and epsilon from instance
    weight = instance.weight
    epsilon = instance.variance_epsilon

    # TODO: Implement METAX-specific optimized version
    # For now, use PyTorch reference implementation
    if residual is not None:
        x = x + residual
        residual = x

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    x = x * weight

    if residual is not None:
        return x, residual
    return x
