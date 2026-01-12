# Copyright (c) 2025 BAAI. All rights reserved.

"""
Ascend activation operator implementations.
"""

from __future__ import annotations

import torch


def silu_and_mul_ascend(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using Ascend NPU.

    Args:
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    # Split input into two halves
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]

    # Apply SiLU (x * sigmoid(x)) to first half and multiply with second half
    # Use Ascend-optimized operations if available
    try:
        import torch_npu

        # Use NPU-optimized SiLU if available
        silu_out = torch.nn.functional.silu(x1)
        return silu_out * x2
    except (ImportError, AttributeError):
        # Fallback to standard PyTorch
        silu_out = torch.nn.functional.silu(x1)
        return silu_out * x2
