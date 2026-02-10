# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX activation operator implementations.
"""

from __future__ import annotations

import torch


def silu_and_mul_metax(instance, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using METAX/MACA.

    Args:
        instance: The calling instance (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    # TODO: Implement METAX-specific optimized version
    # For now, use PyTorch reference implementation
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.nn.functional.silu(x1) * x2
