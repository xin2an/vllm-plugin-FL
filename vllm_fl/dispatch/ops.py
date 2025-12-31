"""
Convenience API for operator dispatch.

Provides simple function calls that automatically use the dispatch system.

Usage:
    from vllm_fl.dispatch.ops import rms_norm, silu_and_mul

    # Automatically dispatches to best available implementation
    output = rms_norm(x, weight, eps=1e-6)
    output = silu_and_mul(x)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .manager import get_default_manager

_manager = get_default_manager()


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, ...]:
    """
    RMSNorm with automatic backend dispatch.

    Args:
        x: Input tensor
        weight: Normalization weight
        eps: Epsilon for numerical stability
        residual: Optional residual tensor

    Returns:
        Normalized output (and residual if provided)
    """
    return _manager.call("rms_norm", x, weight, eps, residual)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation with element-wise multiplication.

    Args:
        x: Input tensor (last dim will be split in half)

    Returns:
        silu(x[..., :d]) * x[..., d:]
    """
    return _manager.call("silu_and_mul", x)


def rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    head_size: int,
    is_neox_style: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rotary Position Embedding (RoPE).

    Args:
        query: Query tensor
        key: Key tensor
        cos: Cosine cache
        sin: Sine cache
        positions: Position indices
        rotary_dim: Rotary dimension
        head_size: Head size
        is_neox_style: Use NeoX style rotation

    Returns:
        (rotated_query, rotated_key)
    """
    return _manager.call(
        "rotary_embedding",
        query, key, cos, sin, positions,
        rotary_dim, head_size, is_neox_style
    )
