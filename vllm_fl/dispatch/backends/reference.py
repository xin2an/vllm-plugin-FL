# Copyright (c) 2025 BAAI. All rights reserved.

"""
Reference backend implementations (REFERENCE).

Pure PyTorch implementations that serve as fallback.
Always available on any platform with PyTorch installed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from ..types import OpImpl, OpImplKind

if TYPE_CHECKING:
    from ..registry import OpRegistry

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# RMSNorm
# -----------------------------------------------------------------------------


def _rms_norm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
) -> tuple:
    """RMSNorm reference implementation using PyTorch."""
    if residual is not None:
        x = x + residual
        residual = x

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    output = x_normed * weight

    if residual is None:
        return output
    else:
        return output, residual


_rms_norm_reference._is_available = lambda: True


# -----------------------------------------------------------------------------
# SiluAndMul
# -----------------------------------------------------------------------------


def _silu_and_mul_reference(x: torch.Tensor) -> torch.Tensor:
    """SiluAndMul reference implementation using PyTorch."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.nn.functional.silu(x1) * x2


_silu_and_mul_reference._is_available = lambda: True


# -----------------------------------------------------------------------------
# RotaryEmbedding (RoPE)
# -----------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half for interleaved RoPE."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _rope_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    head_size: int,
    is_neox_style: bool = True,
) -> tuple:
    """RoPE reference implementation using PyTorch."""
    num_tokens = positions.shape[0]
    query_shape = query.shape
    key_shape = key.shape

    query = query.view(num_tokens, -1, head_size)
    key = key.view(num_tokens, -1, head_size)

    query_rot = query[..., :rotary_dim]
    key_rot = key[..., :rotary_dim]

    # Get cos/sin for positions
    cos_pos = cos[positions]
    sin_pos = sin[positions]

    # Apply rotation
    if is_neox_style:
        # Neox style: split in half
        q1, q2 = query_rot.chunk(2, dim=-1)
        k1, k2 = key_rot.chunk(2, dim=-1)

        cos_pos = cos_pos.unsqueeze(1)
        sin_pos = sin_pos.unsqueeze(1)

        q_embed = torch.cat([
            q1 * cos_pos - q2 * sin_pos,
            q1 * sin_pos + q2 * cos_pos
        ], dim=-1)
        k_embed = torch.cat([
            k1 * cos_pos - k2 * sin_pos,
            k1 * sin_pos + k2 * cos_pos
        ], dim=-1)
    else:
        # Interleaved style
        cos_pos = cos_pos.unsqueeze(1)
        sin_pos = sin_pos.unsqueeze(1)
        q_embed = query_rot * cos_pos + _rotate_half(query_rot) * sin_pos
        k_embed = key_rot * cos_pos + _rotate_half(key_rot) * sin_pos

    if rotary_dim < head_size:
        query_pass = query[..., rotary_dim:]
        key_pass = key[..., rotary_dim:]
        query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
        key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
    else:
        query = q_embed.reshape(query_shape)
        key = k_embed.reshape(key_shape)

    return query, key


_rope_reference._is_available = lambda: True


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

# All implementations provided by this backend
_IMPLEMENTATIONS = [
    OpImpl(
        op_name="rms_norm",
        impl_id="reference.torch",
        kind=OpImplKind.REFERENCE,
        fn=_rms_norm_reference,
        priority=50,
        description="RMSNorm reference implementation (PyTorch)",
    ),
    OpImpl(
        op_name="silu_and_mul",
        impl_id="reference.torch",
        kind=OpImplKind.REFERENCE,
        fn=_silu_and_mul_reference,
        priority=50,
        description="SiluAndMul reference implementation (PyTorch)",
    ),
    OpImpl(
        op_name="rotary_embedding",
        impl_id="reference.torch",
        kind=OpImplKind.REFERENCE,
        fn=_rope_reference,
        priority=50,
        description="RoPE reference implementation (PyTorch)",
    ),
]


def register(registry: "OpRegistry") -> None:
    """Register all reference (PyTorch) implementations."""
    for impl in _IMPLEMENTATIONS:
        try:
            registry.register_impl(impl, skip_duplicate=True)
        except Exception as e:
            logger.warning(f"Failed to register {impl.impl_id}: {e}")

    logger.debug(f"Registered {len(_IMPLEMENTATIONS)} reference implementations")


def get_implementations() -> list:
    """Get all implementations provided by this backend."""
    return list(_IMPLEMENTATIONS)
