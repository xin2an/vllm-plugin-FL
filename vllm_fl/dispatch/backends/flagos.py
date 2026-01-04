# Copyright (c) 2025 BAAI. All rights reserved.

"""
FlagOS backend implementations (DEFAULT).

Uses flag_gems library for cross-platform Triton-based implementations.
Works on both CUDA and NPU platforms.
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
# Availability check
# -----------------------------------------------------------------------------


def _is_flag_gems_available() -> bool:
    """Check if flag_gems is available."""
    try:
        import flag_gems
        return True
    except ImportError:
        return False


# -----------------------------------------------------------------------------
# RMSNorm
# -----------------------------------------------------------------------------


def _rms_norm_flagos(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
) -> tuple:
    """RMSNorm using flag_gems (FlagOS default)."""
    from flag_gems.modules.normalization import gems_rms_forward
    return gems_rms_forward(x, residual, weight, eps)


_rms_norm_flagos._is_available = _is_flag_gems_available


# -----------------------------------------------------------------------------
# SiluAndMul
# -----------------------------------------------------------------------------


def _silu_and_mul_flagos(x: torch.Tensor) -> torch.Tensor:
    """SiluAndMul using flag_gems (FlagOS default)."""
    from flag_gems.modules.activation import gems_silu_and_mul

    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gems_silu_and_mul(x1, x2)


_silu_and_mul_flagos._is_available = _is_flag_gems_available


# -----------------------------------------------------------------------------
# RotaryEmbedding (RoPE)
# -----------------------------------------------------------------------------


def _rope_flagos(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    head_size: int,
    is_neox_style: bool = True,
) -> tuple:
    """RoPE using flag_gems (FlagOS default)."""
    from flag_gems.modules.rotary_embedding import gems_rope_forward

    num_tokens = positions.shape[0]
    query_shape = query.shape
    key_shape = key.shape

    query = query.view(num_tokens, -1, head_size)
    key = key.view(num_tokens, -1, head_size)

    query_rot = query[..., :rotary_dim]
    key_rot = key[..., :rotary_dim]

    q_embed, k_embed = gems_rope_forward(
        query_rot,
        key_rot,
        cos,
        sin,
        position_ids=positions,
        rotary_interleaved=not is_neox_style,
        inplace=True,
    )

    if rotary_dim < head_size:
        query_pass = query[..., rotary_dim:]
        key_pass = key[..., rotary_dim:]
        query = torch.cat((q_embed, query_pass), dim=-1).reshape(query_shape)
        key = torch.cat((k_embed, key_pass), dim=-1).reshape(key_shape)
    else:
        query = q_embed.reshape(query_shape)
        key = k_embed.reshape(key_shape)

    return query, key


_rope_flagos._is_available = _is_flag_gems_available


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

# All implementations provided by this backend
_IMPLEMENTATIONS = [
    OpImpl(
        op_name="rms_norm",
        impl_id="default.flagos",
        kind=OpImplKind.DEFAULT,
        fn=_rms_norm_flagos,
        priority=100,
        description="RMSNorm using flag_gems (FlagOS)",
    ),
    OpImpl(
        op_name="silu_and_mul",
        impl_id="default.flagos",
        kind=OpImplKind.DEFAULT,
        fn=_silu_and_mul_flagos,
        priority=100,
        description="SiluAndMul using flag_gems (FlagOS)",
    ),
    OpImpl(
        op_name="rotary_embedding",
        impl_id="default.flagos",
        kind=OpImplKind.DEFAULT,
        fn=_rope_flagos,
        priority=100,
        description="RoPE using flag_gems (FlagOS)",
    ),
]


def register(registry: "OpRegistry") -> None:
    """Register all FlagOS (DEFAULT) implementations."""
    for impl in _IMPLEMENTATIONS:
        try:
            registry.register_impl(impl, skip_duplicate=True)
        except Exception as e:
            logger.warning(f"Failed to register {impl.impl_id}: {e}")

    logger.debug(f"Registered {len(_IMPLEMENTATIONS)} FlagOS implementations")


def get_implementations() -> list:
    """Get all implementations provided by this backend."""
    return list(_IMPLEMENTATIONS)
