# Copyright (c) 2025 BAAI. All rights reserved.

"""
CUDA backend implementations (VENDOR).

Uses vLLM's optimized CUDA kernels for NVIDIA GPUs.
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
# Availability checks
# -----------------------------------------------------------------------------


def _is_cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        from vllm_fl.utils import is_cuda
        return is_cuda()
    except Exception:
        return torch.cuda.is_available()


def _is_vllm_cuda_ops_available() -> bool:
    """Check if vLLM CUDA ops are available."""
    try:
        import vllm._C
        return _is_cuda_available()
    except ImportError:
        return False


# -----------------------------------------------------------------------------
# RMSNorm
# -----------------------------------------------------------------------------


def _rms_norm_cuda(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
) -> tuple:
    """RMSNorm using vLLM CUDA kernels."""
    if residual is not None:
        torch.ops._C.fused_add_rms_norm(x, residual, weight, eps)
        return x, residual
    else:
        output = torch.empty_like(x)
        torch.ops._C.rms_norm(output, x, weight, eps)
        return output


_rms_norm_cuda._is_available = _is_vllm_cuda_ops_available


# -----------------------------------------------------------------------------
# SiluAndMul
# -----------------------------------------------------------------------------


def _silu_and_mul_cuda(x: torch.Tensor) -> torch.Tensor:
    """SiluAndMul using vLLM CUDA kernels."""
    d = x.shape[-1] // 2
    output = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(output, x)
    return output


_silu_and_mul_cuda._is_available = _is_vllm_cuda_ops_available


# -----------------------------------------------------------------------------
# RotaryEmbedding (RoPE)
# -----------------------------------------------------------------------------


def _rope_cuda(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rotary_dim: int,
    head_size: int,
    is_neox_style: bool = True,
) -> tuple:
    """RoPE using vLLM CUDA kernels."""
    # vLLM expects cos_sin_cache as interleaved [cos, sin] tensor
    # Shape: [max_position, rotary_dim]
    cos_sin_cache = torch.cat([cos, sin], dim=-1)

    # vLLM rotary_embedding modifies query and key in-place
    query = query.clone()
    key = key.clone()
    torch.ops._C.rotary_embedding(
        positions, query, key, head_size, cos_sin_cache, is_neox_style
    )

    return query, key


_rope_cuda._is_available = _is_vllm_cuda_ops_available


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

# All implementations provided by this backend
_IMPLEMENTATIONS = [
    OpImpl(
        op_name="rms_norm",
        impl_id="vendor.cuda",
        kind=OpImplKind.VENDOR,
        vendor="cuda",
        fn=_rms_norm_cuda,
        priority=100,
        description="RMSNorm using vLLM CUDA kernels",
    ),
    OpImpl(
        op_name="silu_and_mul",
        impl_id="vendor.cuda",
        kind=OpImplKind.VENDOR,
        vendor="cuda",
        fn=_silu_and_mul_cuda,
        priority=100,
        description="SiluAndMul using vLLM CUDA kernels",
    ),
    OpImpl(
        op_name="rotary_embedding",
        impl_id="vendor.cuda",
        kind=OpImplKind.VENDOR,
        vendor="cuda",
        fn=_rope_cuda,
        priority=100,
        description="RoPE using vLLM CUDA kernels",
    ),
]


def register(registry: "OpRegistry") -> None:
    """Register all CUDA (VENDOR) implementations."""
    for impl in _IMPLEMENTATIONS:
        try:
            registry.register_impl(impl, skip_duplicate=True)
        except Exception as e:
            logger.warning(f"Failed to register {impl.impl_id}: {e}")

    logger.debug(f"Registered {len(_IMPLEMENTATIONS)} CUDA implementations")


def get_implementations() -> list:
    """Get all implementations provided by this backend."""
    return list(_IMPLEMENTATIONS)
