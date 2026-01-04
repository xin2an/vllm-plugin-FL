# Copyright (c) 2025 BAAI. All rights reserved.

"""
NPU backend implementations (VENDOR).

Uses torch_npu library for Ascend NPU devices.
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


def _is_npu_available() -> bool:
    """Check if NPU is available."""
    try:
        from vllm_fl.utils import is_npu
        return is_npu()
    except Exception:
        return False


def _is_torch_npu_available() -> bool:
    """Check if torch_npu is available."""
    try:
        import torch_npu
        return True
    except ImportError:
        return False


def _is_npu_ops_available() -> bool:
    """Check if NPU and torch_npu are both available."""
    return _is_npu_available() and _is_torch_npu_available()


# -----------------------------------------------------------------------------
# RMSNorm
# -----------------------------------------------------------------------------


def _rms_norm_npu(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    residual: Optional[torch.Tensor] = None,
) -> tuple:
    """RMSNorm using torch_npu (Ascend NPU)."""
    import torch_npu

    if residual is not None:
        x = x + residual
        residual = x

    output = torch_npu.npu_rms_norm(x, weight, epsilon=eps)[0]

    if residual is None:
        return output
    else:
        return output, residual


_rms_norm_npu._is_available = _is_npu_ops_available


# -----------------------------------------------------------------------------
# SiluAndMul
# -----------------------------------------------------------------------------

# Note: NPU-specific SiluAndMul can be added here when available
# Currently, NPU falls back to FlagOS or reference implementation


# -----------------------------------------------------------------------------
# RotaryEmbedding (RoPE)
# -----------------------------------------------------------------------------

# Note: NPU-specific RoPE can be added here when available
# Currently, NPU falls back to FlagOS or reference implementation


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

# All implementations provided by this backend
_IMPLEMENTATIONS = [
    OpImpl(
        op_name="rms_norm",
        impl_id="vendor.npu",
        kind=OpImplKind.VENDOR,
        vendor="npu",
        fn=_rms_norm_npu,
        priority=100,
        description="RMSNorm using torch_npu (Ascend)",
    ),
    # Add more NPU implementations here as they become available:
    # OpImpl(
    #     op_name="silu_and_mul",
    #     impl_id="vendor.npu",
    #     kind=OpImplKind.VENDOR,
    #     vendor="npu",
    #     fn=_silu_and_mul_npu,
    #     priority=100,
    #     description="SiluAndMul using torch_npu (Ascend)",
    # ),
]


def register(registry: "OpRegistry") -> None:
    """Register all NPU (VENDOR) implementations."""
    for impl in _IMPLEMENTATIONS:
        try:
            registry.register_impl(impl, skip_duplicate=True)
        except Exception as e:
            logger.warning(f"Failed to register {impl.impl_id}: {e}")

    logger.debug(f"Registered {len(_IMPLEMENTATIONS)} NPU implementations")


def get_implementations() -> list:
    """Get all implementations provided by this backend."""
    return list(_IMPLEMENTATIONS)
