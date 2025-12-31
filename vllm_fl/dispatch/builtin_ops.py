"""
Builtin operator implementations for vllm-plugin-FL dispatch system.

Registers DEFAULT (FlagOS), REFERENCE (PyTorch), and VENDOR (CUDA/NPU) implementations.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from .types import OpImpl, OpImplKind

if TYPE_CHECKING:
    from .registry import OpRegistry

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


def _is_npu_available() -> bool:
    """Check if NPU is available."""
    try:
        from vllm_fl.utils import is_npu
        return is_npu()
    except Exception:
        return False


def _is_flag_gems_available() -> bool:
    """Check if flag_gems is available."""
    try:
        import flag_gems
        return True
    except ImportError:
        return False


def _is_torch_npu_available() -> bool:
    """Check if torch_npu is available."""
    try:
        import torch_npu
        return True
    except ImportError:
        return False


def _is_vllm_cuda_ops_available() -> bool:
    """Check if vLLM CUDA ops are available."""
    try:
        import vllm._C
        return _is_cuda_available()
    except ImportError:
        return False


# -----------------------------------------------------------------------------
# RMSNorm implementations
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


_rms_norm_flagos._is_available = lambda: _is_flag_gems_available()


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


_rms_norm_npu._is_available = lambda: _is_npu_available() and _is_torch_npu_available()


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


_rms_norm_cuda._is_available = lambda: _is_vllm_cuda_ops_available()


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
# SiluAndMul implementations
# -----------------------------------------------------------------------------


def _silu_and_mul_flagos(x: torch.Tensor) -> torch.Tensor:
    """SiluAndMul using flag_gems (FlagOS default)."""
    from flag_gems.modules.activation import gems_silu_and_mul

    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gems_silu_and_mul(x1, x2)


_silu_and_mul_flagos._is_available = lambda: _is_flag_gems_available()


def _silu_and_mul_cuda(x: torch.Tensor) -> torch.Tensor:
    """SiluAndMul using vLLM CUDA kernels."""
    d = x.shape[-1] // 2
    output = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(output, x)
    return output


_silu_and_mul_cuda._is_available = lambda: _is_vllm_cuda_ops_available()


def _silu_and_mul_reference(x: torch.Tensor) -> torch.Tensor:
    """SiluAndMul reference implementation using PyTorch."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.nn.functional.silu(x1) * x2


_silu_and_mul_reference._is_available = lambda: True


# -----------------------------------------------------------------------------
# RotaryEmbedding implementations
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


_rope_flagos._is_available = lambda: _is_flag_gems_available()


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
    torch.ops._C.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox_style)

    return query, key


_rope_cuda._is_available = lambda: _is_vllm_cuda_ops_available()


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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half for interleaved RoPE."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


_rope_reference._is_available = lambda: True


# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------


def register_builtins(registry: "OpRegistry") -> None:
    """Register all builtin operator implementations."""
    impls = [
        # RMSNorm
        OpImpl(
            op_name="rms_norm",
            impl_id="default.flagos",
            kind=OpImplKind.DEFAULT,
            fn=_rms_norm_flagos,
            priority=100,
            description="RMSNorm using flag_gems (FlagOS)",
        ),
        OpImpl(
            op_name="rms_norm",
            impl_id="vendor.npu",
            kind=OpImplKind.VENDOR,
            vendor="npu",
            fn=_rms_norm_npu,
            priority=100,
            description="RMSNorm using torch_npu (Ascend)",
        ),
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
            op_name="rms_norm",
            impl_id="reference.torch",
            kind=OpImplKind.REFERENCE,
            fn=_rms_norm_reference,
            priority=50,
            description="RMSNorm reference implementation (PyTorch)",
        ),

        # SiluAndMul
        OpImpl(
            op_name="silu_and_mul",
            impl_id="default.flagos",
            kind=OpImplKind.DEFAULT,
            fn=_silu_and_mul_flagos,
            priority=100,
            description="SiluAndMul using flag_gems (FlagOS)",
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
            op_name="silu_and_mul",
            impl_id="reference.torch",
            kind=OpImplKind.REFERENCE,
            fn=_silu_and_mul_reference,
            priority=50,
            description="SiluAndMul reference implementation (PyTorch)",
        ),

        # RotaryEmbedding (RoPE)
        OpImpl(
            op_name="rotary_embedding",
            impl_id="default.flagos",
            kind=OpImplKind.DEFAULT,
            fn=_rope_flagos,
            priority=100,
            description="RoPE using flag_gems (FlagOS)",
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
        OpImpl(
            op_name="rotary_embedding",
            impl_id="reference.torch",
            kind=OpImplKind.REFERENCE,
            fn=_rope_reference,
            priority=50,
            description="RoPE reference implementation (PyTorch)",
        ),
    ]

    for impl in impls:
        try:
            registry.register_impl(impl, skip_duplicate=True)
        except Exception as e:
            logger.warning(f"Failed to register {impl.impl_id}: {e}")

    logger.debug(f"Registered {len(impls)} builtin operator implementations")
