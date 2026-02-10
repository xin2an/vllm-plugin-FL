# Copyright (c) 2026 BAAI. All rights reserved.

"""
METAX backend implementation.

This backend provides operator implementations for Moore Threads METAX GPUs.
METAX uses MACA (Moore Threads Accelerated Computing Architecture) which is
CUDA-compatible.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class MetaxBackend(Backend):
    """
    METAX backend for operator implementations.

    This backend uses MACA libraries to provide high-performance
    operator implementations for Moore Threads METAX GPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "metax"

    @property
    def vendor(self) -> Optional[str]:
        return "metax"

    def is_available(self) -> bool:
        """
        Check if METAX hardware and libraries are available.

        This method uses the platform's vendor information to determine
        if the device is a METAX GPU.
        """
        if MetaxBackend._available is None:
            try:
                # Check if CUDA device is available (MACA is CUDA-compatible)
                if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                    MetaxBackend._available = False
                    return False

                # Use current_platform's vendor information to check if this is METAX
                try:
                    from vllm.platforms import current_platform

                    # Only enable METAX backend for metax vendor
                    if hasattr(current_platform, 'vendor_name') and current_platform.vendor_name == "metax":
                        MetaxBackend._available = True
                    else:
                        MetaxBackend._available = False

                except Exception:
                    # Fallback: check device name for MACA/METAX keywords
                    device_name = torch.cuda.get_device_name(0).upper()
                    if "MACA" in device_name or "METAX" in device_name or "MOORE" in device_name:
                        MetaxBackend._available = True
                    else:
                        MetaxBackend._available = False

            except Exception:
                MetaxBackend._available = False
        return MetaxBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, instance, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            instance: The calling instance (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_metax

        return silu_and_mul_metax(instance, x)

    def rms_norm(
        self,
        instance,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            instance: The calling instance (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_metax

        return rms_norm_metax(instance, x, residual)

    def rotary_embedding(
        self,
        instance,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding.

        Args:
            instance: The calling instance (for interface consistency)
            query: Query tensor
            key: Key tensor
            cos: Cosine cache
            sin: Sine cache
            position_ids: Position indices
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_metax

        return rotary_embedding_metax(
            instance,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False) -> str:
        """
        Get the attention backend class path for METAX.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        from vllm.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            # TODO: Implement METAX MLA backend
            return AttentionBackendEnum.FLASHMLA.get_path()

        # Default to FLASH_ATTN (MACA is CUDA-compatible)
        return AttentionBackendEnum.FLASH_ATTN.get_path()
