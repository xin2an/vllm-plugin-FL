# Copyright (c) 2026 BAAI. All rights reserved.

"""
CUDA backend implementation.

This backend provides operator implementations for NVIDIA CUDA GPUs.
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend


class CudaBackend(Backend):
    """
    CUDA backend for operator implementations.

    This backend uses CUDA libraries to provide high-performance
    operator implementations for NVIDIA GPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def vendor(self) -> Optional[str]:
        return "nvidia"

    def is_available(self) -> bool:
        """Check if CUDA hardware and libraries are available."""
        if CudaBackend._available is None:
            try:
                # Check if CUDA device is available
                if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                    CudaBackend._available = False
                    return False

                # Check if this is a real NVIDIA GPU (not CUDA-alike hardware)
                # Check device name to exclude CUDA-alike vendors
                device_name = torch.cuda.get_device_name(0).upper()

                # Exclude CUDA-alike vendors by device name
                # Note: MACA is the device name (like ROCm), METAX is the vendor name
                cuda_alike_device_names = ["MUSA", "MOORE", "MACA", "ILUVATAR",
                                          "HYGON", "DCU", "KUNLUN", "CAMBRICON"]
                for device_keyword in cuda_alike_device_names:
                    if device_keyword in device_name:
                        CudaBackend._available = False
                        return False

                # Verify it's NVIDIA or has CUDA in the name
                if "NVIDIA" in device_name or "CUDA" in device_name:
                    CudaBackend._available = True
                else:
                    # If device name doesn't contain NVIDIA or CUDA,
                    # it might be a CUDA-alike device
                    CudaBackend._available = False

            except Exception:
                CudaBackend._available = False
        return CudaBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, instance, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Uses vLLM's native CUDA implementation.

        Args:
            instance: The calling instance (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_cuda

        return silu_and_mul_cuda(instance, x)

    def rms_norm(
        self,
        instance,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization using vLLM's CUDA implementation.

        Args:
            instance: The calling instance (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_cuda

        return rms_norm_cuda(instance, x, residual)

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
        Apply rotary position embedding using vLLM's CUDA implementation.

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
        from .impl.rotary import rotary_embedding_cuda

        return rotary_embedding_cuda(
            instance,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, instance, use_mla: bool = False) -> str:
        """
        Get the attention backend class path for CUDA.

        Supports:
        - FLASH_ATTN (default)
        - TRITON_ATTN (when use_flaggems_op("triton_attn") is True)

        Args:
            instance: The calling instance (for interface consistency)
            use_mla: Whether to use Multi-head Latent Attention (MLA)

        Returns:
            Fully qualified class path string
        """
        from vllm.attention.backends.registry import AttentionBackendEnum
        from vllm_fl.utils import use_flaggems_op

        if use_mla:
            return AttentionBackendEnum.FLASHMLA.get_path()

        # Default to FLASH_ATTN
        return AttentionBackendEnum.FLASH_ATTN.get_path()
