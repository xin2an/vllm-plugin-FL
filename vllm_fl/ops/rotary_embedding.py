# Copyright (c) 2025 BAAI. All rights reserved.

"""
Multi-backend Rotary Embedding (RoPE) implementation using dispatch system.
"""
from typing import Optional

import torch
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

from vllm_fl.dispatch import get_default_manager


class RotaryEmbeddingFL(RotaryEmbedding):
    """Multi-backend RoPE implementation using dispatch system.

    Automatically selects the best available implementation based on:
    - Device availability (CUDA, NPU)
    - Library availability (flag_gems)
    - User-configured policy (environment variables)

    Environment Variables:
        VLLM_FL_PREFER: Preferred backend (default/vendor/reference)
        VLLM_FL_PER_OP: Per-operator control (e.g., "rotary_embedding=default|reference")
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings,
            base, is_neox_style, dtype
        )
        self._dispatch_manager = get_default_manager()

    def forward_oot(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass using dispatch system to select best implementation."""
        # Prepare cos/sin cache
        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        positions = positions.flatten()

        cos, sin = self.cos_sin_cache.chunk(2, dim=-1)

        # Use dispatch system to select and call best implementation
        return self._dispatch_manager.call(
            "rotary_embedding",
            query,
            key,
            cos,
            sin,
            positions,
            self.rotary_dim,
            self.head_size,
            self.is_neox_style,
        )


__all__ = ["RotaryEmbeddingFL"]
