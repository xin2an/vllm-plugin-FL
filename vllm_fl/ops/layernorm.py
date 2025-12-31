# Copyright (c) 2025 BAAI. All rights reserved.

"""
Multi-backend LayerNorm implementation using dispatch system.
"""
from typing import Optional, Union

import torch
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_fl.dispatch import get_default_manager


class RMSNormFL(RMSNorm):
    """Multi-backend RMSNorm implementation using dispatch system.

    Automatically selects the best available implementation based on:
    - Device availability (CUDA, NPU)
    - Library availability (flag_gems, torch_npu)
    - User-configured policy (environment variables)

    Environment Variables:
        VLLM_FL_PREFER: Preferred backend (default/vendor/reference)
        VLLM_FL_PER_OP: Per-operator control (e.g., "rms_norm=vendor:npu|default")
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        self._dispatch_manager = get_default_manager()

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using dispatch system to select best implementation."""
        # Use dispatch system to resolve and call the best implementation
        return self._dispatch_manager.call(
            "rms_norm",
            x,
            self.weight,
            self.variance_epsilon,
            residual,
        )


__all__ = ["RMSNormFL"]
