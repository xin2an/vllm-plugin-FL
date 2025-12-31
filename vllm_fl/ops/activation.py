# Copyright (c) 2025 BAAI. All rights reserved.

"""
Multi-backend activation implementations using dispatch system.
"""
import torch
from vllm.model_executor.layers.activation import SiluAndMul

from vllm_fl.dispatch import get_default_manager


class SiluAndMulFL(SiluAndMul):
    """Multi-backend SiluAndMul implementation using dispatch system.

    Automatically selects the best available implementation based on:
    - Device availability (CUDA, NPU)
    - Library availability (flag_gems)
    - User-configured policy (environment variables)

    Environment Variables:
        VLLM_FL_PREFER: Preferred backend (default/vendor/reference)
        VLLM_FL_PER_OP: Per-operator control (e.g., "silu_and_mul=default|reference")
    """

    def __init__(self):
        super().__init__()
        self._dispatch_manager = get_default_manager()

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using dispatch system to select best implementation."""
        return self._dispatch_manager.call("silu_and_mul", x)


__all__ = ["SiluAndMulFL"]
