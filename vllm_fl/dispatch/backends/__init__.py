# Copyright (c) 2025 BAAI. All rights reserved.

"""
Backend implementations for vllm-plugin-FL dispatch system.

Each backend module provides operator implementations for a specific platform:
- flagos: DEFAULT implementations using FlagOS/flag_gems (cross-platform Triton)
- cuda: VENDOR implementations using vLLM CUDA kernels
- npu: VENDOR implementations using torch_npu (Ascend NPU)
- reference: REFERENCE implementations using pure PyTorch

Each module exports a `register(registry)` function that registers all
implementations for that backend.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..registry import OpRegistry


def register_all_backends(registry: "OpRegistry") -> None:
    """Register all backend implementations."""
    from . import cuda, flagos, npu, reference

    # Register in order: DEFAULT first, then VENDOR, then REFERENCE
    flagos.register(registry)
    cuda.register(registry)
    npu.register(registry)
    reference.register(registry)


__all__ = ["register_all_backends"]
