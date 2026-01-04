# Copyright (c) 2025 BAAI. All rights reserved.

"""
Builtin operator implementations for vllm-plugin-FL dispatch system.

This module serves as the entry point for registering all builtin backend
implementations. Each backend is organized in its own module under the
`backends/` directory:

- backends/flagos.py:    DEFAULT implementations (FlagOS/flag_gems)
- backends/cuda.py:      VENDOR implementations (vLLM CUDA kernels)
- backends/npu.py:       VENDOR implementations (torch_npu for Ascend)
- backends/reference.py: REFERENCE implementations (pure PyTorch)

To add a new operator implementation:
1. Add the implementation function to the appropriate backend file
2. Add an OpImpl entry to the backend's _IMPLEMENTATIONS list
3. The implementation will be automatically registered

To add a new backend:
1. Create a new file in backends/ (e.g., backends/my_vendor.py)
2. Implement the `register(registry)` function
3. Import and call it in backends/__init__.py
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import OpRegistry

logger = logging.getLogger(__name__)


def register_builtins(registry: "OpRegistry") -> None:
    """
    Register all builtin operator implementations.

    This function is called by OpManager during initialization.
    It delegates to the backends module which handles registration
    for each backend (FlagOS, CUDA, NPU, Reference).
    """
    from .backends import register_all_backends

    register_all_backends(registry)
    logger.debug("Registered all builtin operator implementations")
