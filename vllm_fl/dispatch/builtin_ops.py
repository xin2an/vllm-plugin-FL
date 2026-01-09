# Copyright (c) 2025 BAAI. All rights reserved.

"""
Built-in operator implementations registration.

This module registers DEFAULT (FlagGems) and REFERENCE (PyTorch) implementations
for all supported operators by calling register_builtins from each backend.
"""

from __future__ import annotations

from .registry import OpRegistry
from .logger_manager import get_logger

logger = get_logger()


def register_builtins(registry: OpRegistry) -> None:
    """
    Register all built-in operator implementations.

    This function registers:
    - DEFAULT implementations (FlagGems)
    - REFERENCE implementations (PyTorch)
    - VENDOR implementations (if available)

    Args:
        registry: Registry to register into
    """
    # Register FlagGems (DEFAULT) implementations
    try:
        from .backends.flaggems.register_ops import register_builtins as register_flaggems

        register_flaggems(registry)
        logger.debug("Registered FlagGems operators")
    except Exception as e:
        logger.warning(f"Failed to register FlagGems operators: {e}")

    # Register PyTorch (REFERENCE) implementations
    try:
        from .backends.reference.register_ops import register_builtins as register_reference

        register_reference(registry)
        logger.debug("Registered Reference operators")
    except Exception as e:
        logger.warning(f"Failed to register Reference operators: {e}")

    # Register VENDOR implementations (if available)
    # Add vendor backends here as they become available
    # Example:
    # try:
    #     from .backends.vendor.cuda.register_ops import register_builtins as register_cuda
    #     register_cuda(registry)
    #     logger.debug("Registered CUDA operators")
    # except Exception as e:
    #     # CUDA may not be available, this is expected
    #     logger.debug(f"CUDA operators not available: {e}")
    #     pass
