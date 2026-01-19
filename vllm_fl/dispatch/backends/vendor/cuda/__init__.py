# Copyright (c) 2025 BAAI. All rights reserved.

"""
CUDA backend for vllm-plugin-FL dispatch.
"""

from .cuda import CudaBackend

__all__ = ["CudaBackend"]
