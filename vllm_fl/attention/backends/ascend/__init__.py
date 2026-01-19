# Copyright (c) 2026 BAAI. All rights reserved.

"""
Ascend NPU attention backend for vllm-plugin-FL.

This package provides native Ascend NPU attention implementation using torch_npu
operators directly, without depending on vllm-ascend package.

Modules:
- attention: Core attention backend classes (AscendAttentionBackend, etc.)
- attention_mask: Attention mask builder and utilities
"""

from vllm_fl.attention.backends.ascend.attention import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
    AscendAttentionState,
    AscendMLABackend,
    is_torch_npu_available,
)
from vllm_fl.attention.backends.ascend.attention_mask import (
    AttentionMaskBuilder,
    get_attention_mask_builder,
)

__all__ = [
    # Attention backend classes
    "AscendAttentionBackend",
    "AscendAttentionBackendImpl",
    "AscendAttentionMetadataBuilder",
    "AscendMetadata",
    "AscendAttentionState",
    "AscendMLABackend",
    # Utilities
    "AttentionMaskBuilder",
    "get_attention_mask_builder",
    "is_torch_npu_available",
]
