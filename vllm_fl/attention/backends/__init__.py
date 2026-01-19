# Copyright (c) 2026 BAAI. All rights reserved.

"""
Attention backends for vllm-plugin-FL.

This module provides attention backend implementations for different hardware platforms.
The dispatch mechanism automatically selects the appropriate backend based on the
available hardware and configuration.

Available backends:
- ascend: Native Ascend NPU attention using torch_npu operators
  - Uses torch_npu.npu_fused_infer_attention_score for prefill
  - Uses torch_npu._npu_paged_attention for decode
  - No dependency on vllm-ascend package
"""

from vllm_fl.attention.backends.ascend import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendAttentionMetadataBuilder,
    AscendMetadata,
    AscendAttentionState,
    AscendMLABackend,
    AttentionMaskBuilder,
    get_attention_mask_builder,
    is_torch_npu_available,
)

__all__ = [
    # Ascend backend
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
