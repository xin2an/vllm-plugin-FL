# Copyright (c) 2025 BAAI. All rights reserved.

"""
vllm-plugin-FL operators with multi-backend dispatch support.

All operators automatically select the best available implementation
based on device, library availability, and user configuration.

Environment Variables:
    VLLM_FL_PREFER: Preferred backend (default/vendor/reference)
    VLLM_FL_STRICT: Strict mode, fail if no match (0/1)
    VLLM_FL_PER_OP: Per-operator order (e.g., "rms_norm=vendor:npu|default")
    VLLM_FL_PLUGIN_MODULES: Load additional vendor implementations
"""
from .activation import SiluAndMulFL
from .layernorm import RMSNormFL
from .rotary_embedding import RotaryEmbeddingFL
from .custom_ops import register_oot_ops

__all__ = [
    "SiluAndMulFL",
    "RMSNormFL",
    "RotaryEmbeddingFL",
    "register_oot_ops",
]
