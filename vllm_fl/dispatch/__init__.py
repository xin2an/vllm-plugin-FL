"""
vllm-plugin-FL Dispatch System

A lightweight operator dispatch framework supporting multiple backends:
- DEFAULT: FlagOS/Triton implementations (default)
- REFERENCE: PyTorch native implementations (fallback)
- VENDOR: Hardware-specific optimizations (CUDA, NPU, etc.)

Usage:
    from vllm_fl.dispatch import OpManager, get_default_manager

    # Use default manager
    manager = get_default_manager()
    fn = manager.resolve("rms_norm")
    result = fn(x, weight, eps=1e-6)

    # Or use call directly
    from vllm_fl.dispatch import call
    result = call("rms_norm", x, weight, eps=1e-6)

Environment Variables:
    VLLM_FL_PREFER: Preferred backend (default/vendor/reference)
    VLLM_FL_STRICT: Strict mode, fail if no match (0/1)
    VLLM_FL_DENY_VENDORS: Comma-separated list of denied vendors
    VLLM_FL_ALLOW_VENDORS: Comma-separated list of allowed vendors
    VLLM_FL_PER_OP: Per-operator order (op=token1|token2;...)
    VLLM_FL_PLUGIN_MODULES: Comma-separated plugin modules to load
"""

from .types import OpImpl, OpImplKind
from .registry import OpRegistry, RegistrySnapshot
from .policy import (
    SelectionPolicy,
    get_policy,
    set_global_policy,
    policy_context,
    with_preference,
    with_denied_vendors,
)
from .manager import (
    OpManager,
    get_default_manager,
    resolve,
    call,
)
from .discovery import discover_plugins

__all__ = [
    # Types
    "OpImpl",
    "OpImplKind",
    # Registry
    "OpRegistry",
    "RegistrySnapshot",
    # Policy
    "SelectionPolicy",
    "get_policy",
    "set_global_policy",
    "policy_context",
    "with_preference",
    "with_denied_vendors",
    # Manager
    "OpManager",
    "get_default_manager",
    "resolve",
    "call",
    # Discovery
    "discover_plugins",
]
