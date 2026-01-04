# Copyright (c) 2025 BAAI. All rights reserved.

"""
Dispatch types for vllm-plugin-FL.

Defines OpImplKind (implementation types) and OpImpl (operator implementation).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, FrozenSet, Optional


class OpImplKind(str, Enum):
    """Operator implementation type."""
    DEFAULT = "default"      # Default implementation (e.g., FlagOS/Triton)
    REFERENCE = "reference"  # Reference implementation (e.g., PyTorch native)
    VENDOR = "vendor"        # Vendor-specific implementation (e.g., CUDA, NPU)


@dataclass(frozen=True)
class OpImpl:
    """
    Represents an operator implementation.

    Attributes:
        op_name: Operator name (e.g., "rms_norm", "silu_and_mul")
        impl_id: Unique implementation ID (e.g., "default.flagos", "vendor.npu")
        kind: Implementation type (DEFAULT, REFERENCE, VENDOR)
        fn: Callable implementation function
        vendor: Vendor name (required for VENDOR kind)
        priority: Priority within same kind (higher = preferred)
        supported_dtypes: Set of supported data types (None = all)
        min_arch: Minimum architecture requirement (e.g., "sm_80", "ascend910b")
        description: Human-readable description
    """
    op_name: str
    impl_id: str
    kind: OpImplKind
    fn: Callable[..., Any]
    vendor: Optional[str] = None
    priority: int = 0
    supported_dtypes: Optional[FrozenSet[str]] = None
    min_arch: Optional[str] = None
    description: str = ""

    def __post_init__(self):
        if self.kind == OpImplKind.VENDOR and not self.vendor:
            raise ValueError(f"VENDOR impl {self.impl_id} must specify vendor name")

    def is_available(self) -> bool:
        """
        Check if this implementation is available at runtime.

        Implementations can attach an `_is_available` callable to the fn
        that returns True/False based on runtime conditions (device, libs, etc).
        """
        avail_fn = getattr(self.fn, "_is_available", None)
        if callable(avail_fn):
            try:
                return bool(avail_fn())
            except Exception:
                return False
        return True

    def __repr__(self) -> str:
        return (
            f"OpImpl(op={self.op_name!r}, id={self.impl_id!r}, "
            f"kind={self.kind.value}, priority={self.priority})"
        )
