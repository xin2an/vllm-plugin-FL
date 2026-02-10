# Copyright (c) 2026 BAAI. All rights reserved.

"""
Automatic operator registration utilities.

This module provides utilities to automatically register operator implementations
from Backend classes without manually listing each operator in register_ops.py.
"""

from __future__ import annotations

import functools
import inspect
from typing import TYPE_CHECKING, List

from .types import OpImpl, BackendImplKind, BackendPriority

if TYPE_CHECKING:
    from .backends.base import Backend
    from .registry import OpRegistry


def _bind_is_available(fn, is_available_fn):
    """Wrap a function and bind _is_available attribute for OpImpl.is_available() check."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    wrapper._is_available = is_available_fn
    return wrapper


def _is_operator_method(name: str, method) -> bool:
    """
    Check if a method is an operator implementation.

    Excludes:
    - Private methods (starting with _)
    - Abstract methods
    - Properties
    - Class methods
    - Static methods
    - Backend infrastructure methods (is_available, name, vendor)

    Args:
        name: Method name
        method: Method object

    Returns:
        True if this is an operator implementation method
    """
    # Skip private methods
    if name.startswith("_"):
        return False

    # Skip infrastructure methods
    if name in ("is_available", "name", "vendor"):
        return False

    # Skip if not callable
    if not callable(method):
        return False

    # Skip properties, classmethods, staticmethods
    if isinstance(inspect.getattr_static(method.__class__, name, None),
                  (property, classmethod, staticmethod)):
        return False

    return True


def auto_register_backend(
    backend: Backend,
    registry: OpRegistry,
    kind: BackendImplKind,
    priority: int = None,
    vendor: str = None,
) -> List[OpImpl]:
    """
    Automatically register all operator implementations from a Backend instance.

    This function inspects the backend class and automatically creates OpImpl
    registrations for all public methods (excluding infrastructure methods like
    is_available, name, vendor).

    Args:
        backend: Backend instance to register
        registry: Registry to register into
        kind: Backend implementation kind (DEFAULT, REFERENCE, VENDOR)
        priority: Priority for selection (defaults based on kind)
        vendor: Vendor name (required if kind is VENDOR)

    Returns:
        List of registered OpImpl instances

    Example:
        ```python
        from vllm_fl.dispatch.auto_register import auto_register_backend
        from vllm_fl.dispatch.types import BackendImplKind, BackendPriority

        def register_builtins(registry):
            from .cuda import CudaBackend

            backend = CudaBackend()
            auto_register_backend(
                backend=backend,
                registry=registry,
                kind=BackendImplKind.VENDOR,
                priority=BackendPriority.VENDOR,
                vendor="cuda",
            )
        ```
    """
    # Set default priority based on kind
    if priority is None:
        if kind == BackendImplKind.DEFAULT:
            priority = BackendPriority.DEFAULT
        elif kind == BackendImplKind.VENDOR:
            priority = BackendPriority.VENDOR
        elif kind == BackendImplKind.REFERENCE:
            priority = BackendPriority.REFERENCE
        else:
            priority = 0

    # Validate vendor for VENDOR kind
    if kind == BackendImplKind.VENDOR and not vendor:
        # Try to get vendor from backend
        vendor = backend.vendor
        if not vendor:
            raise ValueError(
                f"Backend kind is VENDOR but no vendor name provided. "
                f"Either pass vendor parameter or implement backend.vendor property."
            )

    # Get backend name for impl_id
    backend_name = backend.name

    # Collect all operator methods
    impls = []
    is_avail = backend.is_available

    # Inspect backend class for operator methods
    for name in dir(backend):
        # Skip private and infrastructure methods
        if name.startswith("_") or name in ("is_available", "name", "vendor"):
            continue

        try:
            method = getattr(backend, name)
        except Exception:
            continue

        # Check if this is an operator method
        if not _is_operator_method(name, method):
            continue

        # Create impl_id based on kind
        if kind == BackendImplKind.VENDOR:
            impl_id = f"vendor.{vendor}"
        elif kind == BackendImplKind.DEFAULT:
            impl_id = f"default.{backend_name}"
        elif kind == BackendImplKind.REFERENCE:
            impl_id = f"reference.{backend_name}"
        else:
            impl_id = f"{backend_name}.{name}"

        # Create OpImpl
        impl = OpImpl(
            op_name=name,
            impl_id=impl_id,
            kind=kind,
            fn=_bind_is_available(method, is_avail),
            vendor=vendor,
            priority=priority,
        )

        impls.append(impl)

    # Register all implementations
    registry.register_many(impls)

    return impls
