"""
Thread-safe operator registry for vllm-plugin-FL.

Provides OpRegistry for storing and querying operator implementations.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .types import OpImpl, OpImplKind

logger = logging.getLogger(__name__)


@dataclass
class RegistrySnapshot:
    """
    Immutable snapshot of the registry state.

    Used for thread-safe reads without holding locks during dispatch.
    """
    impls_by_op: Dict[str, List[OpImpl]] = field(default_factory=dict)

    def get_impls(self, op_name: str) -> List[OpImpl]:
        """Get all implementations for an operator."""
        return self.impls_by_op.get(op_name, [])

    def get_op_names(self) -> List[str]:
        """Get all registered operator names."""
        return list(self.impls_by_op.keys())


class OpRegistry:
    """
    Thread-safe registry for operator implementations.

    Supports registration, querying, and snapshot for dispatch.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # op_name -> {impl_id -> OpImpl}
        self._impls_by_op: Dict[str, Dict[str, OpImpl]] = {}
        self._version = 0  # Incremented on each modification

    @property
    def version(self) -> int:
        """Current registry version (increments on modifications)."""
        with self._lock:
            return self._version

    def register_impl(self, impl: OpImpl, skip_duplicate: bool = False) -> None:
        """
        Register an operator implementation.

        Args:
            impl: The OpImpl to register
            skip_duplicate: If True, silently skip duplicate impl_id

        Raises:
            ValueError: If impl_id already exists (and skip_duplicate=False)
        """
        with self._lock:
            by_id = self._impls_by_op.setdefault(impl.op_name, {})

            if impl.impl_id in by_id:
                if skip_duplicate:
                    logger.debug(
                        f"Skipping duplicate impl: {impl.impl_id} for op={impl.op_name}"
                    )
                    return
                raise ValueError(
                    f"Duplicate impl_id: {impl.impl_id} for op={impl.op_name}"
                )

            by_id[impl.impl_id] = impl
            self._version += 1

            logger.debug(
                f"Registered {impl.kind.value} impl: {impl.impl_id} "
                f"for op={impl.op_name} (priority={impl.priority})"
            )

    def register_many(
        self,
        impls: Sequence[OpImpl],
        skip_duplicate: bool = False
    ) -> None:
        """Register multiple implementations."""
        for impl in impls:
            self.register_impl(impl, skip_duplicate=skip_duplicate)

    def unregister_impl(self, op_name: str, impl_id: str) -> bool:
        """
        Unregister an implementation.

        Returns:
            True if found and removed, False otherwise
        """
        with self._lock:
            by_id = self._impls_by_op.get(op_name)
            if by_id and impl_id in by_id:
                del by_id[impl_id]
                if not by_id:
                    del self._impls_by_op[op_name]
                self._version += 1
                return True
            return False

    def get_impl(self, op_name: str, impl_id: str) -> Optional[OpImpl]:
        """Get a specific implementation by op_name and impl_id."""
        with self._lock:
            by_id = self._impls_by_op.get(op_name, {})
            return by_id.get(impl_id)

    def get_impls_by_op(self, op_name: str) -> List[OpImpl]:
        """Get all implementations for an operator."""
        with self._lock:
            by_id = self._impls_by_op.get(op_name, {})
            return list(by_id.values())

    def get_impls_by_kind(
        self,
        op_name: str,
        kind: OpImplKind
    ) -> List[OpImpl]:
        """Get implementations of a specific kind for an operator."""
        with self._lock:
            by_id = self._impls_by_op.get(op_name, {})
            return [impl for impl in by_id.values() if impl.kind == kind]

    def has_op(self, op_name: str) -> bool:
        """Check if an operator has any implementations."""
        with self._lock:
            return op_name in self._impls_by_op

    def snapshot(self) -> RegistrySnapshot:
        """
        Create an immutable snapshot of the current registry state.

        Use for dispatch operations to avoid holding locks.
        """
        with self._lock:
            impls_by_op = {
                op: list(by_id.values())
                for op, by_id in self._impls_by_op.items()
            }
        return RegistrySnapshot(impls_by_op=impls_by_op)

    def clear(self) -> None:
        """Clear all registered implementations."""
        with self._lock:
            self._impls_by_op.clear()
            self._version += 1

    def __len__(self) -> int:
        """Total number of registered implementations."""
        with self._lock:
            return sum(len(by_id) for by_id in self._impls_by_op.values())

    def __repr__(self) -> str:
        with self._lock:
            ops = list(self._impls_by_op.keys())
        return f"OpRegistry(ops={ops}, total_impls={len(self)})"
