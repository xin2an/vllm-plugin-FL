# Copyright (c) 2025 BAAI. All rights reserved.

"""
Operator dispatch manager for vllm-plugin-FL.

Provides OpManager for lazy initialization, dispatch caching, and multi-process safety.
"""
from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .discovery import discover_plugins
from .policy import SelectionPolicy, get_policy
from .registry import OpRegistry
from .types import OpImpl, OpImplKind

logger = logging.getLogger(__name__)


@dataclass
class _RuntimeState:
    """Internal runtime state for OpManager."""
    init_pid: int = -1
    initialized: bool = False
    policy_epoch: int = 0


class OpManager:
    """
    Operator dispatch manager.

    Handles:
    - Lazy initialization (first resolve triggers init)
    - Multi-process safety (fork detection via PID)
    - Dispatch caching with policy-aware invalidation
    - Fallback logic based on SelectionPolicy
    """

    def __init__(self, registry: Optional[OpRegistry] = None) -> None:
        self._lock = threading.RLock()
        self._registry = registry or OpRegistry()
        self._state = _RuntimeState()

        # Cache: (op_name, policy_fingerprint, epoch) -> Callable
        self._dispatch_cache: Dict[Tuple[str, str, int], Callable] = {}

        # Try to register fork handler for multi-process safety
        try:
            os.register_at_fork(after_in_child=self._reset_after_fork)
        except (AttributeError, NotImplementedError):
            # Not available on all platforms
            pass

    @property
    def registry(self) -> OpRegistry:
        """Access the underlying registry."""
        return self._registry

    def _reset_after_fork(self) -> None:
        """Reset state after fork (called in child process)."""
        with self._lock:
            self._state.initialized = False
            self._state.init_pid = -1
            self._state.policy_epoch += 1
            self._dispatch_cache.clear()
            logger.debug("OpManager reset after fork")

    def bump_policy_epoch(self) -> None:
        """Invalidate dispatch cache (call when policy changes)."""
        with self._lock:
            self._state.policy_epoch += 1
            self._dispatch_cache.clear()

    def ensure_initialized(self) -> None:
        """
        Ensure the manager is initialized.

        Idempotent - safe to call multiple times.
        Handles fork detection via PID check.
        """
        with self._lock:
            pid = os.getpid()

            # Check if already initialized in this process
            if self._state.initialized and self._state.init_pid == pid:
                return

            # Reset for new process
            self._state.initialized = True
            self._state.init_pid = pid
            self._state.policy_epoch += 1
            self._dispatch_cache.clear()

            # Register builtin implementations
            self._register_builtins()

            # Discover external plugins
            discover_plugins(self._registry)

            logger.debug(
                f"OpManager initialized (pid={pid}, "
                f"registry has {len(self._registry)} impls)"
            )

    def _register_builtins(self) -> None:
        """Register builtin implementations."""
        # Import here to avoid circular imports and keep init lightweight
        try:
            from . import builtin_ops
            builtin_ops.register_builtins(self._registry)
        except ImportError as e:
            logger.warning(f"Failed to import builtin_ops: {e}")
        except Exception as e:
            logger.warning(f"Failed to register builtin ops: {e}")

    def _matches_vendor_filters(
        self,
        impl: OpImpl,
        policy: SelectionPolicy
    ) -> bool:
        """Check if implementation passes vendor filters."""
        if impl.kind != OpImplKind.VENDOR:
            return True

        if impl.vendor is None:
            return False

        vendor_lower = impl.vendor.lower()

        # Check deny list
        if vendor_lower in policy.deny_vendors:
            return False

        # Check allow list (if set)
        if policy.allow_vendors is not None:
            if vendor_lower not in policy.allow_vendors:
                return False

        return True

    def _token_matches(self, impl: OpImpl, token: str) -> bool:
        """
        Check if an implementation matches a selection token.

        Token formats:
        - "default": matches DEFAULT kind
        - "reference": matches REFERENCE kind
        - "vendor": matches any VENDOR kind
        - "vendor:name": matches VENDOR with specific vendor name
        - "impl:id": matches specific impl_id
        """
        token = token.lower()

        if token.startswith("impl:"):
            return impl.impl_id == token[5:]

        if token == "default":
            return impl.kind == OpImplKind.DEFAULT

        if token == "reference":
            return impl.kind == OpImplKind.REFERENCE

        if token == "vendor":
            return impl.kind == OpImplKind.VENDOR

        if token.startswith("vendor:"):
            vendor_name = token[7:]
            return (
                impl.kind == OpImplKind.VENDOR and
                impl.vendor is not None and
                impl.vendor.lower() == vendor_name
            )

        return False

    def resolve(self, op_name: str) -> Callable:
        """
        Resolve the best implementation for an operator.

        Args:
            op_name: Operator name (e.g., "rms_norm")

        Returns:
            The selected implementation function

        Raises:
            RuntimeError: If no implementation is available
        """
        self.ensure_initialized()

        policy = get_policy()
        policy_fp = policy.fingerprint()
        epoch = self._state.policy_epoch

        # Check cache
        cache_key = (op_name, policy_fp, epoch)
        cached = self._dispatch_cache.get(cache_key)
        if cached is not None:
            return cached

        # Get candidates from registry snapshot
        snap = self._registry.snapshot()
        candidates = list(snap.get_impls(op_name))

        if not candidates:
            raise RuntimeError(f"No implementations registered for op={op_name}")

        # Filter by vendor policy
        candidates = [
            c for c in candidates
            if self._matches_vendor_filters(c, policy)
        ]

        # Filter by availability
        available: List[OpImpl] = []
        for c in candidates:
            try:
                if c.is_available():
                    available.append(c)
                else:
                    logger.debug(f"Impl {c.impl_id} not available for op={op_name}")
            except Exception as e:
                logger.debug(f"Impl {c.impl_id} availability check failed: {e}")

        candidates = available

        if not candidates:
            if policy.strict:
                raise RuntimeError(
                    f"No implementation available for op={op_name} "
                    f"under strict policy"
                )
            raise RuntimeError(
                f"No implementation available for op={op_name} "
                f"(all candidates unavailable or filtered)"
            )

        # Select based on policy order
        order = policy.get_order_for_op(op_name)
        chosen: Optional[OpImpl] = None

        for token in order:
            matches = [c for c in candidates if self._token_matches(c, token)]
            if not matches:
                continue

            # Sort by priority (descending), then by impl_id (for stability)
            matches.sort(key=lambda x: (-x.priority, x.impl_id))
            chosen = matches[0]
            break

        # Fallback: pick highest priority overall
        if chosen is None:
            candidates.sort(key=lambda x: (-x.priority, x.impl_id))
            chosen = candidates[0]

        # Cache and return
        self._dispatch_cache[cache_key] = chosen.fn
        logger.debug(
            f"Resolved op={op_name} -> {chosen.impl_id} "
            f"(kind={chosen.kind.value}, priority={chosen.priority})"
        )

        return chosen.fn

    def resolve_candidates(self, op_name: str) -> List[OpImpl]:
        """
        Get all available implementations for an operator.

        Returns implementations sorted by selection priority.
        """
        self.ensure_initialized()

        policy = get_policy()
        snap = self._registry.snapshot()
        candidates = list(snap.get_impls(op_name))

        # Filter by vendor and availability
        result: List[OpImpl] = []
        for c in candidates:
            if not self._matches_vendor_filters(c, policy):
                continue
            try:
                if c.is_available():
                    result.append(c)
            except Exception:
                pass

        # Sort by priority
        result.sort(key=lambda x: (-x.priority, x.impl_id))
        return result

    def call(self, op_name: str, *args, **kwargs) -> Any:
        """
        Resolve and call an operator.

        Args:
            op_name: Operator name
            *args, **kwargs: Arguments to pass to the implementation

        Returns:
            Result from the implementation
        """
        fn = self.resolve(op_name)
        return fn(*args, **kwargs)

    def has_op(self, op_name: str) -> bool:
        """Check if an operator has any registered implementations."""
        self.ensure_initialized()
        return self._registry.has_op(op_name)

    def get_op_names(self) -> List[str]:
        """Get all registered operator names."""
        self.ensure_initialized()
        return self._registry.snapshot().get_op_names()

    def __repr__(self) -> str:
        return (
            f"OpManager(initialized={self._state.initialized}, "
            f"ops={len(self._registry)}, "
            f"cache_size={len(self._dispatch_cache)})"
        )


# Default global manager instance
_default_manager: Optional[OpManager] = None
_default_manager_lock = threading.Lock()


def get_default_manager() -> OpManager:
    """Get the default global OpManager instance."""
    global _default_manager

    if _default_manager is None:
        with _default_manager_lock:
            if _default_manager is None:
                _default_manager = OpManager()

    return _default_manager


def resolve(op_name: str) -> Callable:
    """Resolve an operator using the default manager."""
    return get_default_manager().resolve(op_name)


def call(op_name: str, *args, **kwargs) -> Any:
    """Call an operator using the default manager."""
    return get_default_manager().call(op_name, *args, **kwargs)
