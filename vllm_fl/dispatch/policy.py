"""
Selection policy for operator dispatch.

Provides SelectionPolicy for controlling how implementations are chosen,
with support for environment variables and context-based overrides.
"""
from __future__ import annotations

import contextvars
import os
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# Environment variable prefix
ENV_PREFIX = "VLLM_FL_"


def _parse_bool(value: str, default: bool = False) -> bool:
    """Parse a boolean from string."""
    if not value:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _parse_csv_set(value: str) -> FrozenSet[str]:
    """Parse comma-separated values into a frozen set."""
    if not value:
        return frozenset()
    items = [x.strip().lower() for x in value.split(",") if x.strip()]
    return frozenset(items)


def _parse_per_op(value: str) -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    """
    Parse per-op order specification.

    Format: "op1=token1|token2;op2=token3|token4"
    Example: "rms_norm=vendor:npu|default;silu=default|reference"

    Returns:
        Tuple of (op_name, (tokens...)) pairs
    """
    if not value:
        return ()

    result: List[Tuple[str, Tuple[str, ...]]] = []
    parts = [p.strip() for p in value.split(";") if p.strip()]

    for part in parts:
        if "=" not in part:
            continue
        op, order = part.split("=", 1)
        op = op.strip()
        order_list = tuple(x.strip() for x in order.split("|") if x.strip())
        if op and order_list:
            result.append((op, order_list))

    return tuple(result)


@dataclass(frozen=True)
class SelectionPolicy:
    """
    Policy for selecting operator implementations.

    Attributes:
        prefer: Preferred implementation type ("default", "vendor", "reference")
        strict: If True, fail when no implementation matches (no fallback)
        per_op_order: Per-operator selection order as ((op_name, (tokens,)),...)
        deny_vendors: Set of vendor names to exclude
        allow_vendors: If set, only allow these vendors (whitelist)
    """
    prefer: str = "default"
    strict: bool = False
    per_op_order: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    deny_vendors: FrozenSet[str] = field(default_factory=frozenset)
    allow_vendors: Optional[FrozenSet[str]] = None

    def get_default_order(self) -> List[str]:
        """
        Get the default selection order based on preference.

        Returns:
            List of tokens like ["vendor", "default", "reference"]
        """
        if self.prefer == "vendor":
            return ["vendor", "default", "reference"]
        elif self.prefer == "reference":
            return ["reference", "default", "vendor"]
        else:  # default or flagos
            return ["default", "vendor", "reference"]

    def get_order_for_op(self, op_name: str) -> List[str]:
        """Get the selection order for a specific operator."""
        for op, order in self.per_op_order:
            if op == op_name:
                return list(order)
        return self.get_default_order()

    def fingerprint(self) -> str:
        """
        Generate a unique fingerprint for this policy.

        Used for cache keys.
        """
        allow = ",".join(sorted(self.allow_vendors)) if self.allow_vendors else ""
        deny = ",".join(sorted(self.deny_vendors))
        per_op = ";".join(
            f"{op}={'|'.join(order)}"
            for op, order in sorted(self.per_op_order)
        )
        return f"pref={self.prefer};strict={int(self.strict)};allow={allow};deny={deny};per={per_op}"

    @classmethod
    def from_env(cls) -> "SelectionPolicy":
        """Create a policy from environment variables."""
        prefer = os.getenv(f"{ENV_PREFIX}PREFER", "default").strip().lower()
        strict = _parse_bool(os.getenv(f"{ENV_PREFIX}STRICT", ""))
        deny = _parse_csv_set(os.getenv(f"{ENV_PREFIX}DENY_VENDORS", ""))
        allow_str = os.getenv(f"{ENV_PREFIX}ALLOW_VENDORS", "").strip()
        allow = _parse_csv_set(allow_str) if allow_str else None
        per_op = _parse_per_op(os.getenv(f"{ENV_PREFIX}PER_OP", ""))

        return cls(
            prefer=prefer,
            strict=strict,
            per_op_order=per_op,
            deny_vendors=deny,
            allow_vendors=allow,
        )

    @classmethod
    def from_dict(cls, **kwargs) -> "SelectionPolicy":
        """Create a policy from keyword arguments."""
        # Convert mutable types to immutable
        if "deny_vendors" in kwargs and isinstance(kwargs["deny_vendors"], set):
            kwargs["deny_vendors"] = frozenset(kwargs["deny_vendors"])
        if "allow_vendors" in kwargs and isinstance(kwargs["allow_vendors"], set):
            kwargs["allow_vendors"] = frozenset(kwargs["allow_vendors"])
        if "per_op_order" in kwargs and isinstance(kwargs["per_op_order"], dict):
            kwargs["per_op_order"] = tuple(
                (k, tuple(v)) for k, v in kwargs["per_op_order"].items()
            )
        return cls(**kwargs)


# Context variable for thread/async-local policy override
_policy_ctx: contextvars.ContextVar[Optional[SelectionPolicy]] = contextvars.ContextVar(
    "vllm_fl_dispatch_policy",
    default=None
)

# Global default policy (lazily initialized from env)
_global_policy: Optional[SelectionPolicy] = None
_global_policy_initialized = False


def get_policy() -> SelectionPolicy:
    """
    Get the current selection policy.

    Priority:
    1. Context-local override (via policy_context or set_context_policy)
    2. Global policy (via set_global_policy or from environment)
    """
    # Check context-local first
    ctx_policy = _policy_ctx.get()
    if ctx_policy is not None:
        return ctx_policy

    # Fall back to global
    global _global_policy, _global_policy_initialized
    if not _global_policy_initialized:
        _global_policy = SelectionPolicy.from_env()
        _global_policy_initialized = True

    return _global_policy  # type: ignore


def set_global_policy(policy: SelectionPolicy) -> None:
    """Set the global default policy."""
    global _global_policy, _global_policy_initialized
    _global_policy = policy
    _global_policy_initialized = True


def set_context_policy(policy: Optional[SelectionPolicy]) -> contextvars.Token:
    """
    Set a context-local policy override.

    Returns a token that can be used to reset the policy.
    """
    return _policy_ctx.set(policy)


def reset_context_policy(token: contextvars.Token) -> None:
    """Reset context policy using the token from set_context_policy."""
    _policy_ctx.reset(token)


class policy_context:
    """
    Context manager for temporarily overriding the selection policy.

    Example:
        with policy_context(SelectionPolicy(prefer="vendor")):
            result = op_manager.call("rms_norm", x, w)
    """

    def __init__(self, policy: SelectionPolicy):
        self._policy = policy
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "policy_context":
        self._token = set_context_policy(self._policy)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._token is not None:
            reset_context_policy(self._token)


class with_preference:
    """
    Context manager for temporarily changing preference.

    Example:
        with with_preference("vendor"):
            result = op_manager.call("rms_norm", x, w)
    """

    def __init__(self, prefer: str):
        self._prefer = prefer
        self._ctx: Optional[policy_context] = None

    def __enter__(self) -> "with_preference":
        current = get_policy()
        new_policy = SelectionPolicy(
            prefer=self._prefer,
            strict=current.strict,
            per_op_order=current.per_op_order,
            deny_vendors=current.deny_vendors,
            allow_vendors=current.allow_vendors,
        )
        self._ctx = policy_context(new_policy)
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._ctx:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)


class with_denied_vendors:
    """
    Context manager for temporarily denying specific vendors.

    Example:
        with with_denied_vendors({"npu"}):
            result = op_manager.call("rms_norm", x, w)
    """

    def __init__(self, vendors: Set[str]):
        self._vendors = frozenset(vendors)
        self._ctx: Optional[policy_context] = None

    def __enter__(self) -> "with_denied_vendors":
        current = get_policy()
        new_policy = SelectionPolicy(
            prefer=current.prefer,
            strict=current.strict,
            per_op_order=current.per_op_order,
            deny_vendors=current.deny_vendors | self._vendors,
            allow_vendors=current.allow_vendors,
        )
        self._ctx = policy_context(new_policy)
        self._ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._ctx:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)
