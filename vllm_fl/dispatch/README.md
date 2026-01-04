# vllm-plugin-FL Dispatch System

A lightweight operator dispatch framework for managing multiple backend implementations.

## Overview

The dispatch system provides:

- **Multi-backend support**: DEFAULT (FlagOS), VENDOR (CUDA/NPU), REFERENCE (PyTorch)
- **Automatic selection**: Best implementation chosen based on availability and policy
- **Plugin discovery**: External vendors can register implementations via entry points or environment variables
- **Policy control**: Fine-grained control over implementation selection
- **Production-ready**: Thread-safe, fork-safe, with dispatch caching

## Implementation Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│  DEFAULT (FlagOS/flag_gems)                                     │
│  - Cross-platform Triton-based implementations                  │
│  - Works on both CUDA and NPU                                   │
│  - Default choice, highest priority                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓ fallback
┌─────────────────────────────────────────────────────────────────┐
│  VENDOR (Hardware-specific)                                     │
│  ┌─────────────────────────┐ ┌─────────────────────────┐       │
│  │  vendor.cuda            │ │  vendor.npu             │       │
│  │  vLLM CUDA kernels      │ │  torch_npu ops          │       │
│  └─────────────────────────┘ └─────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              ↓ fallback
┌─────────────────────────────────────────────────────────────────┐
│  REFERENCE (PyTorch)                                            │
│  - Pure PyTorch implementations                                 │
│  - Always available, lowest priority                            │
│  - Used as final fallback                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Registered Operators

| Operator | DEFAULT | VENDOR (CUDA) | VENDOR (NPU) | REFERENCE |
|----------|---------|---------------|--------------|-----------|
| `rms_norm` | flag_gems | vLLM _C | torch_npu | PyTorch |
| `silu_and_mul` | flag_gems | vLLM _C | - | PyTorch |
| `rotary_embedding` | flag_gems | vLLM _C | - | PyTorch |

## Quick Start

### Basic Usage

```python
from vllm_fl.dispatch import get_default_manager, call

# Method 1: Get manager and resolve
manager = get_default_manager()
fn = manager.resolve("rms_norm")
result = fn(x, weight, eps=1e-6)

# Method 2: Direct call
result = call("rms_norm", x, weight, eps=1e-6)

# Method 3: Convenience API
from vllm_fl.dispatch.ops import rms_norm
result = rms_norm(x, weight, eps=1e-6)
```

### Check Available Implementations

```python
from vllm_fl.dispatch import get_default_manager

manager = get_default_manager()

# List all registered operators
print(manager.get_op_names())
# ['rms_norm', 'silu_and_mul', 'rotary_embedding']

# Get all available implementations for an operator
for impl in manager.resolve_candidates("rms_norm"):
    print(f"{impl.impl_id}: {impl.description}")
# default.flagos: RMSNorm using flag_gems (FlagOS)
# vendor.npu: RMSNorm using torch_npu (Ascend)
# vendor.cuda: RMSNorm using vLLM CUDA kernels
# reference.torch: RMSNorm reference implementation (PyTorch)
```

## Policy Control

### Use Cases

#### 1. Use FlagOS globally (default behavior)

```bash
# No configuration needed, this is the default
python my_script.py
```

Or explicitly:

```bash
export VLLM_FL_PREFER=default
```

#### 2. Use vendor implementations globally

```bash
export VLLM_FL_PREFER=vendor
```

```python
from vllm_fl.dispatch import SelectionPolicy, set_global_policy

set_global_policy(SelectionPolicy(prefer="vendor"))
```

#### 3. Use specific vendor (e.g., only CUDA)

```bash
export VLLM_FL_PREFER=vendor
export VLLM_FL_ALLOW_VENDORS=cuda
```

```python
set_global_policy(SelectionPolicy(
    prefer="vendor",
    allow_vendors=frozenset({"cuda"}),
))
```

#### 4. Use FlagOS globally, but specific operator uses vendor

This is useful when FlagOS works well for most operators, but you want to use
hardware-specific implementation for certain operators.

```bash
# Global: FlagOS, but rms_norm uses NPU vendor
export VLLM_FL_PREFER=default
export VLLM_FL_PER_OP="rms_norm=vendor:npu|default|reference"
```

```python
set_global_policy(SelectionPolicy(
    prefer="default",  # Global: use FlagOS
    per_op_order=(
        # rms_norm: try NPU vendor first, then FlagOS, then PyTorch
        ("rms_norm", ("vendor:npu", "default", "reference")),
    ),
))
```

#### 5. Multiple operators with custom order

```bash
export VLLM_FL_PER_OP="rms_norm=vendor:npu|default;silu_and_mul=vendor:cuda|default"
```

```python
set_global_policy(SelectionPolicy(
    prefer="default",
    per_op_order=(
        ("rms_norm", ("vendor:npu", "default", "reference")),
        ("silu_and_mul", ("vendor:cuda", "default", "reference")),
    ),
))
```

#### 6. Temporary override (context manager)

```python
from vllm_fl.dispatch import (
    policy_context,
    with_preference,
    with_denied_vendors,
    SelectionPolicy,
    call,
)

# Temporarily use reference implementation
with with_preference("reference"):
    result = call("rms_norm", x, weight)

# Temporarily deny NPU vendor
with with_denied_vendors({"npu"}):
    result = call("rms_norm", x, weight)

# Fully custom temporary policy
with policy_context(SelectionPolicy(
    prefer="vendor",
    allow_vendors=frozenset({"cuda"}),
)):
    result = call("rms_norm", x, weight)
```

## Environment Variables

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `VLLM_FL_PREFER` | Preferred backend type | `default` | `vendor`, `reference` |
| `VLLM_FL_STRICT` | Fail if no implementation matches | `0` | `1` |
| `VLLM_FL_DENY_VENDORS` | Vendors to exclude (comma-separated) | (empty) | `npu,cuda` |
| `VLLM_FL_ALLOW_VENDORS` | Only allow these vendors | (empty=all) | `cuda` |
| `VLLM_FL_PER_OP` | Per-operator selection order | (empty) | See below |
| `VLLM_FL_PLUGIN_MODULES` | Plugin modules to load | (empty) | `my_vendor.ops` |

### VLLM_FL_PER_OP Format

```
op1=token1|token2|token3;op2=token4|token5
```

Examples:
```bash
# rms_norm prefers NPU, others use default
VLLM_FL_PER_OP="rms_norm=vendor:npu|default|reference"

# Multiple operators
VLLM_FL_PER_OP="rms_norm=vendor:npu|default;silu_and_mul=vendor:cuda|default"
```

## Token Types

| Token | Matches | Example |
|-------|---------|---------|
| `default` | DEFAULT implementations (FlagOS) | `default` |
| `vendor` | Any VENDOR implementation | `vendor` |
| `vendor:name` | Specific vendor | `vendor:cuda`, `vendor:npu` |
| `reference` | REFERENCE implementations (PyTorch) | `reference` |
| `impl:id` | Exact implementation ID | `impl:vendor.cuda` |

## Selection Algorithm

1. Get all registered implementations for the operator
2. Filter by vendor policy (`allow_vendors`, `deny_vendors`)
3. Filter by availability (`is_available()` check)
4. Apply selection order:
   - Process tokens in order (e.g., `["vendor:npu", "default", "reference"]`)
   - Within each token, sort by priority (higher first)
   - Return first match
5. Fallback: if no token matches, use highest priority implementation

## Creating a Vendor Plugin

### Method 1: Entry Points (Recommended)

```toml
# pyproject.toml
[project.entry-points."vllm_fl.dispatch.plugins"]
my_vendor = "my_vendor_pkg:vllm_fl_dispatch_register"
```

```python
# my_vendor_pkg/__init__.py
from vllm_fl.dispatch import OpImpl, OpImplKind

def _my_rms_norm(x, weight, eps=1e-6, residual=None):
    # Your optimized implementation
    ...

_my_rms_norm._is_available = lambda: check_my_vendor_available()

def vllm_fl_dispatch_register(registry):
    registry.register_impl(OpImpl(
        op_name="rms_norm",
        impl_id="vendor.my_vendor",
        kind=OpImplKind.VENDOR,
        vendor="my_vendor",
        fn=_my_rms_norm,
        priority=200,  # Higher = preferred within same kind
        description="RMSNorm using MyVendor library",
    ))
```

### Method 2: Environment Variable

```bash
VLLM_FL_PLUGIN_MODULES=my_vendor_pkg python my_script.py
```

### Availability Check

Attach `_is_available` to your function to control when it's selectable:

```python
def _my_impl(x, weight):
    import my_library
    return my_library.rms_norm(x, weight)

def _check_available():
    try:
        import my_library
        return my_library.is_device_available()
    except ImportError:
        return False

_my_impl._is_available = _check_available
```

## API Reference

### OpImpl

```python
@dataclass(frozen=True)
class OpImpl:
    op_name: str              # Operator name (e.g., "rms_norm")
    impl_id: str              # Unique ID (e.g., "vendor.cuda")
    kind: OpImplKind          # DEFAULT, REFERENCE, or VENDOR
    fn: Callable              # Implementation function
    vendor: Optional[str]     # Vendor name (required for VENDOR kind)
    priority: int = 0         # Higher = preferred within same kind
    description: str = ""     # Human-readable description

    def is_available(self) -> bool:
        # Checks fn._is_available() if defined, else True
```

### OpManager

```python
class OpManager:
    def resolve(self, op_name: str) -> Callable:
        """Get best implementation for operator."""

    def call(self, op_name: str, *args, **kwargs) -> Any:
        """Resolve and call operator."""

    def resolve_candidates(self, op_name: str) -> List[OpImpl]:
        """Get all available implementations, sorted by priority."""

    def has_op(self, op_name: str) -> bool:
        """Check if operator has any implementations."""

    def get_op_names(self) -> List[str]:
        """Get all registered operator names."""
```

### SelectionPolicy

```python
@dataclass(frozen=True)
class SelectionPolicy:
    prefer: str = "default"                    # default, vendor, or reference
    strict: bool = False                       # Fail if no match (no fallback)
    per_op_order: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    deny_vendors: FrozenSet[str] = frozenset() # Vendors to exclude
    allow_vendors: Optional[FrozenSet[str]] = None  # Vendor whitelist

    @classmethod
    def from_env(cls) -> SelectionPolicy:
        """Create policy from environment variables."""

    def get_order_for_op(self, op_name: str) -> List[str]:
        """Get selection order for specific operator."""
```

### Context Managers

```python
# Temporary policy override
with policy_context(SelectionPolicy(...)):
    ...

# Temporary preference change
with with_preference("vendor"):
    ...

# Temporarily deny vendors
with with_denied_vendors({"npu", "cuda"}):
    ...
```

## Files

| File | Description |
|------|-------------|
| `types.py` | OpImplKind, OpImpl definitions |
| `registry.py` | Thread-safe OpRegistry with snapshot support |
| `policy.py` | SelectionPolicy and context managers |
| `discovery.py` | Plugin discovery (entry points + env) |
| `manager.py` | OpManager (main dispatch logic) |
| `builtin_ops.py` | Entry point for backend registration |
| `ops.py` | Convenience API functions |
| `backends/` | Backend implementations directory |
| `backends/__init__.py` | Backend module exports |
| `backends/flagos.py` | DEFAULT implementations (FlagOS/flag_gems) |
| `backends/cuda.py` | VENDOR implementations (vLLM CUDA kernels) |
| `backends/npu.py` | VENDOR implementations (torch_npu for Ascend) |
| `backends/reference.py` | REFERENCE implementations (pure PyTorch) |

## Adding New Operators

To add a new operator implementation:

1. **Choose the appropriate backend file** in `backends/`:
   - `flagos.py` for FlagOS/Triton implementations
   - `cuda.py` for CUDA-specific implementations
   - `npu.py` for NPU-specific implementations
   - `reference.py` for pure PyTorch fallback

2. **Add the implementation function**:
   ```python
   def _my_op_backend(x: torch.Tensor, ...) -> torch.Tensor:
       """My operator using backend."""
       ...

   _my_op_backend._is_available = lambda: check_availability()
   ```

3. **Add to `_IMPLEMENTATIONS` list**:
   ```python
   _IMPLEMENTATIONS = [
       # ... existing implementations ...
       OpImpl(
           op_name="my_op",
           impl_id="default.flagos",  # or vendor.cuda, etc.
           kind=OpImplKind.DEFAULT,
           fn=_my_op_backend,
           priority=100,
           description="My op using backend",
       ),
   ]
   ```

## Adding New Backends

To add a new backend (e.g., for a new hardware vendor):

1. **Create a new file** `backends/my_vendor.py`:
   ```python
   from ..types import OpImpl, OpImplKind

   def _my_op_vendor(x, ...):
       ...

   _IMPLEMENTATIONS = [
       OpImpl(
           op_name="my_op",
           impl_id="vendor.my_vendor",
           kind=OpImplKind.VENDOR,
           vendor="my_vendor",
           fn=_my_op_vendor,
           priority=100,
           description="My op using my_vendor",
       ),
   ]

   def register(registry):
       for impl in _IMPLEMENTATIONS:
           registry.register_impl(impl, skip_duplicate=True)

   def get_implementations():
       return list(_IMPLEMENTATIONS)
   ```

2. **Update `backends/__init__.py`**:
   ```python
   def register_all_backends(registry):
       from . import cuda, flagos, npu, reference, my_vendor
       # ...
       my_vendor.register(registry)
   ```

## Debugging

Enable debug logging to see dispatch decisions:

```python
import logging
logging.getLogger("vllm_fl.dispatch").setLevel(logging.DEBUG)
```

Output example:
```
DEBUG:vllm_fl.dispatch.manager:Resolved op=rms_norm -> vendor.npu (kind=vendor, priority=100)
```
