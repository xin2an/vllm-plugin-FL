# Dispatch Mechanism

This directory implements the operator dispatch mechanism for vllm-plugin-FL, providing a flexible operator dispatch system that selects between different backend implementations (FlagGems, PyTorch, etc.) based on availability and policy configuration.

## Directory Structure

```
dispatch/
├── __init__.py              # Module entry point, exports public API
├── types.py                 # Core type definitions (OpImpl, BackendImplKind)
├── registry.py              # Thread-safe operator registry
├── policy.py                # Selection policy management
├── manager.py               # Core dispatch manager
├── builtin_ops.py           # Built-in operator registration (calls backend register_ops)
├── ops.py                   # Backend base interface (VLLMFLBackendBase)
├── discovery.py             # Plugin discovery mechanism
├── logger_manager.py        # Centralized logging configuration
└── backends/                # Backend implementations
    ├── __init__.py
    ├── base.py              # Backend abstract base class
    ├── flaggems/            # FlagGems backend
    │   ├── __init__.py
    │   ├── flaggems.py      # Backend class
    │   ├── register_ops.py  # Operator registration
    │   └── impl/            # Operator implementations
    │       ├── __init__.py
    │       ├── activation.py
    │       ├── normalization.py
    │       └── rotary.py
    ├── reference/           # Reference backend (PyTorch)
    │   ├── __init__.py
    │   ├── reference.py     # Backend class
    │   ├── register_ops.py  # Operator registration
    │   └── impl/            # Operator implementations
    │       ├── __init__.py
    │       ├── activation.py
    │       ├── normalization.py
    │       └── rotary.py
    └── vendor/              # Vendor-specific backends
        └── __init__.py      # (Add CUDA, etc. as needed)
```

## Core Concepts

### 1. Backend Implementation Kind (BackendImplKind)

- **DEFAULT**: Default implementation (FlagGems), priority 150
- **REFERENCE**: Reference implementation (PyTorch native), priority 50
- **VENDOR**: Vendor-specific implementation (e.g., CUDA), requires vendor name

### 2. Operator Implementation (OpImpl)

Each operator implementation contains the following attributes:
- `op_name`: Operator name (e.g., "silu_and_mul", "rmsnorm")
- `impl_id`: Unique implementation identifier (e.g., "default.flaggems")
- `kind`: Implementation type
- `fn`: Actual implementation function
- `vendor`: Vendor name (required for VENDOR type)
- `priority`: Selection priority (higher value = preferred)

### 3. Selection Policy (SelectionPolicy)

Policy controls operator implementation selection behavior:
- `prefer`: Preferred implementation type
- `strict`: Strict mode, whether to raise error when primary implementation fails
- `per_op_order`: Custom selection order for each operator
- `deny_vendors`: List of denied vendors
- `allow_vendors`: Whitelist of allowed vendors

## Quick Start

### Basic Usage

```python
from vllm_fl.dispatch import call_op, resolve_op

# Method 1: Call operator directly
result = call_op("silu_and_mul", x)

# Method 2: Resolve first, then call
fn = resolve_op("rmsnorm")
result = fn(x, residual, weight, epsilon)
```

### Using the Manager

```python
from vllm_fl.dispatch import get_default_manager

manager = get_default_manager()

# Resolve operator
fn = manager.resolve("rotary_embedding")
result = fn(query, key, cos, sin, position_ids)

# Or call directly
result = manager.call("silu_and_mul", x)
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `VLLM_FL_PREFER` | Preferred backend | `flaggems`, `vendor`, `reference` |
| `VLLM_FL_STRICT` | Strict mode | `1` or `0` |
| `VLLM_FL_DENY_VENDORS` | Denied vendors list | `vendor1,vendor2` |
| `VLLM_FL_ALLOW_VENDORS` | Allowed vendors whitelist | `vendor1,vendor2` |
| `VLLM_FL_PER_OP` | Per-operator selection order | `op1=a\|b\|c;op2=x\|y` |
| `VLLM_FL_PLUGIN_MODULES` | Plugin modules to load | `my_plugin,another_plugin` |
| `VLLM_FL_LOG_LEVEL` | Log level | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Examples

```bash
# Prefer FlagGems implementation
export VLLM_FL_PREFER=flaggems

# Enable strict mode (auto-fallback on failure)
export VLLM_FL_STRICT=1

# Deny specific vendors
export VLLM_FL_DENY_VENDORS=vendor_a,vendor_b

# Specify selection order for specific operator
export VLLM_FL_PER_OP="rmsnorm=vendor|flaggems|reference"

# Load external plugins
export VLLM_FL_PLUGIN_MODULES=my_custom_backend

# Set log level
export VLLM_FL_LOG_LEVEL=DEBUG
```

## Policy Context Management

Supports temporary policy override in code:

```python
from vllm_fl.dispatch import (
    policy_context,
    with_strict_mode,
    with_preference,
    with_allowed_vendors,
    with_denied_vendors,
    SelectionPolicy,
)

# Temporarily enable strict mode
with with_strict_mode():
    result = call_op("silu_and_mul", x)

# Temporarily switch preferred backend
with with_preference("reference"):
    result = call_op("rmsnorm", x, residual, weight, epsilon)

# Temporarily restrict allowed vendors
with with_allowed_vendors("vendor_a"):
    result = call_op("rotary_embedding", query, key, cos, sin, position_ids)

# Use custom policy
custom_policy = SelectionPolicy.from_dict(
    prefer="flaggems",
    strict=True,
    deny_vendors={"vendor_x"},
)
with policy_context(custom_policy):
    result = call_op("silu_and_mul", x)
```

## Supported Operators

Currently supported operators:

| Operator | Description | FlagGems | Reference |
|----------|-------------|----------|-----------|
| `silu_and_mul` | SiLU activation + element-wise multiplication | ✓ | ✓ |
| `rmsnorm` | RMS normalization | ✓ | ✓ |
| `rotary_embedding` | Rotary position embedding | ✓ | ✓ |

## Selection Process

1. **Cache Check**: Check if dispatch cache hits
2. **Get Implementations**: Retrieve all registered implementations from registry
3. **Vendor Filtering**: Filter by policy's allow/deny lists
4. **Availability Check**: Call `is_available()` to check if implementation is available
5. **Priority Sorting**: Select best implementation based on per-op order or default order
6. **Cache Result**: Cache selection result to speed up subsequent calls

## Fallback Mechanism

When `VLLM_FL_STRICT=1`, if the primary implementation fails, the system automatically tries other available implementations:

```
Op 'rmsnorm' using 'default.flaggems' (kind=flaggems, vendor=None)
[WARNING] Implementation 'default.flaggems' failed for op 'rmsnorm': ...
Op 'rmsnorm' fallback to 'reference.torch' (kind=reference, vendor=None)
```

## Extending with New Operators

When adding a new operator (e.g., `layernorm`), modify the following files:

| File | Changes |
|------|---------|
| `backends/flaggems/impl/normalization.py` | Add FlagGems implementation |
| `backends/flaggems/flaggems.py` | Add method to backend class |
| `backends/flaggems/register_ops.py` | Register OpImpl |
| `backends/reference/impl/normalization.py` | Add PyTorch implementation |
| `backends/reference/reference.py` | Add method to backend class |
| `backends/reference/register_ops.py` | Register OpImpl |
| `ops.py` | Add abstract method declaration |

## Extending with New Backends

### 1. Create Backend Directory Structure

```
backends/my_backend/
├── __init__.py
├── my_backend.py      # Backend class
├── register_ops.py    # Operator registration
└── impl/              # Operator implementations
    ├── __init__.py
    ├── activation.py
    └── ...
```

### 2. Implement Backend Class

```python
# backends/my_backend/my_backend.py
from ..base import Backend

class MyBackend(Backend):
    @property
    def name(self) -> str:
        return "my_backend"

    def is_available(self) -> bool:
        try:
            import my_library
            return True
        except ImportError:
            return False

    def silu_and_mul(self, x):
        from .impl.activation import silu_and_mul_my_backend
        return silu_and_mul_my_backend(x)
```

### 3. Create Registration Module

```python
# backends/my_backend/register_ops.py
from ...types import OpImpl, BackendImplKind

def register_builtins(registry) -> None:
    from .my_backend import MyBackend

    backend = MyBackend()
    is_avail = backend.is_available

    impls = [
        OpImpl(
            op_name="silu_and_mul",
            impl_id="default.my_backend",
            kind=BackendImplKind.DEFAULT,
            fn=_bind_is_available(backend.silu_and_mul, is_avail),
            priority=100,
        ),
    ]

    registry.register_many(impls)
```

### 4. Update builtin_ops.py

```python
# In builtin_ops.py, add:
try:
    from .backends.my_backend.register_ops import register_builtins as register_my_backend
    register_my_backend(registry)
except Exception as e:
    logger.warning(f"Failed to register MyBackend operators: {e}")
```

## Plugin Discovery

External plugins can register operators via:

### 1. Entry Points (Recommended)

```python
# In your plugin's setup.py or pyproject.toml
[project.entry-points."vllm_fl.plugin"]
my_plugin = "my_plugin_package:register"
```

```python
# my_plugin_package/__init__.py
def register(registry):
    # Register your operators
    registry.register_impl(OpImpl(...))
```

### 2. Environment Variable

```bash
export VLLM_FL_PLUGIN_MODULES=my_plugin_module
```

```python
# my_plugin_module.py
def vllm_fl_register(registry):
    # Register your operators
    pass
```

## Multi-Process Safety

OpManager supports multi-process environments:
- Uses `os.register_at_fork()` to automatically reset state after fork
- PID detection ensures independent initialization per process
- Thread-safe registry and cache operations

## API Reference

### Convenience Functions

- `call_op(op_name, *args, **kwargs)`: Call an operator
- `resolve_op(op_name)`: Resolve operator implementation

### Policy Management

- `get_policy()`: Get current policy
- `set_global_policy(policy)`: Set global policy
- `reset_global_policy()`: Reset to environment variable defaults
- `policy_context(policy)`: Temporary policy context

### Manager

- `get_default_manager()`: Get default manager instance
- `reset_default_manager()`: Reset default manager

### Plugin Discovery

- `discover_plugins(registry)`: Discover and load plugins
- `get_discovered_plugins()`: Get list of discovered plugins
- `clear_discovered_plugins()`: Clear discovered plugins list

### Logging

- `get_logger(name)`: Get logger instance
- `set_log_level(level, name)`: Set log level
