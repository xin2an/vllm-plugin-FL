# Copyright (c) 2025 BAAI. All rights reserved.

"""
Plugin discovery for vllm-plugin-FL dispatch system.

Supports two discovery mechanisms:
1. Entry points (vllm_fl.dispatch.plugins group)
2. Environment variable module import (VLLM_FL_PLUGIN_MODULES)
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import TYPE_CHECKING, Callable, List, Optional

if TYPE_CHECKING:
    from .registry import OpRegistry

logger = logging.getLogger(__name__)

# Entry points group name
PLUGIN_GROUP = "vllm_fl.dispatch.plugins"

# Environment variable for module discovery
ENV_PLUGIN_MODULES = "VLLM_FL_PLUGIN_MODULES"


def _try_import_entry_points():
    """Try to import entry_points from importlib.metadata."""
    try:
        from importlib.metadata import entry_points
        return entry_points
    except ImportError:
        pass

    try:
        from importlib_metadata import entry_points  # type: ignore
        return entry_points
    except ImportError:
        pass

    return None


def _get_register_function(module) -> Optional[Callable]:
    """
    Get the register function from a module.

    Looks for (in order):
    1. vllm_fl_dispatch_register
    2. dispatch_register
    3. register
    """
    for name in ("vllm_fl_dispatch_register", "dispatch_register", "register"):
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    return None


def discover_from_entry_points(registry: "OpRegistry") -> List[str]:
    """
    Discover and register plugins from entry points.

    Entry points should be in the 'vllm_fl.dispatch.plugins' group.

    Example pyproject.toml:
        [project.entry-points."vllm_fl.dispatch.plugins"]
        my_vendor = "my_vendor_pkg:vllm_fl_dispatch_register"

    Returns:
        List of successfully loaded plugin names
    """
    entry_points_fn = _try_import_entry_points()
    if entry_points_fn is None:
        logger.debug("importlib.metadata.entry_points not available")
        return []

    loaded: List[str] = []

    try:
        eps = entry_points_fn()

        # Handle both old and new entry_points API
        if hasattr(eps, "select"):
            # Python 3.10+ / importlib_metadata 3.6+
            group = eps.select(group=PLUGIN_GROUP)
        elif hasattr(eps, "get"):
            # Older API returns dict
            group = eps.get(PLUGIN_GROUP, [])
        else:
            # Fallback: try to iterate
            group = [ep for ep in eps if getattr(ep, "group", None) == PLUGIN_GROUP]

        for ep in group:
            try:
                fn = ep.load()
                if callable(fn):
                    fn(registry)
                    loaded.append(ep.name)
                    logger.info(f"Loaded dispatch plugin: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load dispatch plugin {ep.name}: {e}")

    except Exception as e:
        logger.warning(f"Error discovering entry points: {e}")

    return loaded


def discover_from_env_modules(registry: "OpRegistry") -> List[str]:
    """
    Discover and register plugins from environment variable.

    Set VLLM_FL_PLUGIN_MODULES to a comma-separated list of module names.

    Example:
        VLLM_FL_PLUGIN_MODULES=my_vendor.ops,another_vendor.kernels

    Each module should have a register function:
        def vllm_fl_dispatch_register(registry: OpRegistry) -> None:
            registry.register_impl(...)

    Returns:
        List of successfully loaded module names
    """
    modules_str = os.getenv(ENV_PLUGIN_MODULES, "").strip()
    if not modules_str:
        return []

    loaded: List[str] = []
    module_names = [m.strip() for m in modules_str.split(",") if m.strip()]

    for name in module_names:
        try:
            module = importlib.import_module(name)
            register_fn = _get_register_function(module)

            if register_fn:
                register_fn(registry)
                loaded.append(name)
                logger.info(f"Loaded dispatch plugin module: {name}")
            else:
                logger.warning(
                    f"Module {name} has no register function "
                    f"(expected vllm_fl_dispatch_register, dispatch_register, or register)"
                )

        except ImportError as e:
            logger.warning(f"Failed to import dispatch plugin module {name}: {e}")
        except Exception as e:
            logger.warning(f"Failed to register dispatch plugin module {name}: {e}")

    return loaded


def discover_plugins(registry: "OpRegistry") -> List[str]:
    """
    Discover and register all plugins from all sources.

    Returns:
        List of all successfully loaded plugin/module names
    """
    loaded: List[str] = []

    # Entry points first
    loaded.extend(discover_from_entry_points(registry))

    # Then environment modules
    loaded.extend(discover_from_env_modules(registry))

    if loaded:
        logger.info(f"Discovered {len(loaded)} dispatch plugin(s): {loaded}")
    else:
        logger.debug("No external dispatch plugins discovered")

    return loaded
