# Copyright (c) 2025 BAAI. All rights reserved.
"""
Device utilities for vllm_fl plugin.
This module provides device detection with lazy initialization to avoid
NPU initialization during module import (which causes issues with multiprocessing spawn).
Supports multiple backends: NVIDIA (CUDA), Ascend (NPU), etc.
"""
import importlib
import os
from typing import Optional

import torch


class DeviceInfo:
    """Lazy device information class that doesn't initialize device on import.

    Supports multiple backends:
    - nvidia (cuda)
    - ascend (npu)
    - cambricon (mlu)
    - mthreads (musa)
    - iluvatar (corex)
    """

    _instance: Optional["DeviceInfo"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceInfo, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if DeviceInfo._initialized:
            return
        DeviceInfo._initialized = True

        self._device_type: Optional[str] = None
        self._vendor_name: Optional[str] = None
        self._dispatch_key: Optional[str] = None
        self._torch_device_fn = None
        self._torch_backend_device = None

        # Detect vendor without initializing the device
        self._vendor_name = self._detect_vendor()
        self._device_type = self._get_device_name(self._vendor_name)
        self._dispatch_key = self._device_type.upper()

    def _detect_vendor(self) -> str:
        """Detect vendor without triggering device initialization."""
        # Check environment variable first
        vendor_from_env = os.environ.get("GEMS_VENDOR")
        if vendor_from_env:
            return vendor_from_env

        # Check for specific torch modules
        vendor_map = {
            "ascend": "npu",
            "cambricon": "mlu",
            "mthreads": "musa",
            "iluvatar": "corex",
        }

        for vendor_name, flag in vendor_map.items():
            if hasattr(torch, flag):
                return vendor_name

        # Check for torch_npu
        try:
            import torch_npu
            if hasattr(torch_npu, "npu"):
                return "ascend"
        except ImportError:
            pass

        # Default to nvidia/cuda if available
        if torch.cuda.is_available():
            return "nvidia"

        return "nvidia"  # fallback

    def _get_device_name(self, vendor_name: str) -> str:
        """Get device name from vendor name."""
        device_map = {
            "nvidia": "cuda",
            "ascend": "npu",
            "cambricon": "mlu",
            "mthreads": "musa",
            "iluvatar": "corex",
        }
        return device_map.get(vendor_name, "cuda")

    @property
    def dispatch_key(self) -> str:
        return self._dispatch_key

    @property
    def vendor_name(self) -> str:
        return self._vendor_name

    @property
    def device_type(self) -> str:
        return self._device_type

    @property
    def torch_device_fn(self):
        """Lazily get torch device function (e.g., torch.cuda or torch.npu)."""
        if self._torch_device_fn is None:
            self._torch_device_fn = getattr(torch, self._device_type)
        return self._torch_device_fn

    @property
    def torch_backend_device(self):
        """Lazily get torch backend device (e.g., torch.backends.cuda)."""
        if self._torch_backend_device is None:
            if self._device_type in ("musa", "aipu", "npu"):
                self._torch_backend_device = None
            else:
                try:
                    self._torch_backend_device = importlib.import_module(
                        f"torch.backends.{self._device_type}"
                    )
                except ImportError:
                    self._torch_backend_device = None
        return self._torch_backend_device

    def is_cuda(self) -> bool:
        """Check if the current device is NVIDIA CUDA."""
        return self._vendor_name == "nvidia"

    def is_npu(self) -> bool:
        """Check if the current device is Ascend NPU."""
        return self._vendor_name == "ascend"

    def get_supported_device(self) -> bool:
        """Check if the current vendor is supported."""
        supported_device = ["nvidia", "ascend"]
        if self.vendor_name not in supported_device:
            raise NotImplementedError(f"{self.vendor_name} is not supported now!")
        return True


# Global device info instance (lazy initialization)
_device_info: Optional[DeviceInfo] = None


def get_device_info() -> DeviceInfo:
    """Get the global DeviceInfo instance with lazy initialization."""
    global _device_info
    if _device_info is None:
        _device_info = DeviceInfo()
    return _device_info


def is_npu() -> bool:
    """Check if running on Ascend NPU."""
    return get_device_info().is_npu()


def is_cuda() -> bool:
    """Check if running on NVIDIA CUDA."""
    return get_device_info().is_cuda()


if __name__ == "__main__":
    device = DeviceInfo()
    print(f"Vendor: {device.vendor_name}")
    print(f"Device type: {device.device_type}")
    print(f"Dispatch key: {device.dispatch_key}")
    print(f"Is CUDA: {device.is_cuda()}")
    print(f"Is NPU: {device.is_npu()}")
