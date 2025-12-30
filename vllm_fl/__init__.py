# Copyright (c) 2025 BAAI. All rights reserved.

import os


def register():
    """Register the FL platform."""
    from vllm_fl.utils import get_device_info

    device_info = get_device_info()

    # Set multiprocessing method based on backend
    # NPU requires spawn method for multiprocessing
    # CUDA can use fork or spawn (spawn is safer for torch.cuda)
    multiproc_method = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")
    if multiproc_method is None:
        if device_info.is_npu():
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        else:
            # For CUDA, spawn is recommended but not required
            # Let vllm decide by default, or use spawn for safety
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    return "vllm_fl.platform.PlatformFL"


# def register_connector():
#     from vllm_ascend.distributed import register_connector
#     register_connector()
