# Copyright (c) 2025 BAAI. All rights reserved.

from typing import Optional, Union
import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm_fl.utils import is_npu, is_cuda


class RMSNormFL(RMSNorm):
    def __init__(self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        has_weight: bool = True,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using backend-specific implementation."""
        if is_npu():
            return self._forward_npu(x, residual)
        else:
            return self._forward_cuda(x, residual)

    def _forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using flag_gems for CUDA."""
        from flag_gems.modules.normalization import gems_rms_forward
        return gems_rms_forward(x, residual, self.weight, self.variance_epsilon)

    def _forward_npu(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass using torch_npu for NPU.

        Matches vllm-ascend implementation of npu_rms_norm_forward.
        """
        import torch_npu

        if residual is not None:
            x = x + residual
            residual = x

        output = torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]

        if residual is None:
            return output
        else:
            return output, residual


__all__ = ["RMSNormFL"]
        
