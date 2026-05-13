# CUDA vendor GemmaRMSNorm — delegates to SGLang's native sgl_kernel.

from __future__ import annotations

from typing import Optional, Union

import torch


def gemma_rms_norm_cuda(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    GemmaRMSNorm using SGLang's native CUDA kernel (sgl_kernel).

    Delegates to obj._forward_impl which calls sgl_kernel.gemma_rmsnorm.
    """
    return obj._forward_impl(x, residual)
