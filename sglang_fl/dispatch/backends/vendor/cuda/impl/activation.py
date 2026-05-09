# CUDA activation operator implementations using sgl_kernel.

from __future__ import annotations

import torch


def silu_and_mul_cuda(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using sgl_kernel.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    from sgl_kernel import silu_and_mul as sgl_silu_and_mul

    d = x.shape[-1] // 2
    out = torch.empty(*x.shape[:-1], d, dtype=x.dtype, device=x.device)
    sgl_silu_and_mul(out, x)
    return out
