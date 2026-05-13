# Reference GemmaRMSNorm operator implementation using pure PyTorch.
# GemmaRMSNorm: output = rms_norm(x) * (weight + 1.0)

from __future__ import annotations

from typing import Optional, Union

import torch


def gemma_rms_norm_torch(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    Gemma-style RMS normalization using PyTorch.

    Difference from standard RMSNorm: weight semantics are (weight + 1.0).

    Args:
        obj: The calling obj (provides obj.weight, obj.variance_epsilon)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual provided
    """
    weight = obj.weight
    epsilon = obj.variance_epsilon

    orig_dtype = x.dtype

    if residual is not None:
        x = x + residual
        residual = x

    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    # Gemma-style: multiply by (weight + 1.0)
    output = x * (1.0 + weight.float())
    output = output.to(orig_dtype)

    if residual is not None:
        return output, residual
    return output
