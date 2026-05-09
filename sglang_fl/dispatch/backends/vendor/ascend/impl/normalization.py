# Ascend normalization operator implementations.

from __future__ import annotations

from typing import Optional, Union

import torch


def rms_norm_ascend(
    obj,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """
    RMS normalization using Ascend NPU.

    Args:
        obj: The calling obj (provides obj.weight, obj.variance_epsilon)
        x: Input tensor
        residual: Optional residual tensor

    Returns:
        Normalized tensor, or tuple of (normalized, residual) if residual provided
    """
    import torch_npu

    weight = obj.weight
    epsilon = obj.variance_epsilon

    if residual is not None:
        x, _, residual = torch_npu.npu_add_rms_norm(x, residual, weight, epsilon)
        return x, residual

    x, _ = torch_npu.npu_rms_norm(x, weight, epsilon)
    return x
