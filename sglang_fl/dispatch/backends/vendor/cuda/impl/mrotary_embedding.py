# CUDA vendor MRotaryEmbedding — delegates to SGLang's native implementation.

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_cuda(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MRotaryEmbedding using SGLang's native CUDA/triton kernels.

    For 2D positions: calls forward_triton (triton_mrope_fused).
    For 1D positions: calls parent RotaryEmbedding.forward_cuda logic.
    """
    if positions.ndim == 2 and hasattr(obj, "mrope_section") and obj.mrope_section:
        return obj.forward_triton(positions, query, key)
    # 1D positions: use standard sgl_kernel rope
    from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding

    return RotaryEmbedding.forward_cuda(obj, positions, query, key)
