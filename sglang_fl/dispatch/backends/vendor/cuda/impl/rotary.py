# CUDA rotary embedding operator implementations using sgl_kernel.

from __future__ import annotations

import torch


def rotary_embedding_cuda(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding using sgl_kernel.

    Args:
        obj: The calling obj (for interface consistency)
        query: Query tensor
        key: Key tensor
        cos: Cosine cache
        sin: Sine cache
        position_ids: Position indices
        rotary_interleaved: Whether to use interleaved rotary
        inplace: Whether to modify tensors in-place

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    from sgl_kernel import rotary_embedding as sgl_rotary_embedding

    sgl_rotary_embedding(
        position_ids,
        query,
        key,
        cos,
        sin,
        rotary_interleaved,
    )
    return query, key
