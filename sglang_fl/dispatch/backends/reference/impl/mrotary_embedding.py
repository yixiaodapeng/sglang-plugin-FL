# Reference MRotaryEmbedding operator implementation using pure PyTorch.
# Handles both 1D positions (text-only) and 2D positions (multimodal).

from __future__ import annotations

from typing import Tuple

import torch


def mrotary_embedding_torch(
    obj,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Multimodal rotary position embedding using PyTorch.

    Handles:
    - 1D positions [num_tokens]: standard rope applied uniformly
    - 2D positions [3, num_tokens]: mrope with section-based splitting

    Args:
        obj: The MRotaryEmbedding instance (provides cos_sin_cache, mrope_section,
             head_size, is_neox_style, rotary_dim)
        positions: Position tensor, 1D or 2D
        query: Query tensor [num_tokens, num_heads * head_size]
        key: Key tensor [num_tokens, num_kv_heads * head_size]

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    head_size = obj.head_size
    rotary_dim = obj.rotary_dim
    is_neox_style = obj.is_neox_style
    cos_sin_cache = obj.cos_sin_cache.to(dtype=query.dtype)

    num_tokens = query.shape[0]
    query_shape = query.shape
    key_shape = key.shape

    # Reshape to [num_tokens, num_heads, head_size]
    query = query.view(num_tokens, -1, head_size)
    key = key.view(num_tokens, -1, head_size)

    if positions.ndim == 2 and hasattr(obj, "mrope_section") and obj.mrope_section:
        # Multimodal: positions is [3, num_tokens], mrope_section splits rotary_dim
        # e.g. mrope_section = [24, 20, 20] for Qwen3-VL
        mrope_section = obj.mrope_section
        cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)

        # Apply rope per section
        query_out = query.clone()
        key_out = key.clone()

        offset = 0
        for i, section_dim in enumerate(mrope_section):
            # Each section uses positions[i] and a slice of cos/sin
            pos_i = positions[i]  # [num_tokens]
            cos_i = cos_cache[pos_i]  # [num_tokens, half_rotary_dim]
            sin_i = sin_cache[pos_i]  # [num_tokens, half_rotary_dim]

            # Slice the section from cos/sin (each section covers section_dim/2 of cos)
            half_section = section_dim // 2
            cos_section = cos_i[:, offset // 2 : offset // 2 + half_section]
            sin_section = sin_i[:, offset // 2 : offset // 2 + half_section]

            # Slice query/key for this section
            q_section = query_out[:, :, offset : offset + section_dim]
            k_section = key_out[:, :, offset : offset + section_dim]

            # Expand cos/sin for broadcasting: [num_tokens, 1, section_dim]
            cos_expanded = cos_section.unsqueeze(1).repeat(1, 1, 2)
            sin_expanded = sin_section.unsqueeze(1).repeat(1, 1, 2)

            # Apply rope
            if is_neox_style:
                q1 = q_section[..., : section_dim // 2]
                q2 = q_section[..., section_dim // 2 :]
                q_rotated = torch.cat((-q2, q1), dim=-1)
                query_out[:, :, offset : offset + section_dim] = (
                    q_section * cos_expanded + q_rotated * sin_expanded
                )

                k1 = k_section[..., : section_dim // 2]
                k2 = k_section[..., section_dim // 2 :]
                k_rotated = torch.cat((-k2, k1), dim=-1)
                key_out[:, :, offset : offset + section_dim] = (
                    k_section * cos_expanded + k_rotated * sin_expanded
                )
            else:
                q1 = q_section[..., ::2]
                q2 = q_section[..., 1::2]
                q_rotated = torch.stack((-q2, q1), dim=-1).flatten(-2)
                query_out[:, :, offset : offset + section_dim] = (
                    q_section * cos_expanded + q_rotated * sin_expanded
                )

                k1 = k_section[..., ::2]
                k2 = k_section[..., 1::2]
                k_rotated = torch.stack((-k2, k1), dim=-1).flatten(-2)
                key_out[:, :, offset : offset + section_dim] = (
                    k_section * cos_expanded + k_rotated * sin_expanded
                )

            offset += section_dim

        query = query_out.reshape(query_shape)
        key = key_out.reshape(key_shape)
        return query, key

    else:
        # 1D positions: standard rope (same logic as rotary_embedding reference)
        if positions.ndim == 2:
            # Fallback: use first row if no mrope_section
            positions = positions[0]

        cos_cache, sin_cache = cos_sin_cache.chunk(2, dim=-1)
        cos_selected = cos_cache[positions].unsqueeze(1)  # [num_tokens, 1, half_dim]
        sin_selected = sin_cache[positions].unsqueeze(1)

        # Only apply to rotary_dim dimensions
        if rotary_dim < head_size:
            query_rot = query[..., :rotary_dim]
            key_rot = key[..., :rotary_dim]
            query_pass = query[..., rotary_dim:]
            key_pass = key[..., rotary_dim:]
        else:
            query_rot = query
            key_rot = key
            query_pass = None
            key_pass = None

        # Expand cos/sin to match rotary_dim
        half_dim = cos_selected.shape[-1]
        if half_dim * 2 > rotary_dim:
            cos_selected = cos_selected[..., : rotary_dim // 2]
            sin_selected = sin_selected[..., : rotary_dim // 2]

        cos_expanded = torch.cat([cos_selected, cos_selected], dim=-1)
        sin_expanded = torch.cat([sin_selected, sin_selected], dim=-1)

        if is_neox_style:
            q1 = query_rot[..., : rotary_dim // 2]
            q2 = query_rot[..., rotary_dim // 2 :]
            q_rotated = torch.cat((-q2, q1), dim=-1)
            q_embed = query_rot * cos_expanded + q_rotated * sin_expanded

            k1 = key_rot[..., : rotary_dim // 2]
            k2 = key_rot[..., rotary_dim // 2 :]
            k_rotated = torch.cat((-k2, k1), dim=-1)
            k_embed = key_rot * cos_expanded + k_rotated * sin_expanded
        else:
            q1 = query_rot[..., ::2]
            q2 = query_rot[..., 1::2]
            q_rotated = torch.stack((-q2, q1), dim=-1).flatten(-2)
            q_embed = query_rot * cos_expanded + q_rotated * sin_expanded

            k1 = key_rot[..., ::2]
            k2 = key_rot[..., 1::2]
            k_rotated = torch.stack((-k2, k1), dim=-1).flatten(-2)
            k_embed = key_rot * cos_expanded + k_rotated * sin_expanded

        if query_pass is not None:
            query = torch.cat([q_embed, query_pass], dim=-1)
            key = torch.cat([k_embed, key_pass], dim=-1)
        else:
            query = q_embed
            key = k_embed

        query = query.reshape(query_shape)
        key = key.reshape(key_shape)
        return query, key
