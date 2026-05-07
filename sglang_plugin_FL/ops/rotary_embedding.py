"""FlagGems RotaryEmbedding operator for OOT plugin.

Replaces SGLang's forward_cuda which uses sgl_kernel JIT CUDA kernels
(apply_rope_with_cos_sin_cache_inplace) with FlagGems' gems_rope_forward.

Interface contract: same signature as RotaryEmbedding.forward_cuda:
    forward(self, positions, query, key, offsets=None, fused_set_kv_buffer_arg=None)
"""

import torch
import torch.distributed as dist

_call_count = 0


def rotary_embedding_flaggems(self, positions, query, key, offsets=None, fused_set_kv_buffer_arg=None):
    """Replace sgl_kernel CUDA RotaryEmbedding with FlagGems Triton implementation."""
    global _call_count
    _call_count += 1
    if _call_count <= 3 and (not dist.is_initialized() or dist.get_rank() == 0):
        print(f"  [OOT-FlagGems] RotaryEmbedding #{_call_count}, q_shape={tuple(query.shape)}")

    from flag_gems.modules.rotary_embedding import gems_rope_forward

    # fused_set_kv_buffer_arg is a CUDA-specific optimization (fused rope + kv cache write).
    # On non-NVIDIA we skip it and just do rope.
    assert fused_set_kv_buffer_arg is None, (
        "fused_set_kv_buffer_arg is not supported in FlagGems rotary embedding"
    )

    if offsets is not None:
        positions = positions + offsets

    positions = positions.flatten()
    num_tokens = positions.shape[0]

    # Reshape query/key: [num_tokens, num_heads * head_size] -> [num_tokens, num_heads, head_size]
    query_shape = query.shape
    key_shape = key.shape
    query = query.view(num_tokens, -1, self.head_size)
    key = key.view(num_tokens, -1, self.head_size)

    # cos_sin_cache: [max_seq_len, rotary_dim] where first half is cos, second half is sin
    cos_sin_cache = self.cos_sin_cache.to(query.dtype)
    cos, sin = cos_sin_cache.chunk(2, dim=-1)

    if self.rotary_dim == self.head_size:
        # Full rotary: apply inplace for efficiency
        gems_rope_forward(
            query,
            key,
            cos,
            sin,
            position_ids=positions,
            rotary_interleaved=not self.is_neox_style,
            inplace=True,
        )
    else:
        # Partial rotary: only rotate first rotary_dim elements, pass through the rest
        query_rot = query[..., : self.rotary_dim].contiguous()
        key_rot = key[..., : self.rotary_dim].contiguous()

        q_embed, k_embed = gems_rope_forward(
            query_rot,
            key_rot,
            cos,
            sin,
            position_ids=positions,
            rotary_interleaved=not self.is_neox_style,
            inplace=False,
        )

        query = torch.cat((q_embed, query[..., self.rotary_dim:]), dim=-1)
        key = torch.cat((k_embed, key[..., self.rotary_dim:]), dim=-1)

    return query.reshape(query_shape), key.reshape(key_shape)
