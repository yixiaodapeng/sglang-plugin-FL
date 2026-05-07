"""Reference backend — PyTorch native implementations as fallback.

These are the lowest-priority implementations, used when neither
FlagGems (flagos) nor vendor backends are available.
"""

import torch
import torch.nn.functional as F


def silu_and_mul_reference(self, x):
    """PyTorch native SiluAndMul."""
    d = x.shape[-1] // 2
    return F.silu(x[..., :d]) * x[..., d:]


def rms_norm_reference(self, x, residual=None, post_residual_addition=None):
    """PyTorch native RMSNorm."""
    if x.numel() == 0:
        return x

    if self.variance_size_override is not None:
        return self.forward_native(x, residual, post_residual_addition)

    if residual is not None:
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        x = x + residual
        residual = x

    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + self.variance_epsilon)
    out = x.to(self.weight.dtype) * self.weight

    if residual is not None:
        return out, residual
    return out


def rotary_embedding_reference(self, positions, query, key, offsets=None, fused_set_kv_buffer_arg=None):
    """PyTorch native RotaryEmbedding — delegates to SGLang's forward_native."""
    return self.forward_native(positions, query, key, offsets, fused_set_kv_buffer_arg)
