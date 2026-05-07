"""FlagGems RMSNorm operator for OOT plugin."""
import torch.distributed as dist

import flag_gems

_call_count = 0


def rms_norm_flaggems(self, x, residual=None, post_residual_addition=None):
    """Replace sgl_kernel CUDA RMSNorm with FlagGems Triton implementation."""
    global _call_count
    _call_count += 1
    if _call_count <= 3 and (not dist.is_initialized() or dist.get_rank() == 0):
        print(f"  [OOT-FlagGems] RMSNorm #{_call_count}, shape={tuple(x.shape)}, has_residual={residual is not None}")

    if x.numel() == 0:
        return x

    needs_reshape = x.dim() != 2 and residual is None
    if needs_reshape:
        original_shape = x.shape
        x = x.contiguous().reshape(-1, original_shape[-1])

    if self.variance_size_override is not None:
        return self.forward_native(x, residual, post_residual_addition)

    normalized_shape = [self.hidden_size]

    if residual is not None:
        if post_residual_addition is not None:
            residual = residual + post_residual_addition
        x, residual = flag_gems.fused_add_rms_norm(
            x, residual, normalized_shape, self.weight.data, self.variance_epsilon
        )
        return x, residual

    out = flag_gems.rms_norm(x, normalized_shape, self.weight.data, self.variance_epsilon)
    if needs_reshape:
        out = out.reshape(original_shape)
    return out
