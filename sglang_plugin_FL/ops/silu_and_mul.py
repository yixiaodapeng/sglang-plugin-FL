"""FlagGems SiluAndMul operator for OOT plugin."""
import torch.distributed as dist

import flag_gems

_call_count = 0


def silu_and_mul_flaggems(self, x):
    """Replace sgl_kernel CUDA SiluAndMul with FlagGems Triton implementation."""
    global _call_count
    _call_count += 1
    if _call_count <= 3 and (not dist.is_initialized() or dist.get_rank() == 0):
        print(f"  [OOT-FlagGems] SiluAndMul #{_call_count}, shape={tuple(x.shape)}")
    d = x.shape[-1] // 2
    return flag_gems.silu_and_mul(x[..., :d], x[..., d:])
