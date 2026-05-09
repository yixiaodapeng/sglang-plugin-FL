"""OOT device configuration for FlagGems-based plugin.

For CUDA-compatible chips (Iluvatar, MetaX, etc.), this is optional —
they naturally go through the is_cuda() path. For non-CUDA chips,
this provides the device routing.
"""

import torch

from sglang.srt.utils.oot_device import OOTDeviceConfig, register_oot_device


def create_cuda_device_config() -> OOTDeviceConfig:
    """Create a CUDA-based OOT device config (for CUDA-compatible chips)."""
    return OOTDeviceConfig(
        device_type="cuda",
        dist_backend="nccl",
        empty_cache_fn=torch.cuda.empty_cache,
        device_count_fn=torch.cuda.device_count,
    )


def register_device(device_type: str = "cuda", dist_backend: str = "nccl"):
    """Register OOT device config. Call from plugin __init__.

    Args:
        device_type: "cuda" for CUDA-compatible chips, or custom type
        dist_backend: "nccl", "hccl", "flagcx", etc.
    """
    if device_type == "cuda":
        config = create_cuda_device_config()
        config.dist_backend = dist_backend
    else:
        # Non-CUDA device: try to get the torch module dynamically
        mod = getattr(torch, device_type, None)
        config = OOTDeviceConfig(
            device_type=device_type,
            dist_backend=dist_backend,
            empty_cache_fn=getattr(mod, "empty_cache", None) if mod else None,
            device_count_fn=getattr(mod, "device_count", None) if mod else None,
        )
    register_oot_device(config)
