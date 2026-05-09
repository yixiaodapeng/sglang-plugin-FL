# CUDA backend implementation.

from __future__ import annotations

from typing import Optional, Union

import torch

from sglang_fl.dispatch.backends import Backend


class CudaBackend(Backend):
    """
    CUDA backend for operator implementations.

    Uses sgl_kernel (SGLang native CUDA kernels) for NVIDIA GPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "cuda"

    @property
    def vendor(self) -> Optional[str]:
        return "nvidia"

    def is_available(self) -> bool:
        """Check if CUDA hardware and sgl_kernel are available."""
        if CudaBackend._available is None:
            try:
                if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                    CudaBackend._available = False
                    return False
                # Verify this is a real NVIDIA GPU (not MACA/MUSA etc.)
                import sgl_kernel  # noqa: F401

                CudaBackend._available = True
            except (ImportError, Exception):
                CudaBackend._available = False
        return CudaBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_cuda

        return silu_and_mul_cuda(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import rms_norm_cuda

        return rms_norm_cuda(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from .impl.rotary import rotary_embedding_cuda

        return rotary_embedding_cuda(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )
