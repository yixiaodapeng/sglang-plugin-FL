# Reference backend implementation using PyTorch.

from __future__ import annotations

from typing import Optional, Union

import torch

from sglang_fl.dispatch.backends import Backend


class ReferenceBackend(Backend):
    """
    Reference backend for operator implementations.

    Uses native PyTorch operations; always available as fallback.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "reference"

    def is_available(self) -> bool:
        """Check if PyTorch is available."""
        if ReferenceBackend._available is None:
            try:
                import torch  # noqa: F401

                ReferenceBackend._available = True
            except ImportError:
                ReferenceBackend._available = False
        return ReferenceBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        from .impl.activation import silu_and_mul_torch

        return silu_and_mul_torch(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        from .impl.normalization import rms_norm_torch

        return rms_norm_torch(obj, x, residual)

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
        from .impl.rotary import rotary_embedding_torch

        return rotary_embedding_torch(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )
