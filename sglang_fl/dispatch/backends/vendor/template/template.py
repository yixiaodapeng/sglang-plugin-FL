# Template backend implementation.
#
# Copy this directory as your starting point for a new vendor backend.
# Replace 'template' with your vendor name throughout.

from __future__ import annotations

from typing import Optional

from sglang_fl.dispatch.backends import Backend


class TemplateBackend(Backend):
    """
    Template backend for operator implementations.

    Replace this with your vendor-specific backend.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "template"

    @property
    def vendor(self) -> Optional[str]:
        return "template"

    def is_available(self) -> bool:
        """Check if vendor hardware/libraries are available."""
        if TemplateBackend._available is None:
            try:
                # TODO: Add your vendor-specific availability check here
                # Example:
                # import your_vendor_library
                # TemplateBackend._available = your_vendor_library.is_available()
                TemplateBackend._available = False
            except (ImportError, AttributeError):
                TemplateBackend._available = False
        return TemplateBackend._available

    # ==================== Operator Implementations ====================
    # Implement the operators your backend supports below.
    #
    # def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
    #     from .impl.activation import silu_and_mul_template
    #     return silu_and_mul_template(obj, x)
    #
    # def rms_norm(self, obj, x, residual=None):
    #     from .impl.normalization import rms_norm_template
    #     return rms_norm_template(obj, x, residual)
    #
    # def rotary_embedding(self, obj, query, key, cos, sin, position_ids,
    #                      rotary_interleaved=False, inplace=True):
    #     from .impl.rotary import rotary_embedding_template
    #     return rotary_embedding_template(
    #         obj, query, key, cos, sin, position_ids,
    #         rotary_interleaved, inplace,
    #     )
