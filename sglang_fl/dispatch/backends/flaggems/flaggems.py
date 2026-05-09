# FlagGems backend class.

from __future__ import annotations

from .. import Backend


class FlagGemsBackend(Backend):
    """FlagGems (FlagOS) default backend — Triton-based implementations."""

    _available = None

    @property
    def name(self) -> str:
        return "flaggems"

    def is_available(self) -> bool:
        if FlagGemsBackend._available is None:
            try:
                import flag_gems  # noqa: F401

                FlagGemsBackend._available = True
            except ImportError:
                FlagGemsBackend._available = False
        return FlagGemsBackend._available

    def silu_and_mul(self, obj, x):
        from .impl.activation import silu_and_mul_flaggems

        return silu_and_mul_flaggems(obj, x)

    def rms_norm(self, obj, x, residual=None):
        from .impl.normalization import rms_norm_flaggems

        return rms_norm_flaggems(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query,
        key,
        cos,
        sin,
        position_ids,
        rotary_interleaved=False,
        inplace=True,
    ):
        from .impl.rotary import rotary_embedding_flaggems

        return rotary_embedding_flaggems(
            obj, query, key, cos, sin, position_ids, rotary_interleaved, inplace
        )
