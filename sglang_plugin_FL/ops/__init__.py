"""OOT operator implementations.

Each operator provides two backends:
  - flagos:     FlagGems Triton kernel (high performance)
  - reference:  PyTorch native (fallback / debugging)
"""

from sglang_plugin_FL.ops.silu_and_mul import silu_and_mul_flaggems
from sglang_plugin_FL.ops.rms_norm import rms_norm_flaggems
from sglang_plugin_FL.ops.rotary_embedding import rotary_embedding_flaggems
from sglang_plugin_FL.ops.reference import (
    silu_and_mul_reference,
    rms_norm_reference,
    rotary_embedding_reference,
)

__all__ = [
    "silu_and_mul_flaggems",
    "rms_norm_flaggems",
    "rotary_embedding_flaggems",
    "silu_and_mul_reference",
    "rms_norm_reference",
    "rotary_embedding_reference",
]
