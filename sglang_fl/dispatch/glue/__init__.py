# Glue layer: translates between SGLang framework-specific parameters
# and the standardized dispatch op signatures.
#
# SGLang's AROUND hook calls glue functions which handle framework-specific
# parameters (post_residual_addition, fused_set_kv_buffer_arg, etc.),
# then call dispatch.call_op() with the standardized signature.

from sglang_fl.dispatch.glue.silu_and_mul import silu_and_mul_glue
from sglang_fl.dispatch.glue.rms_norm import rms_norm_glue
from sglang_fl.dispatch.glue.rotary_embedding import rotary_embedding_glue

__all__ = [
    "silu_and_mul_glue",
    "rms_norm_glue",
    "rotary_embedding_glue",
]
