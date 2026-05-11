# Bridge layer: translates between SGLang framework-specific parameters
# and the standardized dispatch op signatures.
#
# SGLang's AROUND hook calls bridge functions which handle framework-specific
# parameters (post_residual_addition, fused_set_kv_buffer_arg, etc.),
# then call dispatch.call_op() with the standardized signature.

from sglang_fl.dispatch.bridge.silu_and_mul import silu_and_mul_bridge
from sglang_fl.dispatch.bridge.rms_norm import rms_norm_bridge
from sglang_fl.dispatch.bridge.rotary_embedding import rotary_embedding_bridge

__all__ = [
    "silu_and_mul_bridge",
    "rms_norm_bridge",
    "rotary_embedding_bridge",
]
