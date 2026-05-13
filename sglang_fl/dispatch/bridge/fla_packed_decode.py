# Copyright (c) 2026 BAAI. All rights reserved.
"""Bridge for FLA fused_recurrent_gated_delta_rule_packed_decode function."""

import torch

from sglang_fl.dispatch import call_op


def fused_recurrent_gated_delta_rule_packed_decode_bridge(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Bridge function for fused_recurrent_gated_delta_rule_packed_decode.

    Dispatches to backend implementations via call_op().
    This is the decode-path kernel with a packed mixed_qkv layout.
    """
    return call_op(
        "fused_recurrent_gated_delta_rule_packed_decode",
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
