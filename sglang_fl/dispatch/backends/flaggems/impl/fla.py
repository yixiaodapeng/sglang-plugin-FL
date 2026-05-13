# Copyright (c) 2026 BAAI. All rights reserved.
"""FlagGems implementations for FLA ops."""

from typing import Optional, Tuple
import torch


def chunk_gated_delta_rule_flaggems(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    initial_state_indices: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """
    FlagGems implementation of chunk_gated_delta_rule.

    Currently NOT compatible with SGLang's API:
    - FlagGems signature: (q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens)
    - SGLang signature: (q, k, v, g, beta, scale, initial_state, initial_state_indices, cu_seqlens, head_first, use_qk_l2norm_in_kernel)

    Key differences:
    1. FlagGems lacks `initial_state_indices` (required for radix cache multi-sequence scenarios)
    2. FlagGems has `output_final_state` (removed from SGLang's public API)
    3. FlagGems lacks `head_first` and `use_qk_l2norm_in_kernel`

    Raising NotImplementedError to trigger fallback to vendor.cuda.
    When FlagGems updates its signature, remove this guard and call flag_gems.fused.FLA.chunk_gated_delta_rule_fwd.
    """
    raise NotImplementedError(
        "FlagGems chunk_gated_delta_rule_fwd signature is incompatible with SGLang. "
        "Missing: initial_state_indices, head_first, use_qk_l2norm_in_kernel. "
        "Falling back to vendor.cuda."
    )


def fused_recurrent_gated_delta_rule_flaggems(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    cu_seqlens: Optional[torch.LongTensor] = None,
    ssm_state_indices: Optional[torch.Tensor] = None,
    num_accepted_tokens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """FlagGems implementation of fused_recurrent_gated_delta_rule."""
    try:
        from flag_gems.fused.FLA import fused_recurrent_gated_delta_rule_fwd
    except ImportError:
        raise ImportError(
            "flag_gems.fused.FLA not available. "
            "Install flag-gems with FLA support or use a different backend."
        )

    return fused_recurrent_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_packed_decode_flaggems(
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
    """FlagGems implementation - fallback to SGLang native for now."""
    # TODO: Add FlagGems-specific packed decode kernel if available
    from sglang_fl.dispatch.fla_patch import get_original

    _native_packed_decode = get_original(
        "fused_recurrent_gated_delta_rule_packed_decode"
    )
    return _native_packed_decode(
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
