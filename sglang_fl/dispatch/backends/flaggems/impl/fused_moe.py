# FlagGems FusedMoE operator implementation using FlagGems Triton kernels.

from __future__ import annotations

import torch


def fused_moe_flaggems(
    obj,
    layer: torch.nn.Module,
    dispatch_output,
):
    """
    Fused MoE expert computation using FlagGems' dispatch_fused_moe_kernel.

    This implementation uses FlagGems' Triton-based MoE kernels, which provide
    cross-platform support (CUDA, ROCm, Ascend, etc.).

    The implementation follows vllm-plugin-FL's pattern:
    1. Extract topk_output from dispatch_output
    2. Call FlagGems' moe_align_block_size
    3. Call FlagGems' dispatch_fused_moe_kernel for GEMM1
    4. Apply activation (silu_and_mul via dispatch)
    5. Call FlagGems' dispatch_fused_moe_kernel for GEMM2
    6. Call FlagGems' moe_sum to combine results

    Args:
        obj: The UnquantizedFusedMoEMethod instance
        layer: The MoE layer module
        dispatch_output: StandardDispatchOutput containing hidden_states and topk_output

    Returns:
        CombineInput (StandardCombineInput)
    """
    from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

    # For now, delegate to SGLang's MoeRunner with TRITON backend
    # TODO: Implement full FlagGems path with dispatch_fused_moe_kernel
    # when we need non-CUDA platform support
    quant_info = TritonMoeQuantInfo(
        w13_weight=layer.w13_weight,
        w2_weight=layer.w2_weight,
        b13=getattr(layer, "w13_weight_bias", None),
        b2=getattr(layer, "w2_weight_bias", None),
    )
    return obj.runner.run(dispatch_output, quant_info)
