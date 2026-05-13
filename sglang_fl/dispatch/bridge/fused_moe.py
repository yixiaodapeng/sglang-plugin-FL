# Bridge: UnquantizedFusedMoEMethod
#
# SGLang signature: forward_cuda(self, layer, dispatch_output) -> CombineInput
# Dispatch signature: fn(obj, layer, dispatch_output) -> CombineInput
# Mapping: pass-through (1:1)

from __future__ import annotations

import torch

from sglang_fl.dispatch import call_op


def fused_moe_bridge(
    self,
    layer: torch.nn.Module,
    dispatch_output,
):
    """SGLang UnquantizedFusedMoEMethod forward → dispatch call_op("fused_moe", ...)."""
    return call_op("fused_moe", self, layer, dispatch_output)
