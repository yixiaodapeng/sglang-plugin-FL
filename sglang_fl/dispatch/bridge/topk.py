# Bridge: TopK
#
# SGLang signature: forward_cuda(self, hidden_states, router_logits, *,
#                                num_token_non_padded=None,
#                                expert_location_dispatch_info=None) -> TopKOutput
# Dispatch signature: fn(obj, hidden_states, router_logits, **kwargs) -> TopKOutput
# Mapping: pass-through (1:1)

from __future__ import annotations

from typing import Optional

import torch

from sglang_fl.dispatch import call_op


def topk_bridge(
    self,
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    *,
    num_token_non_padded: Optional[torch.Tensor] = None,
    expert_location_dispatch_info=None,
):
    """SGLang TopK forward → dispatch call_op("topk", ...)."""
    return call_op(
        "topk",
        self,
        hidden_states,
        router_logits,
        num_token_non_padded=num_token_non_padded,
        expert_location_dispatch_info=expert_location_dispatch_info,
    )
