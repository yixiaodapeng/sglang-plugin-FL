# Bridge: GemmaRMSNorm
#
# SGLang signature:
#   forward_cuda(self, x, residual=None, post_residual_addition=None)
#     -> Tensor | tuple[Tensor, Tensor]
#
# Dispatch signature:
#   fn(obj, x, residual=None) -> Tensor | tuple[Tensor, Tensor]
#
# SGLang-specific handling:
#   - post_residual_addition: added to residual before passing to dispatch
#   - GemmaRMSNorm uses weight+1 semantics (handled by reference/vendor impls)

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from sglang_fl.dispatch import call_op


def gemma_rms_norm_bridge(
    self,
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    post_residual_addition: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """SGLang GemmaRMSNorm forward → dispatch call_op("gemma_rms_norm", ...)."""
    # Handle post_residual_addition: merge into residual
    if post_residual_addition is not None and residual is not None:
        residual = residual + post_residual_addition
    elif post_residual_addition is not None and residual is None:
        residual = post_residual_addition

    return call_op("gemma_rms_norm", self, x, residual)
