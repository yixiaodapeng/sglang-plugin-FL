# Glue: SiluAndMul
#
# SGLang signature: forward_cuda(self, x: Tensor) -> Tensor
# Dispatch signature: fn(obj, x: Tensor) -> Tensor
# Mapping: trivial (1:1)

from __future__ import annotations

import torch

from sglang_fl.dispatch import call_op


def silu_and_mul_glue(self, x: torch.Tensor) -> torch.Tensor:
    """SGLang SiluAndMul forward → dispatch call_op("silu_and_mul", ...)."""
    return call_op("silu_and_mul", self, x)
