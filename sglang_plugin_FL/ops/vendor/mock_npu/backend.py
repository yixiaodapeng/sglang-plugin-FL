"""Mock NPU vendor backend — simulates a domestic chip vendor on NVIDIA hardware.

This backend is for testing the vendor auto-discovery and dispatch mechanism
without real domestic hardware. It uses PyTorch native ops so it can run on
any GPU (including NVIDIA H20).

Usage:
    # Force dispatch to mock_npu vendor
    SGLANG_OOT_PREFER=vendor bash validate.sh plugin

    # Or per-op override
    SGLANG_OOT_OP_PREFER=SiluAndMul:vendor:mock_npu bash validate.sh plugin

    # Disable mock_npu (simulate hardware not available)
    SGLANG_MOCK_NPU_AVAILABLE=0 bash validate.sh plugin

    # Enable runtime execution logging
    SGLANG_MOCK_NPU_LOG=1 bash validate.sh vendor
"""

import logging
import os

import torch
import torch.nn.functional as F

from sglang_plugin_FL.ops.vendor.base import VendorBackend

logger = logging.getLogger(__name__)

_RUNTIME_LOG_PATH = os.environ.get("SGLANG_MOCK_NPU_LOG", "").strip() or None


class MockNpuBackend(VendorBackend):
    def __init__(self):
        self._logged = {"silu_and_mul": False, "rms_norm": False}

    @property
    def name(self) -> str:
        return "mock_npu"

    @property
    def vendor(self) -> str:
        return "mock_npu"

    def is_available(self) -> bool:
        """Only available when explicitly enabled via SGLANG_MOCK_NPU_AVAILABLE=1."""
        return os.environ.get("SGLANG_MOCK_NPU_AVAILABLE", "0").strip() in ("1", "true", "yes")

    def _log_runtime(self, op_name: str, x: torch.Tensor):
        if _RUNTIME_LOG_PATH and not self._logged[op_name]:
            self._logged[op_name] = True
            try:
                with open(_RUNTIME_LOG_PATH, "a") as f:
                    f.write(f"[MOCK_NPU_RUNTIME] {op_name} executed (shape={list(x.shape)}, dtype={x.dtype})\n")
            except Exception:
                pass

    def silu_and_mul(self, obj, x):
        """SiluAndMul — PyTorch native implementation (simulates vendor kernel)."""
        self._log_runtime("silu_and_mul", x)
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def rms_norm(self, obj, x, residual=None, post_residual_addition=None):
        """RMSNorm — PyTorch native implementation (simulates vendor kernel)."""
        self._log_runtime("rms_norm", x)

        if x.numel() == 0:
            return x

        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            x = x + residual
            residual = x

        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        out = (x.to(torch.float32) * torch.rsqrt(variance + obj.variance_epsilon)).to(
            obj.weight.dtype
        ) * obj.weight

        if residual is not None:
            return out, residual
        return out
