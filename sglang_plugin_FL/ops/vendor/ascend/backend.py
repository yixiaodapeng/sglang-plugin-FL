"""Ascend (Huawei) vendor backend.

Uses torch_npu operators when available on Ascend NPU hardware.
"""

from sglang_plugin_FL.ops.vendor.base import VendorBackend


class AscendBackend(VendorBackend):
    @property
    def name(self) -> str:
        return "ascend"

    @property
    def vendor(self) -> str:
        return "ascend"

    def is_available(self) -> bool:
        try:
            from flag_gems.runtime.backend.device import DeviceDetector
            return DeviceDetector().vendor_name.lower() == "ascend"
        except Exception:
            try:
                import torch_npu  # noqa: F401
                return True
            except ImportError:
                return False

    def silu_and_mul(self, obj, x):
        """Ascend SiluAndMul via torch_npu.

        Args:
            obj: MultiPlatformOp instance
            x: input tensor
        """
        try:
            import torch_npu
            d = x.shape[-1] // 2
            return torch_npu.npu_swiglu(x[..., :d], x[..., d:])
        except (ImportError, AttributeError):
            import torch.nn.functional as F
            d = x.shape[-1] // 2
            return F.silu(x[..., :d]) * x[..., d:]

    def rms_norm(self, obj, x, residual=None, post_residual_addition=None):
        """Ascend RMSNorm via torch_npu.

        Args:
            obj: MultiPlatformOp instance (has .weight, .variance_epsilon, etc.)
            x: input tensor
        """
        import torch

        if x.numel() == 0:
            return x

        if residual is not None:
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            x = x + residual
            residual = x

        try:
            import torch_npu
            out = torch_npu.npu_rms_norm(x, obj.weight.data, epsilon=obj.variance_epsilon)[0]
        except (ImportError, AttributeError):
            variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
            out = (x.to(torch.float32) * torch.rsqrt(variance + obj.variance_epsilon)).to(
                obj.weight.dtype
            ) * obj.weight

        if residual is not None:
            return out, residual
        return out
