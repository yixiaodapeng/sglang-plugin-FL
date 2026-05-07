"""CUDA (NVIDIA) vendor backend.

Uses SGLang's native CUDA kernels via sgl_kernel when available,
falls back to PyTorch native ops otherwise.
"""

from sglang_plugin_FL.ops.vendor.base import VendorBackend


class CudaBackend(VendorBackend):
    @property
    def name(self) -> str:
        return "cuda"

    @property
    def vendor(self) -> str:
        return "nvidia"

    def is_available(self) -> bool:
        try:
            from flag_gems.runtime.backend.device import DeviceDetector
            return DeviceDetector().vendor_name.lower() == "nvidia"
        except Exception:
            try:
                import torch
                return torch.cuda.is_available()
            except Exception:
                return False

    def silu_and_mul(self, obj, x):
        """CUDA SiluAndMul — try sgl_kernel, fallback to PyTorch.

        Args:
            obj: MultiPlatformOp instance (the nn.Module with weight/config)
            x: input tensor
        """
        try:
            from sgl_kernel import silu_and_mul as _silu_and_mul
            return _silu_and_mul(x)
        except ImportError:
            import torch.nn.functional as F
            d = x.shape[-1] // 2
            return F.silu(x[..., :d]) * x[..., d:]

    def rms_norm(self, obj, x, residual=None, post_residual_addition=None):
        """CUDA RMSNorm — try sgl_kernel, fallback to PyTorch.

        Args:
            obj: MultiPlatformOp instance (has .weight, .variance_epsilon, etc.)
            x: input tensor
        """
        import torch

        if x.numel() == 0:
            return x

        if obj.variance_size_override is not None:
            return obj.forward_native(x, residual, post_residual_addition)

        try:
            from sgl_kernel import rmsnorm as _rmsnorm
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                x = x + residual
                residual = x
            out = _rmsnorm(x, obj.weight.data, obj.variance_epsilon)
            if residual is not None:
                return out, residual
            return out
        except ImportError:
            if residual is not None:
                if post_residual_addition is not None:
                    residual = residual + post_residual_addition
                x = x + residual
                residual = x
            variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
            x = x * torch.rsqrt(variance + obj.variance_epsilon)
            out = x.to(obj.weight.dtype) * obj.weight
            if residual is not None:
                return out, residual
            return out
