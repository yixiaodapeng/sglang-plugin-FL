"""Template vendor backend — copy this directory to add a new chip vendor.

Steps:
1. Copy this directory: cp -r ops/vendor/template/ ops/vendor/my_chip/
2. Rename the class and fill in name/vendor/is_available
3. Implement operator methods (silu_and_mul, rms_norm, etc.)
4. The auto-discovery in vendor/__init__.py will pick it up automatically

NOTE: All operator methods receive (self, obj, ...) where:
  - self: the VendorBackend instance
  - obj:  the MultiPlatformOp nn.Module instance (has .weight, .variance_epsilon, etc.)
"""

from sglang_plugin_FL.ops.vendor.base import VendorBackend


class TemplateBackend(VendorBackend):
    @property
    def name(self) -> str:
        return "template"  # Change to your backend name

    @property
    def vendor(self) -> str:
        return "template"  # Change to your vendor identifier

    def is_available(self) -> bool:
        # Replace with actual hardware detection logic
        return False

    def silu_and_mul(self, obj, x):
        """SiluAndMul — replace with vendor-specific implementation."""
        import torch.nn.functional as F
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def rms_norm(self, obj, x, residual=None, post_residual_addition=None):
        """RMSNorm — replace with vendor-specific implementation."""
        import torch

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
