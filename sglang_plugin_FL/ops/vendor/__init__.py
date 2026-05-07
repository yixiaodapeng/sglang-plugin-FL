"""Vendor backend auto-discovery and registration.

Scans vendor/ subdirectories for backend.py modules, instantiates each
VendorBackend subclass, and registers available operator implementations
into the MultiPlatformOp multi-backend registry.
"""

import importlib
import logging
import os
import pkgutil
import types
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_device_vendor() -> str:
    """Detect current chip vendor via flag_gems DeviceDetector."""
    try:
        from flag_gems.runtime.backend.device import DeviceDetector
        detector = DeviceDetector()
        return detector.vendor_name.lower()
    except Exception:
        return "unknown"


def _bind_is_available(fn, backend):
    """Wrap an operator function with _is_available from its backend.

    The dispatch layer calls wrapper(multiplatform_op, *args) via types.MethodType.
    Since fn is a bound method of backend, we pass multiplatform_op (the nn.Module
    instance with weight/epsilon) as the first arg, and forward the rest.
    """
    def wrapper(self, *args, **kwargs):
        # self = MultiPlatformOp instance (bound by types.MethodType in dispatch_forward)
        # fn = backend.method (bound method, self=backend)
        # Pass MultiPlatformOp instance so backend can access weight/epsilon/etc.
        return fn(self, *args, **kwargs)

    wrapper._is_available = backend.is_available
    wrapper.__name__ = getattr(fn, "__name__", "vendor_op")
    wrapper.__qualname__ = getattr(fn, "__qualname__", "vendor_op")
    return wrapper


# Mapping from method names on VendorBackend to op_names in MultiPlatformOp
_OP_METHOD_MAP = {
    "silu_and_mul": "SiluAndMul",
    "rms_norm": "RMSNorm",
}


def discover_and_register_vendors():
    """Scan vendor/ subdirectories, instantiate backends, register ops.

    Uses MultiPlatformOp.register_oot_forward(op_cls, fn, "vendor:{vendor}")
    so that the priority dispatch hook in __init__.py can select between
    flagos / vendor / reference backends at dispatch_forward() time.
    """
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm

    _op_cls_map = {"SiluAndMul": SiluAndMul, "RMSNorm": RMSNorm}

    vendor_dir = Path(__file__).parent
    registered = 0

    for entry in sorted(vendor_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(("_", ".")):
            continue
        if entry.name == "template":
            continue

        backend_module_path = entry / "backend.py"
        if not backend_module_path.exists():
            continue

        module_name = f"sglang_plugin_FL.ops.vendor.{entry.name}.backend"
        try:
            mod = importlib.import_module(module_name)
        except Exception as e:
            logger.warning(f"Failed to import vendor backend {module_name}: {e}")
            continue

        from sglang_plugin_FL.ops.vendor.base import VendorBackend
        backend_cls = None
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if (
                isinstance(attr, type)
                and issubclass(attr, VendorBackend)
                and attr is not VendorBackend
            ):
                backend_cls = attr
                break

        if backend_cls is None:
            logger.warning(f"No VendorBackend subclass found in {module_name}")
            continue

        try:
            backend = backend_cls()
        except Exception as e:
            logger.warning(f"Failed to instantiate {backend_cls.__name__}: {e}")
            continue

        # Registry key: "vendor:{vendor_name}" e.g. "vendor:nvidia", "vendor:mock_npu"
        registry_key = f"vendor:{backend.vendor}"

        for method_name, op_name in _OP_METHOD_MAP.items():
            method = getattr(backend, method_name, None)
            if method is None:
                continue

            op_cls = _op_cls_map.get(op_name)
            if op_cls is None:
                continue

            wrapped = _bind_is_available(method, backend)
            MultiPlatformOp.register_oot_forward(op_cls, wrapped, registry_key)
            registered += 1

        logger.info(
            f"Vendor backend '{backend.name}' (vendor={backend.vendor}, "
            f"available={backend.is_available()}) registered as key='{registry_key}'"
        )

    logger.info(f"Vendor auto-discovery complete: {registered} ops registered")
