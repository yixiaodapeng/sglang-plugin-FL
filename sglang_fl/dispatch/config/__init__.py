# Dispatch config loader — platform auto-detection and YAML loading.

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent


def get_config_path() -> Optional[Path]:
    """
    Get the platform-specific config file path.

    Priority:
    1. SGLANG_FL_PLATFORM env var (force platform)
    2. Auto-detect via torch
    """
    forced = os.environ.get("SGLANG_FL_PLATFORM", "").strip().lower()
    if forced:
        path = _CONFIG_DIR / f"{forced}.yaml"
        if path.is_file():
            return path
        return None

    # Auto-detect
    try:
        import torch

        if hasattr(torch, "npu") and torch.npu.is_available():
            return _CONFIG_DIR / "ascend.yaml"
        if torch.cuda.is_available():
            return _CONFIG_DIR / "nvidia.yaml"
    except Exception:
        pass

    return None
