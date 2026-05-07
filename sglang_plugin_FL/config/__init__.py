"""Config loader — YAML-based dispatch configuration."""
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
_CONFIG_DIR = Path(__file__).parent


def get_platform_config_path() -> Optional[Path]:
    """Auto-detect platform and return config file path."""
    import torch

    if hasattr(torch, "npu") and torch.npu.is_available():
        return _CONFIG_DIR / "ascend.yaml"
    if torch.cuda.is_available():
        return _CONFIG_DIR / "cuda.yaml"
    return None


def load_config() -> dict:
    """Load config with priority: SGLANG_FL_CONFIG > platform auto-detect > empty."""
    try:
        import yaml
    except ImportError:
        logger.debug("pyyaml not installed, skipping YAML config")
        return {}

    user_path = os.environ.get("SGLANG_FL_CONFIG", "").strip()
    if user_path:
        path = Path(user_path)
    else:
        path = get_platform_config_path()

    if path and path.is_file():
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
        logger.info(f"Loaded dispatch config from {path}")
        return cfg
    return {}
