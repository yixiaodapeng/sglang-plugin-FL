# Dispatch module public API — aligned with vllm-plugin-FL.

from .types import OpImpl, BackendImplKind, BackendPriority, match_token
from .registry import OpRegistry, OpRegistrySnapshot
from .policy import (
    SelectionPolicy,
    PolicyManager,
    get_policy,
    set_global_policy,
    reset_global_policy,
    policy_context,
    policy_from_config,
    with_strict_mode,
    with_preference,
    with_allowed_vendors,
    with_denied_vendors,
    PREFER_DEFAULT,
    PREFER_VENDOR,
    PREFER_REFERENCE,
)
from .manager import OpManager, get_default_manager, reset_default_manager
from .ops import FLBackendBase
from .logger_manager import get_logger, set_log_level


def call_op(op_name: str, *args, **kwargs):
    """
    Convenience function to call an operator through the default manager.

    Args:
        op_name: Name of the operator (e.g., "silu_and_mul", "rms_norm")
        *args, **kwargs: Arguments passed to the operator

    Returns:
        Result from the operator implementation
    """
    return get_default_manager().call(op_name, *args, **kwargs)


def resolve_op(op_name: str):
    """
    Convenience function to resolve an operator through the default manager.

    Args:
        op_name: Name of the operator

    Returns:
        Callable implementation function
    """
    return get_default_manager().resolve(op_name)


__all__ = [
    "OpImpl",
    "BackendImplKind",
    "BackendPriority",
    "match_token",
    "OpRegistry",
    "OpRegistrySnapshot",
    "SelectionPolicy",
    "PolicyManager",
    "get_policy",
    "set_global_policy",
    "reset_global_policy",
    "policy_context",
    "policy_from_config",
    "with_strict_mode",
    "with_preference",
    "with_allowed_vendors",
    "with_denied_vendors",
    "PREFER_DEFAULT",
    "PREFER_VENDOR",
    "PREFER_REFERENCE",
    "OpManager",
    "get_default_manager",
    "reset_default_manager",
    "FLBackendBase",
    "get_logger",
    "set_log_level",
    "call_op",
    "resolve_op",
]
