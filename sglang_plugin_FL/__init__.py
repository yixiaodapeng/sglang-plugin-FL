"""SGLang OOT Plugin — FlagGems-based multi-chip adaptation.

Two entry_points are registered:
  1. Platform Plugin (sglang.srt.platforms): activate_platform()
     Provides PlatformFL — device identity, memory, dist backend, graph capture.
  2. General Plugin (sglang.srt.plugins): load_plugin()
     Registers ATen ops (FlagGems), fused kernel dispatch (AROUND hook),
     and communicator hooks.

Loading:
  SGLANG_PLUGINS=sglang_plugin_FL   (General Plugin whitelist)
  SGLANG_PLATFORM=sglang_plugin_FL  (Platform Plugin explicit selection)
  Or auto-discover (no env vars needed if only one platform plugin installed).

This plugin registers:
  Layer 1: FlagGems ATen operator replacement (flag_gems.enable)
  Layer 2: SGLang fused kernels via MultiPlatformOp.register_oot_forward +
           HookRegistry AROUND hook on dispatch_forward
  Layer 3: FlagCX communicator (via Platform Plugin get_communicator_class)

Environment variables:
  USE_FLAGGEMS=1|0              Master switch (default: 1)
  SGLANG_FLAGGEMS_MODE=all|only|off   ATen replacement mode (default: all)
  SGLANG_FLAGGEMS_INCLUDE=op1,op2     Ops to include (only mode)
  SGLANG_FLAGGEMS_EXCLUDE=op1,op2     Ops to exclude (all mode)
  SGLANG_FLAGGEMS_RECORD=1|0          Enable FlagGems logging (default: 0)
  SGLANG_FLAGGEMS_LOG_ONCE=1|0        Log only once per op (default: 1)
  SGLANG_FLAGGEMS_LOG_PATH=<path>     Log output path
  SGLANG_OOT_PREFER=flagos|vendor|reference  Global backend priority
  SGLANG_OOT_OP_PREFER=Op:kind[,...]        Per-op backend override
  SGLANG_OOT_WHITELIST=op1,op2       Only dispatch listed ops through OOT
  SGLANG_OOT_BLACKLIST=op1,op2       Skip listed ops from OOT dispatch
  SGLANG_OOT_DISPATCH_LOG=<path>      Dispatch log file path
  SGLANG_FL_CONFIG=<path>             YAML config file (overrides platform defaults)
  SGLANG_FL_DIST_BACKEND=nccl|hccl|flagcx   Override distributed backend
  FLAGCX_PATH=<path>                         If set, default to flagcx backend
"""

import logging
import multiprocessing
import os

import torch

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _is_rank0() -> bool:
    """Return True if this is the main process (rank 0) — safe to call at any stage."""
    # During load_plugin(): dist not initialized yet, use multiprocessing parent check
    if multiprocessing.parent_process() is not None:
        return False  # We are a spawned subprocess (tp_rank >= 1)
    return True


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_bool(val: str, default: bool = False) -> bool:
    return val.strip().lower() in ("1", "true", "yes") if val else default


def _parse_list(val: str) -> list:
    return [x.strip() for x in val.split(",") if x.strip()] if val else []


def _parse_op_prefer(val: str) -> dict:
    """Parse SGLANG_OOT_OP_PREFER="SiluAndMul:flagos,RMSNorm:vendor:ascend"
    Returns dict: {op_name -> key_string}
    e.g. {"SiluAndMul": "flagos", "RMSNorm": "vendor:ascend"}
    """
    result = {}
    for item in _parse_list(val):
        parts = item.split(":", 2)
        if len(parts) == 2:
            result[parts[0].strip()] = parts[1].strip()
        elif len(parts) == 3:
            result[parts[0].strip()] = f"{parts[1].strip()}:{parts[2].strip()}"
    return result



def _key_to_log_label(key: str) -> str:
    """Convert registry key to dispatch log label.
    "flagos"       -> "flagos(flagos)"
    "vendor:nvidia"-> "vendor(vendor.nvidia)"
    "reference"    -> "reference(reference)"
    """
    parts = key.split(":", 1)
    kind = parts[0]
    if kind == "vendor" and len(parts) == 2:
        return f"vendor(vendor.{parts[1]})"
    return f"{kind}({kind})"


# ─── Config builder ───────────────────────────────────────────────────────────

def _build_config() -> dict:
    """Build unified config: env vars > YAML (via SGLANG_FL_CONFIG) > defaults.

    Priority chain:
      env vars (SGLANG_OOT_*, SGLANG_FLAGGEMS_*)
        > YAML config (SGLANG_FL_CONFIG or platform auto-detect)
          > code defaults
    """
    from sglang_plugin_FL.config import load_config

    yaml_cfg = load_config()

    # prefer: SGLANG_OOT_PREFER > yaml.prefer > "flagos"
    prefer = (
        os.environ.get("SGLANG_OOT_PREFER", "").strip()
        or yaml_cfg.get("prefer", "")
        or "flagos"
    ).lower()

    # op_backends: SGLANG_OOT_OP_PREFER > yaml.op_backends > {}
    legacy_per_op = os.environ.get("SGLANG_OOT_OP_PREFER", "").strip()
    if legacy_per_op:
        parsed = _parse_op_prefer(legacy_per_op)
        op_backends = {k: [v] for k, v in parsed.items()}
    else:
        op_backends = yaml_cfg.get("op_backends", {})
        # Normalize: yaml values may be lists already or strings
        if isinstance(op_backends, dict):
            for k, v in op_backends.items():
                if isinstance(v, str):
                    op_backends[k] = [v]


    # oot_blacklist: SGLANG_OOT_BLACKLIST > yaml.oot_blacklist > []
    oot_blacklist_str = os.environ.get("SGLANG_OOT_BLACKLIST", "").strip()
    if oot_blacklist_str:
        oot_blacklist = _parse_list(oot_blacklist_str)
    else:
        oot_blacklist = yaml_cfg.get("oot_blacklist", []) or []

    oot_whitelist = _parse_list(os.environ.get("SGLANG_OOT_WHITELIST", ""))

    # flagos_blacklist: SGLANG_FLAGGEMS_EXCLUDE > yaml.flagos_blacklist > []
    flagos_bl_str = os.environ.get("SGLANG_FLAGGEMS_EXCLUDE", "").strip()
    if flagos_bl_str:
        flagos_blacklist = _parse_list(flagos_bl_str)
    else:
        flagos_blacklist = yaml_cfg.get("flagos_blacklist", []) or []

    # dispatch_log: SGLANG_OOT_DISPATCH_LOG (no change)
    dispatch_log = os.environ.get("SGLANG_OOT_DISPATCH_LOG", "").strip()

    # flaggems settings (unchanged env vars, no YAML mapping)
    flaggems_record = _parse_bool(os.environ.get("SGLANG_FLAGGEMS_RECORD", "0"))
    flaggems_log_path = os.environ.get("SGLANG_FLAGGEMS_LOG_PATH", "").strip()

    return {
        "prefer": prefer,
        "op_backends": op_backends,
        "oot_blacklist": oot_blacklist,
        "oot_whitelist": oot_whitelist,
        "flagos_blacklist": flagos_blacklist,
        "dispatch_log": dispatch_log,
        "flaggems_record": flaggems_record,
        "flaggems_log_path": flaggems_log_path,
    }


# ─── Priority resolution ──────────────────────────────────────────────────────

def _resolve_dispatch_key(op_name: str, registry: dict, registered_cls=None, config: dict = None) -> str | None:
    """Resolve the best available registry key for the given op.

    Priority chain (respects config["prefer"] and config["op_backends"]):
      1. Per-op override (config["op_backends"])
      2. Global preference (config["prefer"], default: flagos)
      3. Fallback: flagos → reference

    Returns the first key that exists in registry and whose _is_available()
    returns True (or None if not set). Returns None if nothing matches.
    """
    if config is None:
        # Fallback for direct calls without config (backward compat)
        per_op_env = _parse_op_prefer(os.environ.get("SGLANG_OOT_OP_PREFER", ""))
        global_prefer = os.environ.get("SGLANG_OOT_PREFER", "flagos").lower().strip()
        prefer = per_op_env.get(op_name, global_prefer)
        per_op_backends = None
    else:
        per_op_backends = config.get("op_backends", {})
        global_prefer = config.get("prefer", "flagos")
        # Check per-op backends (new format: {op_name: [list of backends]})
        if op_name in per_op_backends:
            # New format: explicit ordered list
            candidates = list(per_op_backends[op_name])
            prefer = None  # skip default logic
        else:
            prefer = global_prefer

    # Build ordered candidate list
    candidates_built = False
    if config is not None and per_op_backends and op_name in per_op_backends:
        # Expand "vendor" to actual vendor keys (e.g. "vendor:cuda")
        raw = list(per_op_backends[op_name])
        candidates = []
        for c in raw:
            if c == "vendor":
                vendor_keys = sorted(k for k in registry if k.startswith("vendor:"))
                candidates.extend(vendor_keys)
            else:
                candidates.append(c)
        candidates_built = True

    if not candidates_built:
        candidates = []
        if prefer.startswith("vendor:"):
            candidates.append(prefer)
            candidates.append("flagos")
            candidates.append("reference")
        elif prefer == "vendor":
            vendor_keys = sorted(k for k in registry if k.startswith("vendor:"))
            candidates.extend(vendor_keys)
            candidates.append("flagos")
            candidates.append("reference")
        elif prefer == "reference":
            candidates.extend(["reference", "flagos"])
        else:
            # "flagos" or unknown → flagos first
            candidates.extend(["flagos", "reference"])

    # Resolve op_cls: use provided registered_cls, or look up by name
    if registered_cls is not None:
        op_cls = registered_cls
    else:
        from sglang.srt.layers.activation import SiluAndMul
        from sglang.srt.layers.layernorm import RMSNorm
        from sglang.srt.layers.rotary_embedding import RotaryEmbedding
        _name_to_cls = {"SiluAndMul": SiluAndMul, "RMSNorm": RMSNorm, "RotaryEmbedding": RotaryEmbedding}
        op_cls = _name_to_cls.get(op_name)

    for key in candidates:
        if key not in registry:
            continue
        if op_cls is None or op_cls not in registry[key]:
            continue
        fn = registry[key][op_cls]
        is_avail = getattr(fn, "_is_available", None)
        if is_avail is not None and not is_avail():
            continue
        return key

    return None


# ─── Dispatch AROUND hook ────────────────────────────────────────────────────

def _make_dispatch_hook(config: dict = None):
    """Build the AROUND hook for MultiPlatformOp.dispatch_forward.

    The hook intercepts dispatch_forward() for ops that have OOT
    implementations registered. For all other ops it falls through to
    the original SGLang dispatch logic (CUDA/HIP/etc.).

    Respects config["oot_whitelist"] / config["oot_blacklist"]:
      - WHITELIST: only listed ops use OOT dispatch (comma-separated class names)
      - BLACKLIST: listed ops skip OOT dispatch (comma-separated class names)
      - Cannot set both simultaneously.
      - Empty (default): all registered ops use OOT dispatch.
    """
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

    if config is None:
        dispatch_log_path = os.environ.get("SGLANG_OOT_DISPATCH_LOG", "").strip()
        whitelist = _parse_list(os.environ.get("SGLANG_OOT_WHITELIST", ""))
        blacklist = _parse_list(os.environ.get("SGLANG_OOT_BLACKLIST", ""))
    else:
        dispatch_log_path = config.get("dispatch_log", "")
        whitelist = config.get("oot_whitelist", [])
        blacklist = config.get("oot_blacklist", [])

    _log_file = open(dispatch_log_path, "a") if dispatch_log_path else None
    if whitelist and blacklist:
        raise ValueError(
            "Cannot set both SGLANG_OOT_WHITELIST and SGLANG_OOT_BLACKLIST. "
            "Use one or the other."
        )
    if whitelist:
        logger.info(f"OOT dispatch whitelist: {whitelist}")
    if blacklist:
        logger.info(f"OOT dispatch blacklist: {blacklist}")

    def _dispatch_hook(original_fn, self):
        op_cls = type(self)
        op_name = op_cls.__name__
        registry = MultiPlatformOp._oot_forward_registry

        # P1: Whitelist/blacklist gate
        if whitelist and op_name not in whitelist:
            return original_fn(self)
        if blacklist and op_name in blacklist:
            return original_fn(self)

        # Only intercept ops that have at least one OOT registration
        # Walk MRO to support subclasses (e.g. Llama3RotaryEmbedding → RotaryEmbedding)
        def _find_registered_cls(cls):
            for parent in cls.__mro__:
                if any(parent in v for v in registry.values()):
                    return parent
            return None

        registered_cls = _find_registered_cls(op_cls)
        if registered_cls is None:
            return original_fn(self)

        key = _resolve_dispatch_key(op_name, registry, registered_cls=registered_cls, config=config)
        if key is not None and key in registry and registered_cls in registry[key]:
            if _log_file:
                _log_file.write(f"[OOT-DISPATCH] {op_name} → {_key_to_log_label(key)}\n")
                _log_file.flush()
            return registry[key][registered_cls].__get__(self)

        # Registered but no available backend found — fall through
        return original_fn(self)

    return _dispatch_hook


# ─── FlagGems setup ───────────────────────────────────────────────────────────

def _setup_flaggems(config: dict = None):
    use_fg = _parse_bool(os.environ.get("USE_FLAGGEMS", "1"), default=True)
    fg_mode = os.environ.get("SGLANG_FLAGGEMS_MODE", "all").strip().lower()

    if not use_fg:
        logger.info("FlagGems disabled (USE_FLAGGEMS=0)")
        return
    if fg_mode == "off":
        logger.info("FlagGems ATen replacement skipped (SGLANG_FLAGGEMS_MODE=off)")
        return

    import flag_gems

    if config is not None:
        record = config.get("flaggems_record", False)
        log_path = config.get("flaggems_log_path", "")
        exclude = config.get("flagos_blacklist", [])
    else:
        record = _parse_bool(os.environ.get("SGLANG_FLAGGEMS_RECORD", "0"))
        log_path = os.environ.get("SGLANG_FLAGGEMS_LOG_PATH", "").strip()
        exclude = _parse_list(os.environ.get("SGLANG_FLAGGEMS_EXCLUDE", ""))

    fg_kwargs = dict(
        record=record,
        once=_parse_bool(os.environ.get("SGLANG_FLAGGEMS_LOG_ONCE", "1"), default=True),
    )
    if log_path:
        fg_kwargs["path"] = log_path

    if fg_mode == "only":
        include = _parse_list(os.environ.get("SGLANG_FLAGGEMS_INCLUDE", ""))
        if not include:
            logger.warning(
                "SGLANG_FLAGGEMS_MODE=only but SGLANG_FLAGGEMS_INCLUDE is empty. "
                "No ATen ops will be replaced."
            )
        else:
            flag_gems.only_enable(include=include, **fg_kwargs)
            logger.info(f"FlagGems only_enable: {include}")
    else:
        if exclude:
            flag_gems.enable(unused=exclude, **fg_kwargs)
            logger.info(f"FlagGems enable (excluding: {exclude})")
        else:
            flag_gems.enable(**fg_kwargs)
            logger.info("FlagGems enable: ALL ATen ops replaced with Triton kernels")

    # Filter gems_aten log to only record Layer 1 (ATen dispatch) calls.
    # Exclude Layer 2 flagos calls (flag_gems.fused.*, flag_gems.modules.*)
    # so the log purely reflects ATen operator replacement.
    if fg_kwargs.get("record"):
        import logging as _logging

        class _AtenOnlyFilter(_logging.Filter):
            """Only pass log records from flag_gems.ops.* (ATen replacements)."""
            def filter(self, record):
                return record.name.startswith("flag_gems.ops")

        fg_logger = _logging.getLogger("flag_gems")
        for h in fg_logger.handlers:
            if getattr(h, "_flaggems_owned", False):
                h.addFilter(_AtenOnlyFilter())


# ─── Op registration ──────────────────────────────────────────────────────────

def _register_ops():
    """Register FlagGems + reference + vendor implementations into
    MultiPlatformOp._oot_forward_registry via register_oot_forward().
    """
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp
    from sglang.srt.layers.activation import SiluAndMul
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding
    from sglang_plugin_FL.ops import (
        silu_and_mul_flaggems,
        rms_norm_flaggems,
        rotary_embedding_flaggems,
        silu_and_mul_reference,
        rms_norm_reference,
        rotary_embedding_reference,
    )

    # FlagOS backend (FlagGems Triton kernels) — highest default priority
    MultiPlatformOp.register_oot_forward(SiluAndMul, silu_and_mul_flaggems, "flagos")
    MultiPlatformOp.register_oot_forward(RMSNorm, rms_norm_flaggems, "flagos")
    MultiPlatformOp.register_oot_forward(RotaryEmbedding, rotary_embedding_flaggems, "flagos")

    # Reference backend (PyTorch native) — lowest priority, fallback
    MultiPlatformOp.register_oot_forward(SiluAndMul, silu_and_mul_reference, "reference")
    MultiPlatformOp.register_oot_forward(RMSNorm, rms_norm_reference, "reference")
    MultiPlatformOp.register_oot_forward(RotaryEmbedding, rotary_embedding_reference, "reference")

    # Vendor backends (auto-discovered from ops/vendor/ subdirectories)
    from sglang_plugin_FL.ops.vendor import discover_and_register_vendors
    discover_and_register_vendors()


# ─── Communicator AROUND hooks ────────────────────────────────────────────────

def _setup_communicator_hooks():
    """Register AROUND hooks on GroupCoordinator to inject CommunicatorFL.

    Auto-activates when the Platform Plugin is OOT. The CommunicatorFL
    transparently routes through FlagCX (if available) or torch.distributed.

    Hooks:
      - __init__: inject self.fl_communicator after original init
      - all_reduce, reduce_scatter_tensor, all_gather_into_tensor,
        reduce_scatterv, all_gatherv, send, recv: delegate to fl_communicator
    """
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    _GC_TARGET = "sglang.srt.distributed.parallel_state.GroupCoordinator"

    # ── __init__ hook: create and attach CommunicatorFL ──

    def _init_hook(original_fn, self, *args, **kwargs):
        # Run original __init__ first
        original_fn(self, *args, **kwargs)
        # Attach CommunicatorFL if world_size > 1
        if getattr(self, 'world_size', 1) > 1:
            try:
                from sglang_plugin_FL.communicator import CommunicatorFL
                self.fl_communicator = CommunicatorFL(
                    cpu_group=self.cpu_group,
                    device=self.device,
                    device_group=self.device_group,
                    world_size=self.world_size,
                    rank_in_group=self.rank_in_group,
                    ranks=self.ranks,
                )
            except Exception as e:
                logger.warning(f"CommunicatorFL creation failed: {e}")
                self.fl_communicator = None
        else:
            self.fl_communicator = None

    # ── all_reduce hook ──

    def _all_reduce_hook(original_fn, self, input_):
        comm = getattr(self, 'fl_communicator', None)
        if comm is not None and not comm.disabled:
            return comm.all_reduce(input_)
        return original_fn(self, input_)

    # ── reduce_scatter_tensor hook ──

    def _reduce_scatter_tensor_hook(original_fn, self, output, input_):
        comm = getattr(self, 'fl_communicator', None)
        if comm is not None and not comm.disabled:
            comm.reduce_scatter(output, input_)
            return
        return original_fn(self, output, input_)

    # ── all_gather_into_tensor hook ──

    def _all_gather_into_tensor_hook(original_fn, self, output, input_):
        comm = getattr(self, 'fl_communicator', None)
        if comm is not None and not comm.disabled:
            comm.all_gather(output, input_)
            return
        return original_fn(self, output, input_)

    # ── reduce_scatterv hook ──

    def _reduce_scatterv_hook(original_fn, self, input_, output=None, sizes=None):
        comm = getattr(self, 'fl_communicator', None)
        if comm is not None and not comm.disabled:
            return comm.reduce_scatterv(input_, output=output, sizes=sizes)
        return original_fn(self, input_, output=output, sizes=sizes)

    # ── all_gatherv hook ──

    def _all_gatherv_hook(original_fn, self, input_, sizes=None):
        comm = getattr(self, 'fl_communicator', None)
        if comm is not None and not comm.disabled:
            return comm.all_gatherv(input_, sizes=sizes)
        return original_fn(self, input_, sizes=sizes)

    # ── send hook ──

    def _send_hook(original_fn, self, tensor, dst=None):
        comm = getattr(self, 'fl_communicator', None)
        if comm is not None and not comm.disabled:
            if dst is None:
                dst = (self.rank_in_group + 1) % self.world_size
            comm.send(tensor, dst)
            return
        return original_fn(self, tensor, dst=dst)

    # ── recv hook ──

    def _recv_hook(original_fn, self, size, dtype, src=None):
        comm = getattr(self, 'fl_communicator', None)
        if comm is not None and not comm.disabled:
            if src is None:
                src = (self.rank_in_group - 1) % self.world_size
            tensor = torch.empty(size, dtype=dtype, device=self.device)
            comm.recv(tensor, src)
            return tensor
        return original_fn(self, size, dtype, src=src)

    # ── Register all hooks ──

    HookRegistry.register(f"{_GC_TARGET}.__init__", _init_hook, HookType.AROUND)
    HookRegistry.register(f"{_GC_TARGET}.all_reduce", _all_reduce_hook, HookType.AROUND)
    HookRegistry.register(f"{_GC_TARGET}._reduce_scatter_tensor", _reduce_scatter_tensor_hook, HookType.AROUND)
    HookRegistry.register(f"{_GC_TARGET}._all_gather_into_tensor", _all_gather_into_tensor_hook, HookType.AROUND)
    HookRegistry.register(f"{_GC_TARGET}.reduce_scatterv", _reduce_scatterv_hook, HookType.AROUND)
    HookRegistry.register(f"{_GC_TARGET}.all_gatherv", _all_gatherv_hook, HookType.AROUND)
    HookRegistry.register(f"{_GC_TARGET}.send", _send_hook, HookType.AROUND)
    HookRegistry.register(f"{_GC_TARGET}.recv", _recv_hook, HookType.AROUND)

    logger.info("CommunicatorFL AROUND hooks registered on GroupCoordinator")


# ─── Platform Plugin entry point ─────────────────────────────────────────────

def activate_platform() -> str | None:
    """Entry point for sglang.srt.platforms — called during platform discovery.

    Returns the fully-qualified class path of PlatformFL if hardware is detected,
    or None if FlagGems DeviceDetector fails (no supported hardware).
    """
    try:
        from flag_gems.runtime.backend.device import DeviceDetector
        detector = DeviceDetector()
        logger.info(
            "sglang_plugin_FL platform activating: vendor=%s, device=%s",
            detector.vendor_name, detector.name,
        )
        return "sglang_plugin_FL.platform:PlatformFL"
    except Exception as e:
        logger.warning("sglang_plugin_FL platform activation failed: %s", e)
        return None


# ─── General Plugin entry point ──────────────────────────────────────────────

_plugin_loaded = False


def load_plugin():
    """Entry point for sglang.srt.plugins — called by SGLang's load_plugins().

    Also invoked directly when SGLANG_OOT_PLUGINS=sglang_plugin_FL (legacy).
    Idempotent: safe to call multiple times.
    """
    global _plugin_loaded
    if _plugin_loaded:
        return
    _plugin_loaded = True

    # Suppress info logs on non-rank-0 processes to avoid duplicate output
    if not _is_rank0():
        logger.setLevel(logging.WARNING)

    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    # 0. Build unified config (YAML + env vars)
    config = _build_config()

    # 1. FlagGems ATen ops
    _setup_flaggems(config)

    # 2. Register OOT op implementations into MultiPlatformOp registry
    _register_ops()

    # 3. Install dispatch AROUND hook (priority: flagos > vendor > reference)
    HookRegistry.register(
        "sglang.srt.layers.utils.multi_platform.MultiPlatformOp.dispatch_forward",
        _make_dispatch_hook(config),
        HookType.AROUND,
    )

    # 4. Communicator hooks (CommunicatorFL with FlagCX/torch.distributed)
    _setup_communicator_hooks()

    # 5. Summary banner — confirm plugin is active (rank 0 only)
    if _is_rank0():
        fg_mode = os.environ.get("SGLANG_FLAGGEMS_MODE", "all").strip().lower()
        use_fg = _parse_bool(os.environ.get("USE_FLAGGEMS", "1"), default=True)
        aten_status = "OFF" if (not use_fg or fg_mode == "off") else f"ON (mode={fg_mode})"
        dist_backend = os.environ.get("SGLANG_FL_DIST_BACKEND", "").strip() or (
            "flagcx" if os.environ.get("FLAGCX_PATH", "").strip() else "nccl"
        )
        banner = (
            "\n"
            "+" + "=" * 58 + "+\n"
            "|" + "sglang_plugin_FL activated".center(58) + "|\n"
            "+" + "-" * 58 + "+\n"
            "|" + f"  Layer 1 (ATen -> FlagGems):  {aten_status}".ljust(58) + "|\n"
            "|" + f"  Layer 2 (Fused Ops):         prefer={config['prefer']}".ljust(58) + "|\n"
            "|" + f"  Layer 3 (Communication):     {dist_backend}".ljust(58) + "|\n"
            "+" + "=" * 58 + "+"
        )
        logger.info(banner)


# ─── Legacy SGLANG_OOT_PLUGINS compatibility ──────────────────────────────────
# When loaded via old SGLANG_OOT_PLUGINS env var (pre-entry_points mechanism),
# the module is imported directly. Detect this and call load_plugin() ourselves.
# Users should migrate to: SGLANG_PLUGINS=sglang_plugin_FL
if "sglang_plugin_FL" in os.environ.get("SGLANG_OOT_PLUGINS", ""):
    load_plugin()
