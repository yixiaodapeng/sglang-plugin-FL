# sglang_plugin_FL

sglang_plugin_FL is an out-of-tree (OOT) plugin for [SGLang](https://github.com/sgl-project/sglang), built on FlagOS's unified multi-chip backend — including the unified operator library [FlagGems](https://github.com/flagos-ai/FlagGems) and the unified communication library [FlagCX](https://github.com/flagos-ai/FlagCX). It extends SGLang's inference capabilities across diverse hardware platforms. Without changing SGLang's original interfaces or usage patterns, the same command can run model inference on different chips.

## Overview

SGLang's inference engine relies on NVIDIA-specific components: flashinfer for attention, sgl_kernel for fused CUDA kernels, and NCCL for distributed communication. Running on alternative hardware (Huawei Ascend, Cambricon MLU, Iluvatar, etc.) would otherwise require invasive source modifications.

This plugin provides a non-intrusive adaptation layer through three levels of replacement:

- **Layer 1 — ATen Operators**: Replaces PyTorch's low-level ops (matmul, softmax, embedding, etc.) with FlagGems Triton kernels via PyTorch's dispatch mechanism
- **Layer 2 — SGLang Fused Kernels**: Intercepts SGLang's custom fused ops (SiluAndMul, RMSNorm, RotaryEmbedding) via HookRegistry AROUND hooks, routing to FlagGems or vendor-native implementations
- **Layer 3 — Distributed Communication**: Replaces NCCL-based collectives with CommunicatorFL (backed by FlagCX or torch.distributed), enabling multi-card inference on any hardware

```
┌──────────────────────────────────────────────────────────────┐
│                       SGLang Runtime                         │
├──────────────────────────────────────────────────────────────┤
│  Layer 2: SGLang Fused Ops (AROUND hook on dispatch_forward) │
│    SiluAndMul / RMSNorm / RotaryEmbedding                    │
│      → flagos (FlagGems Triton) | vendor (chip-native) | ref │
├──────────────────────────────────────────────────────────────┤
│  Layer 1: ATen Ops (flag_gems.enable → PyTorch dispatch)     │
│    torch.mm / torch.add / torch.softmax / ...                │
│      → FlagGems Triton kernels                               │
├──────────────────────────────────────────────────────────────┤
│  Layer 3: Communication (AROUND hooks on GroupCoordinator)   │
│    all_reduce / all_gather / reduce_scatter / send / recv    │
│      → CommunicatorFL (FlagCX / torch.distributed)           │
├──────────────────────────────────────────────────────────────┤
│  Triton JIT / Vendor Native → GPU / NPU Kernels              │
└──────────────────────────────────────────────────────────────┘
```

Chip vendors only need to implement a single `backend.py` file. The plugin's auto-discovery mechanism handles the rest.

## Verified Models

In theory, sglang_plugin_FL can support all models available in SGLang, as long as no unsupported operators are involved.

| Model | TP | Status |
|-------|-----|--------|
| Qwen2.5-0.5B-Instruct | tp=1 | Verified |
| Qwen2.5-14B-Instruct | tp=8 | Verified |

## Quick Start

### Setup

1. Install SGLang from the official repository (or from [sglang-FL](https://github.com/flagos-ai/sglang-FL) fork):

```bash
cd sglang/python && pip install .
```

2. Install [FlagGems](https://github.com/flagos-ai/FlagGems):

```bash
git clone https://github.com/flagos-ai/FlagGems
cd FlagGems && pip install .
```

3. Install this plugin:

```bash
git clone https://github.com/flagos-ai/sglang_plugin_FL
cd sglang_plugin_FL && pip install .
```

4. (Optional) Install [FlagCX](https://github.com/flagos-ai/FlagCX) for multi-chip distributed communication:

```bash
git clone https://github.com/flagos-ai/FlagCX.git
cd FlagCX && make USE_NVIDIA=1
export FLAGCX_PATH="$PWD"
```

### Download Models

```bash
# Small model for quick testing (single GPU)
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct

# Larger model for multi-GPU (tp=8)
huggingface-cli download Qwen/Qwen2.5-14B-Instruct
```

If HuggingFace is not accessible, use a mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct
```

Models are cached in `~/.cache/huggingface/hub/` by default. You can also pass a local path to `--model-path`.

### Run a Task

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 \
    --disable-piecewise-cuda-graph
```

Multi-GPU with tensor parallelism:

```bash
python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --tp 8 --port 30000 \
    --disable-piecewise-cuda-graph
```

After the server is ready (`The server is fired up and ready to roll`), send a request:

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "List the first 5 prime numbers."}],
    "temperature": 0
  }' | python -m json.tool
```

### Using Native CUDA Operators

To disable the plugin and use SGLang's original CUDA path:

```bash
SGLANG_PLUGINS="__none__" python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

To disable only the ATen layer (keep fused op dispatch):

```bash
USE_FLAGGEMS=0 python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

## Advanced Configuration

For most use cases, the defaults work out of the box. When you need to customize, use a YAML config file.

### YAML Config File

The plugin ships a [`config/sample.yaml`](sglang_plugin_FL/config/sample.yaml) with all available options. Copy it and customize:

```bash
# Copy the sample config
cp $(python -c "from sglang_plugin_FL.config import _CONFIG_DIR; print(_CONFIG_DIR / 'sample.yaml')") my_config.yaml

# Edit as needed, then launch with it
SGLANG_FL_CONFIG=./my_config.yaml python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

If `SGLANG_FL_CONFIG` is not set, the plugin uses sensible defaults (equivalent to `prefer: flagos` on CUDA). You only need a YAML file when you want to customize behavior.

#### Config fields

```yaml
# Global backend preference: flagos | vendor | reference
prefer: flagos

# Per-op backend priority (ordered list, first available wins)
op_backends:
  RMSNorm: [vendor, flagos, reference]
  SiluAndMul: [flagos, vendor, reference]

# Layer 2 fused ops to skip (fall through to SGLang native CUDA)
# Available: SiluAndMul, RMSNorm, RotaryEmbedding
oot_blacklist:
  - RotaryEmbedding

# Layer 1 ATen ops to exclude from FlagGems Triton replacement
flagos_blacklist:
  - mul
  - sub
```

| Field | Description |
|-------|-------------|
| `prefer` | Global backend preference: `flagos`, `vendor`, `reference` |
| `op_backends` | Per-op ordered backend list (first available wins, can list 1–3 backends) |
| `oot_blacklist` | Layer 2 fused ops to skip from OOT dispatch (fall through to SGLang native CUDA) |
| `flagos_blacklist` | Layer 1 ATen ops to exclude from FlagGems replacement (fall through to PyTorch native) |

#### Common recipes

Each recipe shows a YAML config and expected dispatch result. Use [Dispatch Log](#dispatch-log) to verify.

**1. Skip RotaryEmbedding from OOT dispatch** (fall through to SGLang native CUDA):

```yaml
# my_config.yaml
prefer: flagos
oot_blacklist:
  - RotaryEmbedding
```

Expected dispatch log: only SiluAndMul and RMSNorm appear, no RotaryEmbedding.

**2. Force RMSNorm to use vendor backend, others use flagos:**

```yaml
# my_config.yaml
prefer: flagos
op_backends:
  RMSNorm: [vendor, flagos, reference]
```

Expected dispatch log: `RMSNorm → vendor(vendor.nvidia)`, `SiluAndMul → flagos(flagos)`.

**3. Use pure PyTorch reference for all ops** (useful for precision debugging):

```yaml
# my_config.yaml
prefer: reference
```

Expected dispatch log: all ops → `reference(reference)`.

### Environment Variable Overrides

Environment variables take precedence over YAML config — useful for one-off overrides without editing the file:

| Variable | Overrides YAML | Description |
|----------|---------------|-------------|
| `SGLANG_OOT_PREFER` | `prefer` | Global backend preference |
| `SGLANG_OOT_OP_PREFER` | `op_backends` | Per-op override, e.g. `RMSNorm:vendor` |
| `SGLANG_OOT_BLACKLIST` | `oot_blacklist` | Skip listed ops from OOT dispatch |
| `USE_FLAGGEMS` | — | Master switch for ATen replacement (`0` to disable) |
| `SGLANG_FLAGGEMS_EXCLUDE` | `flagos_blacklist` | ATen ops to exclude |

```bash
# Override YAML: force all ops to reference backend
SGLANG_FL_CONFIG=./my_config.yaml SGLANG_OOT_PREFER=reference \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

**Priority chain:**
```
Environment variables > YAML config file (SGLANG_FL_CONFIG) > Code defaults
```

### Distributed Communication

| Variable | Default | Description |
|----------|---------|-------------|
| `SGLANG_FL_DIST_BACKEND` | `nccl` | Backend: `nccl` / `hccl` / `flagcx` |
| `FLAGCX_PATH` | — | FlagCX installation path |

```bash
# Use FlagCX for multi-card communication
SGLANG_FL_DIST_BACKEND=flagcx FLAGCX_PATH=/path/to/FlagCX \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-14B-Instruct \
    --tp 8 --port 30000 --disable-piecewise-cuda-graph
```

## Debugging & Diagnostics

### Dispatch Log

See which backend each fused op resolved to (written at server startup):

```bash
rm -f /tmp/dispatch.log
SGLANG_OOT_DISPATCH_LOG=/tmp/dispatch.log \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

sort -u /tmp/dispatch.log
# [OOT-DISPATCH] SiluAndMul → flagos(flagos)
# [OOT-DISPATCH] RMSNorm → flagos(flagos)
# [OOT-DISPATCH] RotaryEmbedding → flagos(flagos)
```

### ATen Replacement Log

Record which PyTorch ATen ops were replaced by FlagGems:

```bash
rm -f /tmp/gems_aten.txt
SGLANG_FLAGGEMS_RECORD=1 SGLANG_FLAGGEMS_LOG_PATH=/tmp/gems_aten.txt \
  python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# After first inference request:
sort -u /tmp/gems_aten.txt
```

> The log uses `_AtenOnlyFilter` to record only `flag_gems.ops.*` namespace calls, excluding internal FlagGems calls triggered by Layer 2 implementations.

### Precision Bisection

When numerical differences appear, isolate the responsible layer:

```bash
# Step 1: Disable everything — confirm vanilla SGLang works
SGLANG_PLUGINS="__none__" python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Step 2: Enable only Layer 2 (fused ops), disable ATen replacement
SGLANG_FLAGGEMS_MODE=off python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Step 3: Per-op isolation — only SiluAndMul uses flagos, RMSNorm uses reference
SGLANG_FLAGGEMS_MODE=off \
SGLANG_OOT_OP_PREFER="SiluAndMul:flagos,RMSNorm:reference" \
    python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph

# Step 4: Gradually enable ATen ops
SGLANG_FLAGGEMS_MODE=only SGLANG_FLAGGEMS_INCLUDE=rms_norm,silu \
    python -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --port 30000 --disable-piecewise-cuda-graph
```

> If output diverges at Step N but not Step N-1, the responsible layer/op is isolated.

### Common Issues

| Symptom | Cause & Fix |
|---------|-------------|
| `dispatch.log` is empty | Plugin not loaded — check `pip show sglang_plugin_FL` |
| `gems_aten.txt` is empty | `USE_FLAGGEMS=0` or `SGLANG_FLAGGEMS_MODE=off` is set |
| `forward_cuda` error on non-NVIDIA | An op lacks OOT registration — register it or add to whitelist |
| `ImportError: sgl_kernel` | Normal on non-CUDA — the OOT dispatch bypasses `forward_cuda` |
| tp>1 hangs at startup | Check GPU count, NCCL env vars, model TP compatibility |
| OOM at engine startup | Reduce `--mem-fraction-static` (default 0.5) |

## Vendor Integration

Chip vendors integrate by adding a single directory under `ops/vendor/`:

```bash
cp -r sglang_plugin_FL/ops/vendor/template/ sglang_plugin_FL/ops/vendor/my_chip/
```

Implement `backend.py`:

```python
from sglang_plugin_FL.ops.vendor.base import VendorBackend

class MyChipBackend(VendorBackend):
    @property
    def name(self) -> str:
        return "my_chip"

    @property
    def vendor(self) -> str:
        return "my_vendor"

    def is_available(self) -> bool:
        try:
            import my_chip_sdk
            return my_chip_sdk.device_count() > 0
        except ImportError:
            return False

    def silu_and_mul(self, obj, x):
        import my_chip_sdk
        return my_chip_sdk.silu_and_mul(x)

    def rms_norm(self, obj, x, residual=None, post_residual_addition=None):
        import my_chip_sdk
        return my_chip_sdk.rms_norm(x, obj.weight.data, obj.variance_epsilon)
```

No other files need modification. The auto-discovery mechanism scans `ops/vendor/` at load time, imports each `backend.py`, checks `is_available()`, and registers all operator methods.

### Existing Backends

| Vendor | Directory | Hardware Detection |
|--------|-----------|-------------------|
| NVIDIA CUDA | `vendor/cuda/` | `torch.cuda.is_available()` |
| Huawei Ascend | `vendor/ascend/` | `import torch_npu` |
| Mock NPU | `vendor/mock_npu/` | `SGLANG_MOCK_NPU_AVAILABLE=1` (testing only) |

## Project Structure

```
sglang_plugin_FL/
├── pyproject.toml                    # Package config + entry_points registration
└── sglang_plugin_FL/
    ├── __init__.py                   # Plugin entry: FlagGems + OOT dispatch + communicator hooks
    ├── platform.py                   # PlatformFL (device identity, memory, graph capture)
    ├── communicator.py               # CommunicatorFL (FlagCX / torch.distributed wrapper)
    ├── flagcx_communicator.py        # FlagCX-specific communicator
    ├── config/
    │   ├── __init__.py               # YAML config loader with platform auto-detection
    │   ├── sample.yaml               # Full example config with all options documented
    │   ├── cuda.yaml                 # CUDA platform defaults
    │   └── ascend.yaml               # Ascend platform defaults (with blacklists)
    └── ops/
        ├── __init__.py               # Op implementation exports
        ├── silu_and_mul.py           # SiluAndMul: flagos + reference implementations
        ├── rms_norm.py               # RMSNorm: flagos + reference implementations
        ├── rotary_embedding.py       # RotaryEmbedding: flagos + reference implementations
        └── vendor/                   # Vendor backend auto-discovery
            ├── __init__.py           # discover_and_register_vendors()
            ├── base.py              # VendorBackend ABC
            ├── cuda/backend.py      # NVIDIA CUDA backend
            ├── ascend/backend.py    # Huawei Ascend NPU backend
            ├── mock_npu/backend.py  # Mock backend for testing
            └── template/backend.py  # Template for new vendors
```

## How It Works

### Plugin Loading

The plugin registers two entry_points in `pyproject.toml`:

```toml
[project.entry-points."sglang.srt.plugins"]
sglang_plugin_FL = "sglang_plugin_FL:load_plugin"

[project.entry-points."sglang.srt.platforms"]
sglang_plugin_FL = "sglang_plugin_FL:activate_platform"
```

SGLang discovers and loads the plugin automatically at startup via setuptools entry_points.

### Dispatch Hook

The core mechanism is an AROUND hook on `MultiPlatformOp.dispatch_forward()`:

```
dispatch_forward() called for an op
  → AROUND hook intercepts
    → Check WHITELIST/BLACKLIST
    → Read SGLANG_OOT_PREFER / SGLANG_OOT_OP_PREFER
    → Try candidates in priority order
    → Check is_available() for each candidate
    → Write dispatch log
    → Return selected implementation
  → If no OOT match: fall through to original SGLang dispatch (CUDA/HIP)
```

### ATen Replacement

```
Plugin loads → flag_gems.enable(record=True)
  → PyTorch dispatch table registers Triton kernels for ATen ops
  → On first inference call, each replaced op is logged
  → _AtenOnlyFilter ensures only flag_gems.ops.* calls are recorded
    (excludes internal FlagGems calls from Layer 2 flagos implementations)
```

## Known Issues

- **Piecewise CUDA Graph not supported**: FlagGems Triton kernels contain `logging.Logger` calls that are incompatible with `torch.compile` (used by SGLang's piecewise CUDA graph). Use `--disable-piecewise-cuda-graph` when launching the server. Regular CUDA graph capture works normally.

## License

Apache-2.0
