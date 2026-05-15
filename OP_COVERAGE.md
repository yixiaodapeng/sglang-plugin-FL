# 算子覆盖率报告 (Operator Coverage Report)

> 生成日期: 2026-05-15
> 插件版本: sglang_plugin_FL v0.1.0
> 测试模型: Qwen3.6-27B, Qwen3.6-35B-A3B
> SGLang 版本: v0.5.11

---

## 一、Qwen3.6-27B (Hybrid Attention + FLA + MoE)

模型文件: `sglang/srt/models/qwen3_next.py`

### 1.1 Layer 2 (Fused Ops)

| 算子 | SGLang源码位置 | FlagGems状态 | 当前实现 | 当前fallback机制 |
|------|--------------|-------------|---------|----------------|
| `silu_and_mul` | `srt/layers/activation.py:83` | ✅ 已支持 | `default.flagos` | - |
| `gemma_rms_norm` | `srt/layers/layernorm.py:558` | ❌ **待实现** | `vendor.cuda` | `sgl_kernel.gemma_rmsnorm` (CUDA) |
| `rotary_embedding` | `srt/layers/rotary_embedding/base.py:50` | ✅ 已支持 | `default.flagos` | - |
| `topk` | `srt/layers/moe/topk.py:248` | ⚠️ **接口不匹配** | `vendor.cuda` (fallback) | FlagGems实现报错: `'TopKConfig' object has no attribute 'topk'`，fallback到SGLang native TopK |
| `fused_moe` | `srt/layers/quantization/unquant.py:162` | ❌ **待实现** | `vendor.cuda` | `SGLang MoeRunner` + `TritonMoeQuantInfo` |
| `chunk_gated_delta_rule` | `srt/layers/attention/fla/chunk.py:27` | ❌ **待实现** | `vendor.cuda` | `ChunkGatedDeltaRuleFunction.apply()` → SGLang triton chunk kernel |
| `fused_recurrent_gated_delta_rule` | `srt/layers/attention/fla/fused_recurrent.py:124` | ✅ 已支持 | `default.flagos` | - |
| `fused_recurrent_gated_delta_rule_packed_decode` | `srt/layers/attention/fla/fused_recurrent.py:268` | ❌ **待实现** | `vendor.cuda` | SGLang native triton packed decode kernel |

**小结**: 8个Layer 2算子，FlagGems已支持 **3个**，待实现 **4个**，接口不匹配 **1个**。

### 1.2 Layer 1 (ATen Ops)

FlagGems通过 `flag_gems.enable()` 注册了 **324个ATen算子**。未覆盖的ATen算子自动fallback到PyTorch原生CUDA实现（cuBLAS/cuDNN）。

**已知问题** (需要修复):

| 算子 | 问题描述 | 触发场景 | 当前workaround |
|------|---------|---------|---------------|
| `aten::count_nonzero` | `sparse_coo permute dimension mismatch` | MoE expert location计算 | 环境变量黑名单: `SGLANG_FL_FLAGOS_BLACKLIST=count_nonzero` |
| `aten::index_put_` | `broadcast shape mismatch on 4D tensors` | Mamba state pool allocation | 环境变量黑名单: `SGLANG_FL_FLAGOS_BLACKLIST=index_put_` |

**运行命令**:
```bash
SGLANG_FL_FLAGOS_BLACKLIST=count_nonzero,index_put_ \
python3 examples/qwen3_6_27b_offline_inference.py
```

---

## 二、Qwen3.6-35B-A3B (MoE, 256 experts)

模型文件: `sglang/srt/models/qwen3_moe.py`

### 2.1 Layer 2 (Fused Ops)

| 算子 | SGLang源码位置 | FlagGems状态 | 当前实现 | 当前fallback机制 |
|------|--------------|-------------|---------|----------------|
| `silu_and_mul` | `srt/layers/activation.py:83` | ✅ 已支持 | `default.flagos` | - |
| `rms_norm` | `srt/layers/layernorm.py:175` | ✅ 已支持 | `default.flagos` | - |
| `mrotary_embedding` | `srt/layers/rotary_embedding/mrope.py:42` | ❌ **待实现** | `vendor.cuda` | `triton_mrope_fused` (Triton) |
| `topk` | `srt/layers/moe/topk.py:248` | ⚠️ **接口不匹配** | `vendor.cuda` (fallback) | FlagGems实现报错: `'TopKConfig' object has no attribute 'topk'`，fallback到SGLang native TopK |
| `fused_moe` | `srt/layers/quantization/unquant.py:162` | ❌ **待实现** | `vendor.cuda` | `SGLang MoeRunner` + `TritonMoeQuantInfo` |

**小结**: 5个Layer 2算子，FlagGems已支持 **2个**，待实现 **2个**，接口不匹配 **1个**。

### 2.2 Layer 1 (ATen Ops)

与27B模型相同，FlagGems覆盖324个ATen算子，其余自动fallback到PyTorch CUDA。

**已知问题** (与27B相同):

| 算子 | 问题描述 | 触发场景 | 当前workaround |
|------|---------|---------|---------------|
| `aten::count_nonzero` | `sparse_coo permute dimension mismatch` | MoE expert location计算 | 环境变量黑名单: `SGLANG_FL_FLAGOS_BLACKLIST=count_nonzero` |
| `aten::index_put_` | `broadcast shape mismatch on 4D tensors` | Mamba state pool allocation | 环境变量黑名单: `SGLANG_FL_FLAGOS_BLACKLIST=index_put_` |

**运行命令**:
```bash
SGLANG_FL_FLAGOS_BLACKLIST=count_nonzero,index_put_ \
python3 examples/qwen3_6_35b_a3b_offline_inference.py
```
