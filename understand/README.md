# Batch Invariant 功能总结 (Summary)

本文档是对 vLLM Batch Invariant 功能的快速参考总结。详细分析请查看 `batch_invariant_analysis.md`。

## 什么是 Batch Invariant？

Batch Invariant（批次不变性）确保模型输出具有确定性，相同的输入在不同批次大小下产生完全相同的输出。

## 核心依赖

### 硬件
- **NVIDIA GPU**: H100/H200 (计算能力 9.0) 或 B100/B200 (计算能力 10.0)
- **不支持**: 较旧的 GPU 如 V100

### 软件库
| 库名 | 用途 |
|-----|------|
| **PyTorch** >= 2.9.1 | 深度学习框架，提供算子替换机制 |
| **Triton** | GPU 编程框架，用于编写自定义确定性内核 |
| **CUDA Toolkit** | cuBLAS/cuBLASLt (矩阵运算), NCCL (分布式通信) |
| **FlashInfer/Flash Attention** | 注意力机制后端 |

## 关键实现文件

```
vllm/
├── model_executor/layers/batch_invariant.py  # Python 核心实现
├── csrc/
│   ├── core/batch_invariant.hpp              # C++ 检测函数
│   └── layernorm_kernels.cu                  # CUDA RMS Norm 内核
├── docs/features/batch_invariance.md         # 用户文档
└── tests/v1/determinism/
    └── test_batch_invariance.py              # 测试套件
```

## 核心 Triton 内核

1. **matmul_kernel_persistent**: 持久化矩阵乘法 (确定性 tile 遍历)
2. **bmm_kernel**: 批次矩阵乘法 (3D 张量)
3. **_log_softmax_kernel**: Log Softmax (数值稳定)
4. **mean_kernel**: 均值计算 (确定性归约)
5. **_rms_norm_kernel**: RMS 归一化 (float32 精度)

## 如何启用

### 环境变量
```bash
export VLLM_BATCH_INVARIANT=1
```

### Python 代码
```python
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"

from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(prompts, SamplingParams(seed=42))
```

### 服务器模式
```bash
VLLM_BATCH_INVARIANT=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

## 关键环境配置

```python
# cuBLAS 确定性配置
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# NCCL 确定性配置 (分布式)
os.environ["NCCL_LAUNCH_MODE"] = "GROUP"
os.environ["NCCL_ALGO"] = "allreduce:tree"
os.environ["NCCL_MIN_NCHANNELS"] = "1"
os.environ["NCCL_MAX_NCHANNELS"] = "1"

# 禁用非确定性优化
torch.backends.cuda.matmul.fp32_precision = "ieee"  # 禁用 TF32
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# 禁用自定义 all-reduce
os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
```

## 支持的注意力后端

✅ **支持**:
- `FLASH_ATTN` (最佳)
- `FLASHINFER`
- `FLASH_ATTN_MLA`
- `TRITON_MLA`

❌ **不支持**:
- `FLASHMLA`
- `FLEX_ATTENTION`
- `FLASHINFER_MLA`

## 已验证的模型

- **DeepSeek**: V3, V3-0324, R1, V3.1
- **Qwen3 (Dense)**: 1.7B, 8B
- **Qwen3 (MoE)**: 30B-A3B, Next-80B-A3B
- **Llama 3**: 3.1-8B, 3.2-1B

## 性能影响

- **吞吐量**: 下降 5-15%
- **延迟**: 增加 5-15%
- **内存**: 需要更大的工作区 (2GB)

## 算子替换机制

使用 `torch.library.Library` 替换标准 PyTorch 算子：

```python
_batch_invariant_LIB = torch.library.Library("aten", "IMPL")

# 替换的算子
_batch_invariant_LIB.impl("aten::mm", mm_batch_invariant, "CUDA")
_batch_invariant_LIB.impl("aten::addmm", addmm_batch_invariant, "CUDA")
_batch_invariant_LIB.impl("aten::matmul", matmul_batch_invariant, "CUDA")
_batch_invariant_LIB.impl("aten::linear", linear_batch_invariant, "CUDA")
_batch_invariant_LIB.impl("aten::softmax", softmax_batch_invariant, "CUDA")
_batch_invariant_LIB.impl("aten::_log_softmax", _log_softmax_batch_invariant, "CUDA")
_batch_invariant_LIB.impl("aten::mean.dim", mean_batch_invariant, "CUDA")
_batch_invariant_LIB.impl("aten::bmm", bmm_batch_invariant, "CUDA")
```

## 测试方法

### 1. Needle 测试
验证相同提示在不同批次大小下产生相同输出

### 2. Logprobs 位级不变性测试
比较 BS=1 和 BS=N 的 logprobs，要求位级相同

### 3. 反向测试
禁用批次不变性应该产生不同输出

## 使用场景

**✅ 推荐**:
- 研究实验需要可重复性
- 强化学习训练
- 模型和框架调试
- 需要一致性保证的生产系统

**❌ 不推荐**:
- 对性能极度敏感
- 使用不支持的硬件
- 不需要确定性

## 调试检查清单

```python
# 1. 检查是否启用
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
assert vllm_is_batch_invariant(), "Batch invariance not enabled!"

# 2. 检查硬件支持
from vllm.platforms import current_platform
assert current_platform.has_device_capability(90), "GPU not supported!"

# 3. 验证确定性
outputs1 = llm.generate(prompts, sampling_params)
outputs2 = llm.generate(prompts, sampling_params)
assert outputs1[0].outputs[0].text == outputs2[0].outputs[0].text

# 4. 测量性能开销
import time
start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start
print(f"Time: {elapsed:.2f}s")
```

## 限制

1. **硬件**: 仅 H100/H200, B100/B200
2. **性能**: 5-15% 开销
3. **内存**: 需要更多 GPU 内存
4. **后端**: 某些注意力后端不支持
5. **优化**: TF32 和降精度归约被禁用

## 架构依赖图

```
批次不变性
    ├── GPU: H100/H200/B100/B200 (Compute >= 9.0)
    ├── PyTorch >= 2.9.1
    │   └── torch.library (算子替换)
    ├── Triton
    │   ├── matmul_kernel_persistent
    │   ├── bmm_kernel
    │   ├── _log_softmax_kernel
    │   ├── mean_kernel
    │   └── _rms_norm_kernel
    ├── CUDA Toolkit
    │   ├── cuBLAS (矩阵运算)
    │   ├── cuBLASLt (优化)
    │   └── NCCL (分布式)
    └── FlashInfer/Flash Attention
```

## 参考资源

- **详细分析**: `understand/batch_invariant_analysis.md`
- **用户文档**: `docs/features/batch_invariance.md`
- **实现代码**: `vllm/model_executor/layers/batch_invariant.py`
- **测试代码**: `tests/v1/determinism/test_batch_invariance.py`
- **基准测试**: `benchmarks/benchmark_batch_invariance.py`
- **GitHub Issue**: https://github.com/vllm-project/vllm/issues/27433

## 快速命令参考

```bash
# 检查 GPU 支持
nvidia-smi --query-gpu=compute_cap --format=csv

# 启用批次不变性
export VLLM_BATCH_INVARIANT=1

# 运行测试
pytest tests/v1/determinism/test_batch_invariance.py -v

# 运行基准测试
python benchmarks/benchmark_batch_invariance.py

# 调试模式
VLLM_BATCH_INVARIANT=1 VLLM_LOGGING_LEVEL=DEBUG vllm serve model
```
