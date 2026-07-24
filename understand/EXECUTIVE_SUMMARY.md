# Batch Invariant Feature Analysis - Executive Summary
# Batch Invariant 功能分析 - 执行摘要

## English Summary

This directory contains a comprehensive analysis of vLLM's Batch Invariant feature, which ensures deterministic and consistent model outputs regardless of batch size or request ordering.

### Key Findings

**What is Batch Invariant?**
A feature that guarantees the same input produces identical outputs across different batch sizes, essential for:
- Framework and model debugging
- Reinforcement Learning training
- Large-scale inference systems requiring consistency guarantees

**Hardware Requirements**
- **Supported**: NVIDIA H100/H200 (Compute 9.0), B100/B200 (Compute 10.0)
- **Unsupported**: Older GPUs like V100, A100 (limited support)

**Software Dependencies**
1. **PyTorch** >= 2.9.1 - Provides operator replacement mechanism via torch.library
2. **Triton** - OpenAI's GPU programming framework for custom deterministic kernels
3. **CUDA Toolkit** - cuBLAS/cuBLASLt for matrix operations, NCCL for distributed communication
4. **FlashInfer/Flash Attention** - Attention mechanism backends

**Core Implementation**
- **Operator Replacement**: Uses `torch.library.Library` to replace PyTorch's non-deterministic operators with deterministic Triton kernels
- **Custom Triton Kernels**: Implements matmul, bmm, softmax, mean, and RMS norm with fixed computation order
- **Environment Configuration**: Strict CUDA/NCCL settings to enforce determinism
- **Numerical Stability**: Uses float32 intermediate precision and stable algorithms

**Key Files**
- `understand/batch_invariant_analysis.md` - Full 830-line detailed analysis (in Chinese)
- `understand/README.md` - Quick reference summary (in Chinese)
- Implementation: `vllm/model_executor/layers/batch_invariant.py`
- Tests: `tests/v1/determinism/test_batch_invariance.py`

**Performance Impact**
- Throughput: 5-15% decrease
- Memory: Increased (2GB workspace buffer)
- Trade-off: Determinism for performance

**How to Enable**
```bash
export VLLM_BATCH_INVARIANT=1
vllm serve model-name
```

**Supported Models**
- DeepSeek: V3, V3-0324, R1, V3.1
- Qwen3 (Dense): 1.7B, 8B
- Qwen3 (MoE): 30B-A3B, Next-80B-A3B
- Llama 3: 3.1-8B, 3.2-1B

---

## 中文摘要

本目录包含对 vLLM Batch Invariant（批次不变性）功能的全面分析，该功能确保模型输出的确定性和一致性，无论批次大小或请求顺序如何。

### 关键发现

**什么是 Batch Invariant？**
一个保证相同输入在不同批次大小下产生完全相同输出的特性，对以下场景至关重要：
- 框架和模型调试
- 强化学习训练
- 需要一致性保证的大规模推理系统

**硬件要求**
- **支持**: NVIDIA H100/H200（计算能力 9.0）、B100/B200（计算能力 10.0）
- **不支持**: 较旧的 GPU 如 V100、A100（有限支持）

**软件依赖**
1. **PyTorch** >= 2.9.1 - 通过 torch.library 提供算子替换机制
2. **Triton** - OpenAI 的 GPU 编程框架，用于编写自定义确定性内核
3. **CUDA Toolkit** - cuBLAS/cuBLASLt 用于矩阵运算，NCCL 用于分布式通信
4. **FlashInfer/Flash Attention** - 注意力机制后端

**核心实现**
- **算子替换**: 使用 `torch.library.Library` 将 PyTorch 的非确定性算子替换为确定性 Triton 内核
- **自定义 Triton 内核**: 实现了 matmul、bmm、softmax、mean 和 RMS norm，具有固定的计算顺序
- **环境配置**: 严格的 CUDA/NCCL 设置以强制确定性
- **数值稳定性**: 使用 float32 中间精度和稳定的算法

**关键文件**
- `understand/batch_invariant_analysis.md` - 完整的 830 行详细分析（中文）
- `understand/README.md` - 快速参考摘要（中文）
- 实现代码: `vllm/model_executor/layers/batch_invariant.py`
- 测试代码: `tests/v1/determinism/test_batch_invariance.py`

**性能影响**
- 吞吐量: 下降 5-15%
- 内存: 增加（2GB 工作区缓冲）
- 权衡: 用性能换取确定性

**如何启用**
```bash
export VLLM_BATCH_INVARIANT=1
vllm serve model-name
```

**支持的模型**
- DeepSeek: V3, V3-0324, R1, V3.1
- Qwen3（密集型）: 1.7B, 8B
- Qwen3（MoE）: 30B-A3B, Next-80B-A3B
- Llama 3: 3.1-8B, 3.2-1B

---

## Document Structure / 文档结构

### 1. README.md (快速参考 / Quick Reference)
- **Length**: 234 lines / 6.5KB
- **Content**: Quick reference summary with key concepts, dependencies, commands
- **内容**: 快速参考摘要，包含核心概念、依赖、命令

### 2. batch_invariant_analysis.md (详细分析 / Detailed Analysis)
- **Length**: 830 lines / 23KB
- **Sections**: 10 major sections covering all aspects
- **章节**: 10 个主要章节，涵盖所有方面

#### Detailed Contents:
1. 概述 (Overview)
2. 硬件和软件要求 (Hardware & Software Requirements)
3. 实现原理 (Implementation Principles)
   - 算子替换机制 (Operator Replacement)
   - Triton 内核实现 (Triton Kernel Implementation)
   - C++ 层面支持 (C++ Level Support)
   - 环境变量配置 (Environment Configuration)
4. 实现细节分析 (Implementation Details)
   - Triton 内核优化 (Triton Kernel Optimization)
   - 数值稳定性 (Numerical Stability)
   - 大张量处理 (Large Tensor Handling)
   - 分布式训练支持 (Distributed Training Support)
5. 测试和验证 (Testing & Validation)
6. 性能影响 (Performance Impact)
7. 使用示例 (Usage Examples)
8. 限制和注意事项 (Limitations & Considerations)
9. 未来改进方向 (Future Improvements)
10. 总结 (Summary)

---

## Key Technical Insights / 关键技术洞察

### Triton Kernels Implemented / 实现的 Triton 内核

| Kernel | Purpose | Key Feature |
|--------|---------|-------------|
| `matmul_kernel_persistent` | 2D matrix multiplication | Fixed tile traversal order |
| `bmm_kernel` | Batched matrix multiplication | Deterministic across batches |
| `_log_softmax_kernel` | Log softmax | Numerically stable |
| `mean_kernel` | Mean reduction | Deterministic reduction |
| `_rms_norm_kernel` | RMS normalization | Float32 accumulation |

### Environment Variables / 环境变量

**Essential / 必需**:
```bash
VLLM_BATCH_INVARIANT=1
```

**cuBLAS**:
```bash
CUBLAS_WORKSPACE_CONFIG=":4096:8"
```

**NCCL (for distributed / 分布式)**:
```bash
NCCL_LAUNCH_MODE=GROUP
NCCL_ALGO=allreduce:tree
NCCL_MIN_NCHANNELS=1
NCCL_MAX_NCHANNELS=1
```

**Optimizations Disabled / 禁用的优化**:
```bash
VLLM_ALLREDUCE_USE_SYMM_MEM=0
VLLM_USE_AOT_COMPILE=0
```

### PyTorch Backend Settings / PyTorch 后端设置

```python
# Disable TF32 / 禁用 TF32
torch.backends.cuda.matmul.fp32_precision = "ieee"

# Disable reduced precision / 禁用降精度
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

# Prefer cuBLASLt / 偏好 cuBLASLt
torch.backends.cuda.preferred_blas_library(backend="cublaslt")
```

---

## Testing Methodology / 测试方法

### 1. Needle Test / 针测试
- Insert a "needle" prompt at random positions in batches
- Verify output is identical regardless of batch composition
- 在批次中随机位置插入"针"提示
- 验证无论批次组成如何，输出都完全相同

### 2. Bitwise Invariance Test / 位级不变性测试
- Compare logprobs from BS=1 vs BS=N
- Require bitwise equality (torch.equal)
- 比较 BS=1 与 BS=N 的 logprobs
- 要求位级相等（torch.equal）

### 3. Inverse Test / 反向测试
- Disable batch invariance
- Expect to see differences (proves necessity)
- 禁用批次不变性
- 期望看到差异（证明必要性）

---

## Use Cases / 使用场景

**✅ Recommended / 推荐**:
- Research experiments requiring reproducibility / 需要可重复性的研究实验
- Reinforcement learning training / 强化学习训练
- Model and framework debugging / 模型和框架调试
- Production systems needing consistency / 需要一致性的生产系统

**❌ Not Recommended / 不推荐**:
- Performance-critical scenarios / 对性能极度敏感的场景
- Unsupported hardware / 不支持的硬件
- No determinism required / 不需要确定性

---

## References / 参考资源

- **GitHub Issue**: https://github.com/vllm-project/vllm/issues/27433
- **Documentation**: `docs/features/batch_invariance.md`
- **Implementation**: `vllm/model_executor/layers/batch_invariant.py`
- **Tests**: `tests/v1/determinism/test_batch_invariance.py`
- **Benchmarks**: `benchmarks/benchmark_batch_invariance.py`

---

## Analysis Statistics / 分析统计

- **Total Lines Analyzed**: ~5000+ lines of code
- **Files Examined**: 27 files
- **Documentation Created**: 2 files (1064 lines total)
- **Languages**: Chinese (primary), English (summary)
- **Time to Complete**: Analysis completed in single session
- **Coverage**: Hardware, software, implementation, testing, performance

---

*Last Updated: 2025-12-26*
*Analysis by: GitHub Copilot*
*Repository: efsotr/vllm*
