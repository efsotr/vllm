# vLLM 优化参数理解文档 / vLLM Optimization Parameters Documentation

本目录包含对 vLLM 优化参数的详细分析和说明文档。

This directory contains detailed analysis and documentation of vLLM optimization parameters.

## 文档列表 / Document List

### 1. 优化参数分析（中文）/ Optimization Parameters Analysis (Chinese)
**文件**: `optimization_parameters_false.md`

详细说明了 vLLM 中所有默认值为 `False` 的优化参数，包括：
- 54 个优化参数的完整列表
- 每个参数的功能说明
- 默认禁用的原因
- 启用建议和使用场景
- 常见优化组合示例

Detailed explanation of all optimization parameters in vLLM with `False` default values, including:
- Complete list of 54 optimization parameters
- Functional description of each parameter
- Reasons for being disabled by default
- Enabling recommendations and use cases
- Examples of common optimization combinations

### 2. Optimization Parameters Analysis (English)
**文件**: `optimization_parameters_false_en.md`

English version of the optimization parameters documentation, covering:
- Complete parameter reference by category
- Usage recommendations and best practices
- Performance vs. stability trade-offs
- Platform-specific considerations
- Quick reference tables

## 参数分类概览 / Parameter Categories Overview

### 1. 编译配置 / Compilation Configuration (11 参数)
- 内核融合优化 / Kernel fusion optimizations
- 图优化 / Graph optimizations
- CUDA Graph 配置 / CUDA Graph configuration

### 2. 缓存配置 / Cache Configuration (3 参数)
- KV 缓存管理 / KV cache management
- 前缀缓存 / Prefix caching
- 内存优化 / Memory optimization

### 3. 并行配置 / Parallel Configuration (10 参数)
- 数据并行 / Data parallelism
- 专家并行 / Expert parallelism
- 负载均衡 / Load balancing

### 4. 模型配置 / Model Configuration (7 参数)
- 注意力机制 / Attention mechanisms
- 分词器行为 / Tokenizer behavior
- 安全设置 / Security settings

### 5. 调度器配置 / Scheduler Configuration (3 参数)
- 异步调度 / Async scheduling
- 多模态处理 / Multimodal processing

### 6. 注意力配置 / Attention Configuration (5 参数)
- 注意力后端 / Attention backends
- 预填充优化 / Prefill optimizations

### 7. 可观测性配置 / Observability Configuration (4 参数)
- 指标收集 / Metrics collection
- 性能分析 / Performance profiling
- 追踪功能 / Tracing capabilities

### 8. 多模态配置 / Multimodal Configuration (3 参数)
- 图像/视频处理 / Image/video processing
- 嵌入处理 / Embedding handling

### 9. 引擎参数 / Engine Arguments (4 参数)
- 日志控制 / Logging control
- LoRA 支持 / LoRA support

### 10. 其他配置 / Other Configuration (8 参数)
- 结构化输出 / Structured outputs
- 推测解码 / Speculative decoding
- 分析器配置 / Profiler configuration

## 关键发现 / Key Findings

### 为什么这些参数默认为 False？ / Why Are These Parameters False by Default?

1. **兼容性 / Compatibility**: 仅在特定硬件或模型架构上有效 / Only work on specific hardware or model architectures

2. **稳定性 / Stability**: 可能引入数值不稳定或兼容性问题 / May introduce numerical instability or compatibility issues

3. **安全性 / Security**: 某些功能有安全风险 / Some features pose security risks (e.g., `trust_remote_code`)

4. **性能权衡 / Performance Trade-offs**: 增加开销，仅在特定工作负载下有益 / Add overhead that only benefits specific workloads

5. **开发状态 / Development Status**: 功能还在开发或测试阶段 / Features still under development or testing

6. **调试工具 / Debugging Tools**: 影响性能的监控和分析功能 / Monitoring and profiling features that impact performance

## 使用建议 / Usage Recommendations

### 启用参数的最佳实践 / Best Practices for Enabling Parameters

✅ **DO 做**:
- 逐个启用参数并测试 / Enable parameters one at a time and test
- 在代表性工作负载上进行基准测试 / Benchmark on representative workloads
- 监控性能和输出质量指标 / Monitor performance and output quality metrics
- 保留可工作的配置备份 / Keep backups of working configurations
- 阅读参数的完整文档 / Read complete documentation for parameters

❌ **DON'T 不要**:
- 同时启用多个未测试的参数 / Enable multiple untested parameters simultaneously
- 在生产环境直接启用实验性功能 / Enable experimental features directly in production
- 忽视硬件和平台限制 / Ignore hardware and platform limitations
- 跳过基准测试和验证 / Skip benchmarking and validation

### 常见优化场景 / Common Optimization Scenarios

#### 🚀 生产部署 / Production Deployment
```python
enable_prefix_caching=True
enable_chunked_prefill=True
# For MoE models:
enable_expert_parallel=True
```

#### 🔬 FP8 量化 / FP8 Quantization
```python
fuse_norm_quant=True
fuse_act_quant=True
fuse_attn_quant=True
eliminate_noops=True
calculate_kv_scales=True
```

#### 🐛 开发调试 / Development & Debugging
```python
enforce_eager=True
kv_cache_metrics=True
cudagraph_metrics=True
enable_mfu_metrics=True
```

#### 🖼️ 多模态应用 / Multimodal Applications
```python
skip_mm_profiling=True  # Faster startup
compile_mm_encoder=True  # If supported
```

## 性能影响矩阵 / Performance Impact Matrix

| 优化类型 / Optimization | 性能提升 / Perf Gain | 稳定性风险 / Risk | 推荐程度 / Recommendation |
|-------------------------|---------------------|------------------|-------------------------|
| 内核融合 (FP8) / Kernel Fusion | 高 / High | 中 / Medium | 在目标硬件充分测试 / Test thoroughly |
| 异步调度 / Async Scheduling | 中 / Medium | 低-中 / Low-Med | 适合在线服务 / Good for serving |
| 专家并行 / Expert Parallel | 高 (MoE) | 低 / Low | MoE 模型安全 / Safe for MoE |
| CUDA Graph | 高 / High | 低 / Low | 通常安全（默认启用）/ Usually safe |
| 自定义 All-Reduce / Custom All-Reduce | 中 / Medium | 低 / Low | 很少需要禁用 / Rarely disable |
| 分析指标 / Profiling Metrics | N/A (开销) | 极低 / Very Low | 监控时启用 / Enable for monitoring |

## 平台特定说明 / Platform-Specific Notes

### NVIDIA CUDA
- ✅ 支持大多数优化 / Most optimizations supported
- ✅ H100/Ada 支持 FP8 / FP8 on H100/Ada
- ✅ Ampere+ 支持 FlashAttention / FlashAttention on Ampere+

### AMD ROCm
- ⚠️ 部分 CUDA 优化已移植 / Some CUDA optimizations ported
- ✅ MI300 系列支持 FP8 / FP8 on MI300 series
- 📚 查看平台特定文档 / Check platform-specific docs

### Google TPU
- ⚠️ 不同的优化集 / Different optimization set
- 🔧 分布式需要 Ray / Ray required for distributed
- 📚 参考 TPU 指南 / Refer to TPU guides

### Intel CPU
- ⚠️ 有限的优化支持 / Limited optimization support
- ❌ 某些架构禁用功能 / Some features disabled on ARM/POWER/RISC-V
- 💡 注重并行而非内核优化 / Focus on parallelism over kernels

## 快速查找 / Quick Reference

### 按功能查找参数 / Find Parameters by Function

- **提高吞吐量 / Improve Throughput**: `enable_chunked_prefill`, `async_scheduling`, `enable_dbo`
- **降低延迟 / Reduce Latency**: `enable_prefix_caching`, CUDA graph optimizations
- **节省内存 / Save Memory**: `cpu_offload_gb`, `enable_prefix_caching`
- **FP8 优化 / FP8 Optimization**: `fuse_norm_quant`, `fuse_act_quant`, `calculate_kv_scales`
- **MoE 模型 / MoE Models**: `enable_expert_parallel`, `enable_eplb`
- **调试分析 / Debugging**: `enforce_eager`, profiling/metrics parameters
- **安全考虑 / Security**: `trust_remote_code`, `enable_prompt_embeds`, `enable_mm_embeds`

### 按影响范围查找 / Find Parameters by Impact

- **编译时 / Compile-time**: Compilation config parameters
- **运行时 / Runtime**: Scheduling, parallel, attention parameters
- **初始化时 / Initialization**: Model config, profiling skip parameters
- **请求级 / Per-request**: LoRA, structured outputs parameters

## 相关资源 / Related Resources

- 📖 [vLLM 官方文档 / Official Docs](https://docs.vllm.ai/)
- 💻 [GitHub 仓库 / Repository](https://github.com/vllm-project/vllm)
- 🔧 [配置源码 / Config Source](../../vllm/config/)
- 💬 [社区讨论 / Discussions](https://github.com/vllm-project/vllm/discussions)

## 贡献指南 / Contributing

如果发现错误或有改进建议 / If you find errors or have suggestions:

1. 检查最新源码以验证准确性 / Check latest source code for accuracy
2. 在 GitHub 上开启 Issue 或 PR / Open issue or PR on GitHub
3. 包含具体的文件位置和行号 / Include specific file locations and line numbers
4. 提供测试结果和基准数据 / Provide test results and benchmark data

---

## 文档维护 / Document Maintenance

- **创建日期 / Created**: 2025-12-23
- **最后更新 / Last Updated**: 2025-12-23
- **版本 / Version**: 1.0
- **基于代码库 / Based on**: vLLM main branch

### 变更日志 / Changelog

#### v1.0 (2025-12-23)
- 初始版本，包含 54 个优化参数的完整分析
- 中英文双语文档
- 按配置类别组织
- 包含使用建议和最佳实践

#### Initial version with complete analysis of 54 optimization parameters
- Bilingual Chinese-English documentation
- Organized by configuration category
- Includes usage recommendations and best practices

---

**注意 / Note**: 本文档基于 vLLM 代码库快照。由于代码持续演进，某些参数可能会变化。请始终参考最新的源代码和官方文档。

**Note**: This documentation is based on a snapshot of the vLLM codebase. As code evolves, some parameters may change. Always refer to the latest source code and official documentation.
