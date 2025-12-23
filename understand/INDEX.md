# 快速导航 / Quick Navigation

## 📚 如何使用这些文档 / How to Use This Documentation

根据你的需求选择合适的文档：

Choose the appropriate document based on your needs:

### 🎯 我想要... / I want to...

#### 📖 了解所有参数的详细说明 / Learn detailed explanations of all parameters
- **中文**: [optimization_parameters_false.md](./optimization_parameters_false.md)
- **English**: [optimization_parameters_false_en.md](./optimization_parameters_false_en.md)

#### 📊 查看参数的可视化总结 / See visual summary of parameters
- [VISUAL_SUMMARY.md](./VISUAL_SUMMARY.md) - 图表、决策树、配置模板 / Charts, decision trees, config templates

#### 🗺️ 获取文档概览和导航 / Get documentation overview and navigation
- [README.md](./README.md) - 双语总览 / Bilingual overview

---

## 🔍 按场景查找 / Find by Scenario

### 生产环境优化 / Production Optimization
→ [VISUAL_SUMMARY.md - Configuration Templates](./VISUAL_SUMMARY.md#configuration-templates)
- 高吞吐量配置 / High throughput
- 低延迟配置 / Low latency
- 内存受限配置 / Memory constrained

### 调试问题 / Debugging Issues
→ [optimization_parameters_false.md - 开发调试](./optimization_parameters_false.md#开发调试)
→ [optimization_parameters_false_en.md - Development & Debugging](./optimization_parameters_false_en.md#development--debugging)

### 使用 MoE 模型 / Using MoE Models
→ Search for "expert_parallel" in detailed docs
→ [VISUAL_SUMMARY.md - MoE Specialist](./VISUAL_SUMMARY.md#-moe-specialist)

### FP8 量化优化 / FP8 Quantization
→ [VISUAL_SUMMARY.md - FP8 Optimized](./VISUAL_SUMMARY.md#-fp8-optimized-h100)

### 多模态应用 / Multimodal Applications
→ Search for "multimodal" or "mm_" in detailed docs

---

## 📋 按类别查找参数 / Find Parameters by Category

| 类别 / Category | 参数数量 / Count | 文档位置 / Location |
|----------------|---------------|-------------------|
| 编译配置 / Compilation | 11 | [详细文档 / Details](./optimization_parameters_false.md#编译配置-compilationconfig) |
| 并行配置 / Parallel | 10 | [详细文档 / Details](./optimization_parameters_false.md#并行配置-parallelconfig) |
| 模型配置 / Model | 7 | [详细文档 / Details](./optimization_parameters_false.md#模型配置-modelconfig) |
| 注意力配置 / Attention | 5 | [详细文档 / Details](./optimization_parameters_false.md#注意力配置-attentionconfig) |
| 可观测性 / Observability | 4 | [详细文档 / Details](./optimization_parameters_false.md#可观测性配置-observabilityconfig) |
| 其他 / Others | 17 | See detailed docs |

---

## 🎓 学习路径 / Learning Path

### 初学者 / Beginners
1. 阅读 [README.md](./README.md) 了解总体情况
2. 浏览 [VISUAL_SUMMARY.md](./VISUAL_SUMMARY.md) 查看图表和模板
3. 根据需要查阅详细文档中的特定参数

1. Read [README.md](./README.md) for overview
2. Browse [VISUAL_SUMMARY.md](./VISUAL_SUMMARY.md) for charts and templates
3. Consult detailed docs for specific parameters as needed

### 高级用户 / Advanced Users
1. 使用 [VISUAL_SUMMARY.md - Decision Tree](./VISUAL_SUMMARY.md#decision-tree-which-parameters-to-enable) 决定启用哪些参数
2. 参考详细文档了解每个参数的细节
3. 根据配置模板定制你的设置

1. Use [VISUAL_SUMMARY.md - Decision Tree](./VISUAL_SUMMARY.md#decision-tree-which-parameters-to-enable) to decide which parameters to enable
2. Refer to detailed docs for parameter specifics
3. Customize based on configuration templates

### 系统管理员 / System Administrators
1. 查看 [Hardware Support Matrix](./VISUAL_SUMMARY.md#hardware-support-matrix) 了解平台支持
2. 参考 [Performance Impact Spectrum](./VISUAL_SUMMARY.md#performance-impact-spectrum)
3. 使用配置模板作为起点

---

## 🔗 快速链接 / Quick Links

### 常见问题 / Common Questions

**Q: 哪些参数最有用？ / Which parameters are most useful?**
→ [VISUAL_SUMMARY.md - Most Commonly Enabled](./VISUAL_SUMMARY.md#most-commonly-enabled-parameters)

**Q: 如何提高性能？ / How to improve performance?**
→ [VISUAL_SUMMARY.md - Performance Impact](./VISUAL_SUMMARY.md#performance-impact-spectrum)
→ [Configuration Templates](./VISUAL_SUMMARY.md#configuration-templates)

**Q: 哪些参数有风险？ / Which parameters are risky?**
→ [VISUAL_SUMMARY.md - Risk Level Distribution](./VISUAL_SUMMARY.md#risk-level-distribution)
→ [README.md - Security Considerations](./README.md#按功能查找参数--find-parameters-by-function)

**Q: 我的硬件支持哪些优化？ / What optimizations does my hardware support?**
→ [VISUAL_SUMMARY.md - Hardware Support Matrix](./VISUAL_SUMMARY.md#hardware-support-matrix)
→ [README.md - Platform Notes](./README.md#平台特定说明--platform-specific-notes)

**Q: 如何调试问题？ / How to debug issues?**
→ [Detailed docs - Development & Debugging sections]
→ [VISUAL_SUMMARY.md - Debug Mode](./VISUAL_SUMMARY.md#-debug-mode)

---

## 📝 参数速查表 / Parameter Quick Reference

### 按功能 / By Function

| 功能 / Function | 参数 / Parameters | 文档 / Doc |
|----------------|-------------------|------------|
| 提高吞吐量 / Throughput | `enable_chunked_prefill`, `async_scheduling`, `enable_dbo` | [Details](./optimization_parameters_false.md) |
| 降低延迟 / Latency | `enable_prefix_caching`, CUDA graph opts | [Details](./optimization_parameters_false.md) |
| 节省内存 / Save Memory | `cpu_offload_gb`, `enable_prefix_caching` | [Details](./optimization_parameters_false.md) |
| FP8 优化 / FP8 | `fuse_norm_quant`, `fuse_act_quant`, etc. | [Details](./optimization_parameters_false.md#编译配置-compilationconfig) |
| MoE 模型 / MoE | `enable_expert_parallel`, `enable_eplb` | [Details](./optimization_parameters_false.md#并行配置-parallelconfig) |
| 调试 / Debug | `enforce_eager`, profiling/metrics params | [Details](./optimization_parameters_false.md) |

### 按平台 / By Platform

| 平台 / Platform | 推荐参数 / Recommended | 限制 / Limitations |
|----------------|---------------------|-------------------|
| NVIDIA CUDA | Most parameters | Few limitations |
| AMD ROCm | Most performance opts | Some features experimental |
| Google TPU | Parallelism params | Limited kernel optimizations |
| Intel CPU | Parallelism params | Many optimizations unsupported |

---

## 📊 统计信息 / Statistics

- **总参数数 / Total Parameters**: 54
- **文档文件 / Documentation Files**: 4
- **支持语言 / Languages**: Chinese (中文) + English
- **配置类别 / Config Categories**: 10
- **文档总大小 / Total Size**: ~64KB
- **源代码文件 / Source Files**: 13+ config files
- **代码行分析 / Lines Analyzed**: 2000+ lines

---

## 🛠️ 工具和资源 / Tools & Resources

### 官方资源 / Official Resources
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [vLLM Discussions](https://github.com/vllm-project/vllm/discussions)

### 源代码位置 / Source Code Locations
- Configuration files: `vllm/config/*.py`
- Engine arguments: `vllm/engine/arg_utils.py`

### 相关文档 / Related Documentation
- vLLM Performance Tuning Guide
- vLLM Deployment Guide
- vLLM Model Support Matrix

---

## 💡 使用提示 / Usage Tips

### ✅ 最佳实践 / Best Practices

1. **逐步启用 / Enable Gradually**: 一次启用一个参数并测试 / Enable one parameter at a time and test
2. **监控指标 / Monitor Metrics**: 观察性能和质量变化 / Watch performance and quality changes
3. **保存配置 / Save Configs**: 备份可工作的配置 / Backup working configurations
4. **阅读文档 / Read Docs**: 理解参数影响 / Understand parameter impact
5. **基准测试 / Benchmark**: 在真实工作负载上测试 / Test on real workloads

### ⚠️ 注意事项 / Warnings

- 🔒 **安全参数 / Security**: `trust_remote_code`, `enable_*_embeds` - 仅对可信用户 / Only for trusted users
- ⚡ **实验性功能 / Experimental**: 标记为 WIP 或实验性的参数谨慎使用 / Use WIP/experimental params cautiously
- 🖥️ **平台限制 / Platform**: 检查硬件兼容性 / Check hardware compatibility
- 📈 **性能权衡 / Trade-offs**: 了解每个参数的成本 / Understand cost of each parameter

---

## 📞 获取帮助 / Getting Help

### 发现错误？ / Found an Error?
1. 检查最新源码 / Check latest source code
2. 在 GitHub 开 Issue / Open issue on GitHub
3. 提供具体细节 / Provide specific details

### 需要更多信息？ / Need More Information?
1. 查阅官方文档 / Consult official docs
2. 搜索 GitHub Discussions
3. 查看源代码注释 / Read source code comments

### 想要贡献？ / Want to Contribute?
1. Fork 仓库 / Fork repository
2. 改进文档 / Improve documentation
3. 提交 PR / Submit pull request

---

## 📅 文档更新 / Document Updates

- **创建日期 / Created**: 2025-12-23
- **版本 / Version**: 1.0
- **维护者 / Maintainer**: vLLM Community
- **下次审查 / Next Review**: Check with vLLM releases

---

**提示 / Tip**: 为快速查找，使用浏览器的"查找"功能 (Ctrl+F / Cmd+F) 搜索特定参数名称。

**Tip**: For quick lookup, use your browser's "Find" function (Ctrl+F / Cmd+F) to search for specific parameter names.

---

*这是一个快速导航文件。详细内容请查看其他文档。*

*This is a quick navigation file. See other documents for detailed content.*
