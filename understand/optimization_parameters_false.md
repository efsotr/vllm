# vLLM 优化参数分析 - 默认值为 False 的参数

本文档详细说明了 vLLM 中所有默认值为 `False` 的优化参数。这些参数被禁用的原因通常是为了保持兼容性、稳定性，或者因为它们是高级功能，需要特定条件才能启用。

## 目录

1. [编译配置 (CompilationConfig)](#编译配置-compilationconfig)
2. [缓存配置 (CacheConfig)](#缓存配置-cacheconfig)
3. [并行配置 (ParallelConfig)](#并行配置-parallelconfig)
4. [模型配置 (ModelConfig)](#模型配置-modelconfig)
5. [调度器配置 (SchedulerConfig)](#调度器配置-schedulerconfig)
6. [注意力配置 (AttentionConfig)](#注意力配置-attentionconfig)
7. [可观测性配置 (ObservabilityConfig)](#可观测性配置-observabilityconfig)
8. [多模态配置 (MultiModalConfig)](#多模态配置-multimodalconfig)
9. [引擎参数 (EngineArgs)](#引擎参数-engineargs)
10. [其他配置](#其他配置)

---

## 编译配置 (CompilationConfig)

### 1. `enable_qk_norm_rope_fusion: bool = False`
**位置**: `vllm/config/compilation.py:147`

**功能**: 启用融合的 Q/K RMSNorm + RoPE 传递。

**为何默认禁用**:
- 这是一个优化功能，将 Query/Key 的 RMS 归一化和旋转位置编码（RoPE）融合在一起
- 仅在 CUDA 或 ROCm 平台上有效
- 需要特定的硬件和模型架构支持
- 可能在某些情况下引入数值不稳定性

**启用场景**: 在支持的 CUDA/ROCm 平台上，且模型使用 RoPE 位置编码时，可以启用以提高性能。

---

### 2. `evaluate_guards: bool = False`
**位置**: `vllm/config/compilation.py:268`

**功能**: 检测和报告 Dynamo 是否对动态形状进行了特化。

**为何默认禁用**:
- 这是一个调试模式，用于检测编译器对动态形状的处理
- 会导致性能下降
- 需要特定的配置（VLLM_USE_BYTECODE_HOOK=0）
- 主要用于开发和调试，不适合生产环境

**启用场景**: 仅在开发和调试动态形状处理问题时使用。

---

### 3. `compile_mm_encoder: bool = False`
**位置**: `vllm/config/compilation.py:427`

**功能**: 是否编译多模态编码器。

**为何默认禁用**:
- 目前仅在特定平台上对 `Qwen2_5_vl` 等少数模型有效
- 功能还在开发中，兼容性不完善
- 可能导致某些模型崩溃或产生错误输出

**启用场景**: 在支持的平台上使用经过测试的多模态模型时可以启用。

---

### 4. `cudagraph_copy_inputs: bool = False`
**位置**: `vllm/config/compilation.py:510`

**功能**: 是否为 CUDA Graph 复制输入张量。

**为何默认禁用**:
- 如果调用者能保证总是使用相同的输入缓冲区，则不需要复制
- 复制操作会带来额外的性能开销
- 仅在 PIECEWISE cudagraph 模式下有效

**启用场景**: 当输入缓冲区可能变化时，需要启用以确保正确性。

---

### 5. PassConfig 中的融合优化参数

所有以下参数位于 `vllm/config/compilation.py:114-127`，默认值为 `None`（在某些情况下会被设置为 `False`）：

- **`fuse_norm_quant: bool`**: 融合 RMSNorm + 量化操作
- **`fuse_act_quant: bool`**: 融合 SiluMul + 量化操作
- **`fuse_attn_quant: bool`**: 融合注意力 + 量化操作
- **`eliminate_noops: bool`**: 消除无操作
- **`enable_sp: bool`**: 启用序列并行
- **`fuse_gemm_comms: bool`**: 启用异步张量并行
- **`fuse_allreduce_rms: bool`**: 启用 flashinfer allreduce 融合

**为何默认禁用**:
- 这些都是高级优化功能，需要特定的硬件支持（如 FP8）
- 某些融合操作可能影响数值精度
- 需要配合其他优化一起使用（如 `eliminate_noops`）
- 可能在某些模型架构上不兼容

**启用场景**: 在支持 FP8 量化的硬件上，且使用兼容的模型时，可以逐个测试启用。

---

## 缓存配置 (CacheConfig)

### 6. `is_attention_free: bool = False`
**位置**: `vllm/config/cache.py:67`

**功能**: 模型是否为无注意力模型。

**为何默认禁用**:
- 大多数模型使用注意力机制
- 这个值主要由 `ModelConfig` 设置，用于特殊的无注意力架构（如某些 Mamba 模型）

**启用场景**: 自动从模型配置中设置，无需手动修改。

---

### 7. `calculate_kv_scales: bool = False`
**位置**: `vllm/config/cache.py:104`

**功能**: 动态计算 FP8 KV 缓存的 `k_scale` 和 `v_scale`。

**为何默认禁用**:
- 默认情况下，这些缩放因子从模型检查点加载
- 动态计算会增加运行时开销
- 仅在使用 FP8 KV 缓存且检查点中没有缩放因子时需要

**启用场景**: 当 `kv_cache_dtype` 为 FP8 且模型检查点中缺少缩放因子时启用。

---

### 8. `kv_sharing_fast_prefill: bool = False`
**位置**: `vllm/config/cache.py:132`

**功能**: 在 KV 共享设置中启用快速预填充优化。

**为何默认禁用**:
- 这是一个正在开发的功能（Work In Progress）
- 目前即使启用，也不会有实际的预填充优化
- 仅适用于特定的 KV 共享架构（如 YOCO）

**启用场景**: 在未来版本中，当功能完善后，可用于支持 KV 共享的模型。

---

## 并行配置 (ParallelConfig)

### 9. `data_parallel_external_lb: bool = False`
**位置**: `vllm/config/parallel.py:110`

**功能**: 是否使用"外部"数据并行负载均衡模式。

**为何默认禁用**:
- 仅在在线服务且 `data_parallel_size > 0` 时适用
- 用于 Kubernetes 中的"每个 Pod 一个 rank"的广域专家并行设置
- 需要显式提供 `--data-parallel-rank` 参数

**启用场景**: 在 Kubernetes 等容器编排环境中，需要外部负载均衡器时启用。

---

### 10. `data_parallel_hybrid_lb: bool = False`
**位置**: `vllm/config/parallel.py:115`

**功能**: 是否使用"混合"数据并行负载均衡模式。

**为何默认禁用**:
- 仅在在线服务且 `data_parallel_size > 0` 时适用
- 需要配合 `--data-parallel-start-rank` 参数使用
- 在单节点或 headless 模式下不适用

**启用场景**: 在多节点部署中，需要在节点内使用本地负载均衡，节点间使用外部负载均衡时启用。

---

### 11. `enable_expert_parallel: bool = False`
**位置**: `vllm/config/parallel.py:122`

**功能**: 对 MoE 层使用专家并行而非张量并行。

**为何默认禁用**:
- 仅适用于混合专家（MoE）模型
- 需要特定的模型架构支持
- 与传统的张量并行有不同的性能特征

**启用场景**: 在使用大型 MoE 模型时，可以启用以提高专家分配效率。

---

### 12. `enable_eplb: bool = False`
**位置**: `vllm/config/parallel.py:124`

**功能**: 启用专家并行负载均衡。

**为何默认禁用**:
- 需要先启用 `enable_expert_parallel`
- 会引入额外的负载监控和重排开销
- 需要配合 `EPLBConfig` 使用

**启用场景**: 在专家并行模式下，当专家负载不均衡时启用。

---

### 13. `disable_custom_all_reduce: bool = False`
**位置**: `vllm/config/parallel.py:152`

**功能**: 禁用自定义的 all-reduce 内核，回退到 NCCL。

**为何默认禁用**:
- vLLM 的自定义 all-reduce 内核通常性能更好
- 使用标准的 NCCL 作为备选

**启用场景**: 当自定义内核出现问题或在特定硬件上兼容性不佳时启用。

---

### 14. `enable_dbo: bool = False`
**位置**: `vllm/config/parallel.py:155`

**功能**: 启用双批次重叠（Dual Batch Overlap）。

**为何默认禁用**:
- 这是一个高级优化功能，增加了系统复杂性
- 需要额外的内存来管理重叠的批次
- 仅在特定工作负载下才有性能提升

**启用场景**: 在大批次处理且需要隐藏通信延迟时可以尝试启用。

---

### 15. `disable_nccl_for_dp_synchronization: bool = False`
**位置**: `vllm/config/parallel.py:171`

**功能**: 强制数据并行同步使用 Gloo 而非 NCCL。

**为何默认禁用**:
- NCCL 通常在 GPU 间通信中性能更好
- Gloo 主要用于 CPU 或特殊情况

**启用场景**: 在 NCCL 不可用或有问题的环境中启用。

---

### 16. `ray_workers_use_nsight: bool = False`
**位置**: `vllm/config/parallel.py:175`

**功能**: 使用 Nsight 对 Ray worker 进行性能分析。

**为何默认禁用**:
- 这是一个调试和性能分析工具
- 会引入显著的性能开销
- 仅用于开发和性能调优

**启用场景**: 在使用 Ray 分布式执行时，需要进行性能分析时启用。

---

### 17. EPLBConfig 中的参数

- **`log_balancedness: bool = False`** (`vllm/config/parallel.py:67`): 记录专家并行的负载均衡度
  - **为何禁用**: 会引入通信开销，仅用于监控
  
- **`use_async: bool = False`** (`vllm/config/parallel.py:72`): 使用非阻塞的 EPLB
  - **为何禁用**: 异步操作增加复杂性，可能引入竞态条件

---

## 模型配置 (ModelConfig)

### 18. `trust_remote_code: bool = False`
**位置**: `vllm/config/model.py:124`

**功能**: 是否信任远程代码。

**为何默认禁用**:
- 安全考虑：远程代码可能包含恶意代码
- 仅在确认模型来源可信时才应启用

**启用场景**: 当使用需要自定义代码的模型（如某些 HuggingFace 模型）时启用。

---

### 19. `enforce_eager: bool = False`
**位置**: `vllm/config/model.py:183`

**功能**: 强制使用 eager 模式的 PyTorch，禁用 CUDA Graph。

**为何默认禁用**:
- CUDA Graph 可以显著提高性能
- Eager 模式主要用于调试

**启用场景**: 在调试或遇到 CUDA Graph 兼容性问题时启用。

---

### 20. `disable_sliding_window: bool = False`
**位置**: `vllm/config/model.py:201`

**功能**: 禁用滑动窗口功能。

**为何默认禁用**:
- 滑动窗口可以处理超长序列
- 仅在特定情况下需要禁用

**启用场景**: 当模型不支持或不需要滑动窗口时可以禁用。

---

### 21. `disable_cascade_attn: bool = False`
**位置**: `vllm/config/model.py:205`

**功能**: 禁用 V1 的级联注意力。

**为何默认禁用**:
- 级联注意力是一个启发式优化
- 在大多数情况下有益
- 仅在遇到数值问题时才需要禁用

**启用场景**: 当观察到数值不稳定或精度问题时可以尝试禁用。

---

### 22. `skip_tokenizer_init: bool = False`
**位置**: `vllm/config/model.py:211`

**功能**: 跳过分词器和反分词器的初始化。

**为何默认禁用**:
- 大多数应用需要分词器来处理文本输入
- 仅在直接提供 token IDs 时才需要跳过

**启用场景**: 在 tokens-only 模式或已有预分词的输入时启用。

---

### 23. `enable_prompt_embeds: bool = False`
**位置**: `vllm/config/model.py:215`

**功能**: 允许通过 `prompt_embeds` 键传递文本嵌入。

**为何默认禁用**:
- 安全考虑：错误形状的嵌入可能导致引擎崩溃
- 仅应对可信用户启用

**启用场景**: 在需要直接传递嵌入向量（而非文本）时，在受控环境中启用。

---

### 24. `enable_sleep_mode: bool = False`
**位置**: `vllm/config/model.py:258`

**功能**: 启用引擎的休眠模式。

**为何默认禁用**:
- 仅在 CUDA 和 HIP 平台支持
- 需要额外的状态管理
- 可能增加响应延迟

**启用场景**: 在需要节能或低负载期间释放资源时启用。

---

## 调度器配置 (SchedulerConfig)

### 25. `is_multimodal_model: bool = False`
**位置**: `vllm/config/scheduler.py:86`

**功能**: 标识模型是否为多模态模型。

**为何默认禁用**:
- 大多数模型是纯文本模型
- 自动从 `ModelConfig` 推断

**启用场景**: 由系统自动设置，无需手动修改。

---

### 26. `disable_chunked_mm_input: bool = False`
**位置**: `vllm/config/scheduler.py:110`

**功能**: 在启用分块预填充时，不部分调度多模态项。

**为何默认禁用**:
- 允许部分调度可以提高 GPU 利用率
- 仅在 V1 中使用

**启用场景**: 当需要确保多模态输入完整处理时（如避免图像被分割）启用。

---

### 27. `async_scheduling: bool = False`
**位置**: `vllm/config/scheduler.py:133`

**功能**: 启用异步调度。

**为何默认禁用**:
- 异步调度增加系统复杂性
- 与某些功能不兼容（如推测解码、流水线并行）
- 需要仔细调优以避免性能下降

**启用场景**: 在需要减少 GPU 空闲时间，提高吞吐量时启用。

---

## 注意力配置 (AttentionConfig)

### 28. `use_prefill_decode_attention: bool = False`
**位置**: `vllm/config/attention.py:28`

**功能**: 使用独立的预填充和解码内核而非统一的 Triton 内核。

**为何默认禁用**:
- 统一内核更简单，兼容性更好
- 独立内核需要更多开发和测试

**启用场景**: 在特定硬件或工作负载下，独立内核可能有性能优势时启用。

---

### 29. `use_cudnn_prefill: bool = False`
**位置**: `vllm/config/attention.py:35`

**功能**: 使用 cuDNN 进行预填充。

**为何默认禁用**:
- 需要特定的 cuDNN 版本
- 可能不如 FlashAttention 等优化内核性能好

**启用场景**: 在某些 NVIDIA 硬件上可能有性能优势。

---

### 30. `use_trtllm_ragged_deepseek_prefill: bool = False`
**位置**: `vllm/config/attention.py:38`

**功能**: 使用 TensorRT-LLM 的 ragged DeepSeek 预填充。

**为何默认禁用**:
- 仅适用于 DeepSeek 模型
- 需要 TensorRT-LLM 库

**启用场景**: 在使用 DeepSeek 模型且安装了 TensorRT-LLM 时启用。

---

### 31. `disable_flashinfer_prefill: bool = False`
**位置**: `vllm/config/attention.py:45`

**功能**: 禁用 FlashInfer 预填充。

**为何默认禁用**:
- FlashInfer 通常提供良好的性能
- 仅在遇到兼容性问题时需要禁用

**启用场景**: 当 FlashInfer 导致错误或性能问题时启用。

---

### 32. `disable_flashinfer_q_quantization: bool = False`
**位置**: `vllm/config/attention.py:48`

**功能**: 使用 FP8 KV 时，不对 Q 进行 FP8 量化。

**为何默认禁用**:
- Q 量化可以进一步提高性能
- 默认情况下应该启用

**启用场景**: 当 Q 量化导致精度问题时启用（禁用 Q 量化）。

---

## 可观测性配置 (ObservabilityConfig)

### 33. `kv_cache_metrics: bool = False`
**位置**: `vllm/config/observability.py:50`

**功能**: 启用 KV 缓存驻留指标（生命周期、空闲时间、重用间隙）。

**为何默认禁用**:
- 使用采样来减少开销，但仍有性能影响
- 仅用于监控和调试

**启用场景**: 在需要监控 KV 缓存行为时启用。

---

### 34. `cudagraph_metrics: bool = False`
**位置**: `vllm/config/observability.py:58`

**功能**: 启用 CUDA Graph 指标。

**为何默认禁用**:
- 会记录额外的统计信息，有轻微性能开销
- 仅用于监控

**启用场景**: 在需要了解 CUDA Graph 使用情况时启用。

---

### 35. `enable_layerwise_nvtx_tracing: bool = False`
**位置**: `vllm/config/observability.py:62`

**功能**: 启用逐层 NVTX 追踪。

**为何默认禁用**:
- 与 CUDA Graph 不兼容
- 引入显著的性能开销
- 主要用于性能分析和调试

**启用场景**: 在使用 Nsight Systems 进行性能分析时启用。

---

### 36. `enable_mfu_metrics: bool = False`
**位置**: `vllm/config/observability.py:67`

**功能**: 启用模型 FLOPs 利用率（MFU）指标。

**为何默认禁用**:
- 计算 MFU 需要额外的计算
- 仅用于性能分析

**启用场景**: 在需要评估模型计算效率时启用。

---

## 多模态配置 (MultiModalConfig)

### 37. `enable_mm_embeds: bool = False`
**位置**: `vllm/config/multimodal.py:74`

**功能**: 允许传递多模态嵌入。

**为何默认禁用**:
- 安全考虑：错误形状的嵌入可能导致引擎崩溃
- 仅应对可信用户启用

**启用场景**: 在需要直接传递多模态嵌入时，在受控环境中启用。

---

### 38. `interleave_mm_strings: bool = False`
**位置**: `vllm/config/multimodal.py:128`

**功能**: 启用完全交错的多模态提示支持。

**为何默认禁用**:
- 需要特定的模板格式支持
- 可能与某些模型不兼容

**启用场景**: 当使用 `--chat-template-content-format=string` 且需要交错处理多模态内容时启用。

---

### 39. `skip_mm_profiling: bool = False`
**位置**: `vllm/config/multimodal.py:131`

**功能**: 跳过多模态内存分析，仅使用语言主干模型进行分析。

**为何默认禁用**:
- 多模态模型需要准确的内存分析
- 跳过可能导致内存分配不足

**启用场景**: 为了加速引擎启动，但用户需要确保有足够的内存。

---

## 引擎参数 (EngineArgs)

### 40. `data_parallel_hybrid_lb: bool = False`
**位置**: `vllm/engine/arg_utils.py:407`

**功能**: 数据并行混合负载均衡。

**为何默认禁用**: 见 [并行配置](#10-data_parallel_hybrid_lb-bool--false) 中的说明。

---

### 41. `data_parallel_external_lb: bool = False`
**位置**: `vllm/engine/arg_utils.py:408`

**功能**: 数据并行外部负载均衡。

**为何默认禁用**: 见 [并行配置](#9-data_parallel_external_lb-bool--false) 中的说明。

---

### 42. `disable_log_stats: bool = False`
**位置**: `vllm/engine/arg_utils.py:447`

**功能**: 禁用统计日志记录。

**为何默认禁用**:
- 统计日志对监控系统健康很重要
- 开销通常可以忽略

**启用场景**: 在需要最小化日志输出或已有外部监控系统时启用。

---

### 43. `aggregate_engine_logging: bool = False`
**位置**: `vllm/engine/arg_utils.py:448`

**功能**: 使用数据并行时，记录聚合的而非每引擎的统计信息。

**为何默认禁用**:
- 默认显示每个引擎的详细统计信息
- 聚合信息可能隐藏局部问题

**启用场景**: 在数据并行部署中，需要汇总视图时启用。

---

### 44. `enable_lora: bool = False`
**位置**: `vllm/engine/arg_utils.py:481`

**功能**: 启用 LoRA 适配器处理。

**为何默认禁用**:
- LoRA 是可选功能，需要额外的内存和计算
- 仅在使用 LoRA 微调模型时需要

**启用场景**: 当需要动态加载和切换 LoRA 适配器时启用。

---

### 45. `tokens_only: bool = False`
**位置**: `vllm/engine/arg_utils.py:578`

**功能**: 仅使用 token 模式，跳过分词器。

**为何默认禁用**:
- 大多数应用需要文本输入/输出
- 见 [模型配置](#22-skip_tokenizer_init-bool--false) 中的相关说明

**启用场景**: 当输入和输出都是 token IDs 时启用。

---

## 其他配置

### 46. `enable_prefix_caching` (在某些情况下为 False)
**位置**: `vllm/config/cache.py:76` (默认为 True，但可能被禁用)

**功能**: 启用前缀缓存。

**何时被禁用**:
- 在 ARM、POWER、S390X 和 RISC-V CPU 上的 V1 后端
- 某些不支持的模型架构（如 pooling 模式）

**原因**: 平台或架构限制。

---

### 47. `enable_chunked_prefill` (在某些情况下为 False)
**位置**: `vllm/config/scheduler.py` (默认根据模型自动设置)

**功能**: 启用分块预填充。

**何时被禁用**:
- 在 ARM、POWER、S390X 和 RISC-V CPU 上的 V1 后端
- 某些不支持的模型架构（如 pooling 模式）

**原因**: 平台或架构限制。

---

### 48. StructuredOutputsConfig 中的参数

位于 `vllm/config/structured_outputs.py`:

- **`disable_fallback: bool = False`** (line 28): 禁用结构化输出的回退机制
- **`disable_any_whitespace: bool = False`** (line 30): 禁用任意空白处理
- **`disable_additional_properties: bool = False`** (line 35): 禁用额外属性
- **`enable_in_reasoning: bool = False`** (line 45): 在推理中启用结构化输出

这些参数用于微调结构化输出行为，默认禁用以保持默认的宽松行为。

---

### 49. LoRAConfig 参数

- **`fully_sharded_loras: bool = False`** (`vllm/config/lora.py:38`): 完全分片的 LoRA

**为何默认禁用**: 这是一个高级功能，增加了内存管理的复杂性。

---

### 50. KVEventsConfig 参数

- **`enable_kv_cache_events: bool = False`** (`vllm/config/kv_events.py:18`): 启用 KV 缓存事件

**为何默认禁用**: 这是一个特殊功能，用于 KV 缓存的高级监控和管理。

---

### 51. KVTransferConfig 参数

- **`enable_permute_local_kv: bool = False`** (`vllm/config/kv_transfer.py:64`): 启用本地 KV 排列

**为何默认禁用**: 这是 KV 传输的高级优化，仅在特定场景下有效。

---

### 52. ProfilerConfig 参数

位于 `vllm/config/profiler.py`:

- **`torch_profiler_with_flops: bool = False`** (line 40): 启用 FLOPs 计算
- **`torch_profiler_record_shapes: bool = False`** (line 49): 记录张量形状
- **`torch_profiler_with_memory: bool = False`** (line 52): 记录内存使用
- **`ignore_frontend: bool = False`** (line 56): 忽略前端分析

**为何默认禁用**: 这些都是性能分析功能，会引入显著开销，仅用于调试和优化。

---

### 53. SpeculativeConfig 参数

- **`disable_padded_drafter_batch: bool = False`** (`vllm/config/speculative.py:99`): 禁用填充的起草器批次

**为何默认禁用**: 填充可以提高批处理效率，仅在特殊情况下需要禁用。

---

### 54. AsyncEngineArgs 参数

- **`enable_log_requests: bool = False`** (`vllm/engine/arg_utils.py:2008`): 启用请求日志

**为何默认禁用**: 请求日志可能包含敏感信息，且会增加日志量。

---

## 总结

vLLM 中有超过 50 个优化参数默认设置为 `False`。这些参数被禁用的主要原因包括：

1. **兼容性**: 某些优化仅在特定硬件、模型架构或配置下有效
2. **稳定性**: 高级优化可能引入数值不稳定或其他问题
3. **安全性**: 某些功能（如 `trust_remote_code`）有安全风险
4. **性能权衡**: 某些功能会增加开销，仅在特定场景下才有益
5. **开发中**: 某些功能还在开发或测试阶段
6. **调试和分析**: 许多监控和分析功能会影响性能，默认关闭

### 启用建议

在启用这些参数时，建议：

1. **阅读文档**: 充分理解参数的作用和影响
2. **逐个测试**: 不要同时启用多个参数，以便隔离问题
3. **监控指标**: 观察性能、内存使用和输出质量的变化
4. **基准测试**: 在代表性工作负载上进行充分测试
5. **备份配置**: 保留可工作的配置以便回退

### 常用优化组合

对于不同的使用场景，以下是一些常见的优化组合：

**生产部署优化**:
```python
# 启用前缀缓存以提高多轮对话性能
enable_prefix_caching=True

# 启用分块预填充以平衡延迟和吞吐量
enable_chunked_prefill=True

# 对于大型 MoE 模型
enable_expert_parallel=True
enable_eplb=True  # 如果负载不均衡
```

**FP8 量化优化** (需要支持的硬件):
```python
# 编译优化
fuse_norm_quant=True
fuse_act_quant=True
fuse_attn_quant=True
eliminate_noops=True

# KV 缓存优化
calculate_kv_scales=True  # 如果模型中没有预训练的缩放因子
```

**开发和调试**:
```python
# 禁用优化以便调试
enforce_eager=True
disable_log_stats=False

# 启用详细监控
kv_cache_metrics=True
cudagraph_metrics=True
enable_mfu_metrics=True
```

**多模态应用**:
```python
# 跳过多模态分析以加快启动（谨慎使用）
skip_mm_profiling=True

# 编译多模态编码器（如果支持）
compile_mm_encoder=True
```

---

## 参考资源

- vLLM 官方文档: https://docs.vllm.ai/
- GitHub 仓库: https://github.com/vllm-project/vllm
- 配置文件源码: `vllm/config/` 目录

## 文档更新

本文档基于 vLLM 代码库创建。由于代码在不断演进，某些参数的行为和默认值可能会发生变化。
建议定期检查最新的代码和文档。

---

*最后更新: 2025-12-23*
