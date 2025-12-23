# vLLM Optimization Parameters Analysis - Parameters with False Default Values

This document provides a detailed explanation of all optimization parameters in vLLM that have a default value of `False`. These parameters are typically disabled to maintain compatibility, stability, or because they are advanced features that require specific conditions to enable.

## Table of Contents

1. [Overview](#overview)
2. [Parameter Categories](#parameter-categories)
3. [Complete Parameter List](#complete-parameter-list)
4. [Usage Recommendations](#usage-recommendations)

---

## Overview

vLLM contains **over 50 optimization parameters** that are set to `False` by default. These parameters span across multiple configuration categories:

- **Compilation optimizations**: Kernel fusion, graph optimization
- **Cache optimizations**: KV cache management, prefix caching
- **Parallelism**: Expert parallel, data parallel, load balancing
- **Model behavior**: Attention mechanisms, tokenization
- **Observability**: Metrics, profiling, tracing
- **Multimodal**: Image/video processing optimizations

### Why Are They Disabled by Default?

Parameters are disabled by default for several reasons:

1. **Hardware/Platform Limitations**: Only work on specific hardware (e.g., CUDA, ROCm)
2. **Model Architecture Requirements**: Only applicable to certain model types
3. **Stability Concerns**: May introduce numerical instability or compatibility issues
4. **Security**: Features that could pose security risks (e.g., `trust_remote_code`)
5. **Performance Trade-offs**: Add overhead that only benefits specific workloads
6. **Development Status**: Features still under development or testing
7. **Debugging/Monitoring**: Tools that impact performance and are only needed for analysis

---

## Parameter Categories

### Compilation Configuration (CompilationConfig)

| Parameter | Location | Purpose | When to Enable |
|-----------|----------|---------|----------------|
| `enable_qk_norm_rope_fusion` | compilation.py:147 | Fuse Q/K RMSNorm + RoPE operations | CUDA/ROCm platforms with RoPE models |
| `evaluate_guards` | compilation.py:268 | Debug mode for dynamic shape specialization | Development/debugging only |
| `compile_mm_encoder` | compilation.py:427 | Compile multimodal encoder | Supported models like Qwen2_5_vl |
| `cudagraph_copy_inputs` | compilation.py:510 | Copy input tensors for CUDA Graph | When input buffers may change |
| `fuse_norm_quant` | compilation.py:114 | Fuse RMSNorm + quantization | FP8 hardware, compatible models |
| `fuse_act_quant` | compilation.py:116 | Fuse SiluMul + quantization | FP8 hardware, compatible models |
| `fuse_attn_quant` | compilation.py:118 | Fuse attention + quantization | FP8 hardware, compatible models |
| `eliminate_noops` | compilation.py:120 | Remove no-op operations | With other fusion optimizations |
| `enable_sp` | compilation.py:122 | Enable sequence parallelism | Large sequence lengths |
| `fuse_gemm_comms` | compilation.py:124 | Enable async tensor parallel | High-communication workloads |
| `fuse_allreduce_rms` | compilation.py:126 | FlashInfer allreduce fusion | Supported hardware/world size |

### Cache Configuration (CacheConfig)

| Parameter | Location | Purpose | When to Enable |
|-----------|----------|---------|----------------|
| `is_attention_free` | cache.py:67 | Mark model as attention-free | Mamba-like architectures |
| `calculate_kv_scales` | cache.py:104 | Dynamically compute FP8 KV scales | FP8 cache without checkpoint scales |
| `kv_sharing_fast_prefill` | cache.py:132 | Fast prefill for KV sharing | Future: YOCO-style models |

### Parallel Configuration (ParallelConfig)

| Parameter | Location | Purpose | When to Enable |
|-----------|----------|---------|----------------|
| `data_parallel_external_lb` | parallel.py:110 | External DP load balancing | Kubernetes/container orchestration |
| `data_parallel_hybrid_lb` | parallel.py:115 | Hybrid DP load balancing | Multi-node with local+external LB |
| `enable_expert_parallel` | parallel.py:122 | Expert parallelism for MoE | Large MoE models |
| `enable_eplb` | parallel.py:124 | Expert parallel load balancing | Imbalanced expert loads |
| `disable_custom_all_reduce` | parallel.py:152 | Fall back to NCCL all-reduce | Custom kernel issues |
| `enable_dbo` | parallel.py:155 | Dual batch overlap | High-throughput workloads |
| `disable_nccl_for_dp_synchronization` | parallel.py:171 | Use Gloo instead of NCCL | NCCL unavailable/problematic |
| `ray_workers_use_nsight` | parallel.py:175 | Profile Ray workers with Nsight | Performance analysis |
| `log_balancedness` | parallel.py:67 | Log expert load balance | Monitoring expert usage |
| `use_async` | parallel.py:72 | Non-blocking EPLB | Advanced load balancing |

### Model Configuration (ModelConfig)

| Parameter | Location | Purpose | When to Enable |
|-----------|----------|---------|----------------|
| `trust_remote_code` | model.py:124 | Trust remote code execution | Verified trusted sources only |
| `enforce_eager` | model.py:183 | Force eager PyTorch mode | Debugging, CUDA Graph issues |
| `disable_sliding_window` | model.py:201 | Disable sliding window | Model doesn't support/need it |
| `disable_cascade_attn` | model.py:205 | Disable cascade attention (V1) | Numerical stability issues |
| `skip_tokenizer_init` | model.py:211 | Skip tokenizer initialization | Tokens-only mode |
| `enable_prompt_embeds` | model.py:215 | Allow passing text embeddings | Trusted users only |
| `enable_sleep_mode` | model.py:258 | Enable engine sleep mode | Energy saving on CUDA/HIP |

### Scheduler Configuration (SchedulerConfig)

| Parameter | Location | Purpose | When to Enable |
|-----------|----------|---------|----------------|
| `is_multimodal_model` | scheduler.py:86 | Mark as multimodal model | Auto-detected from model |
| `disable_chunked_mm_input` | scheduler.py:110 | No partial MM scheduling | Ensure complete MM processing |
| `async_scheduling` | scheduler.py:133 | Enable async scheduling | Reduce GPU idle time |

### Attention Configuration (AttentionConfig)

| Parameter | Location | Purpose | When to Enable |
|-----------|----------|---------|----------------|
| `use_prefill_decode_attention` | attention.py:28 | Separate prefill/decode kernels | Specific hardware optimization |
| `use_cudnn_prefill` | attention.py:35 | Use cuDNN for prefill | Specific NVIDIA GPUs |
| `use_trtllm_ragged_deepseek_prefill` | attention.py:38 | TRT-LLM DeepSeek prefill | DeepSeek models with TRT-LLM |
| `disable_flashinfer_prefill` | attention.py:45 | Disable FlashInfer prefill | Compatibility issues |
| `disable_flashinfer_q_quantization` | attention.py:48 | No Q quantization with FP8 KV | Precision concerns |

### Observability Configuration (ObservabilityConfig)

| Parameter | Location | Purpose | When to Enable |
|-----------|----------|---------|----------------|
| `kv_cache_metrics` | observability.py:50 | KV cache residency metrics | Monitoring cache behavior |
| `cudagraph_metrics` | observability.py:58 | CUDA Graph metrics | Monitoring graph usage |
| `enable_layerwise_nvtx_tracing` | observability.py:62 | Layer-wise NVTX tracing | Performance profiling |
| `enable_mfu_metrics` | observability.py:67 | Model FLOPs Utilization | Performance analysis |

### Multimodal Configuration (MultiModalConfig)

| Parameter | Location | Purpose | When to Enable |
|-----------|----------|---------|----------------|
| `enable_mm_embeds` | multimodal.py:74 | Allow multimodal embeddings | Trusted users only |
| `interleave_mm_strings` | multimodal.py:128 | Interleaved MM prompt support | String format templates |
| `skip_mm_profiling` | multimodal.py:131 | Skip MM memory profiling | Faster startup (use with caution) |

### Engine Arguments (EngineArgs)

| Parameter | Location | Purpose | When to Enable |
|-----------|----------|---------|----------------|
| `disable_log_stats` | arg_utils.py:447 | Disable statistics logging | Minimize logging |
| `aggregate_engine_logging` | arg_utils.py:448 | Aggregate DP engine logs | Data parallel deployments |
| `enable_lora` | arg_utils.py:481 | Enable LoRA adapter handling | Using LoRA fine-tuned models |
| `tokens_only` | arg_utils.py:578 | Tokens-only mode | Pre-tokenized inputs |

### Other Configurations

| Parameter | Location | Purpose | When to Enable |
|-----------|----------|---------|----------------|
| `disable_fallback` | structured_outputs.py:28 | Disable SO fallback | Strict structured output |
| `disable_any_whitespace` | structured_outputs.py:30 | Disable whitespace handling | Strict formatting |
| `disable_additional_properties` | structured_outputs.py:35 | Disable extra properties | Strict schema validation |
| `enable_in_reasoning` | structured_outputs.py:45 | Enable SO in reasoning | Structured reasoning outputs |
| `fully_sharded_loras` | lora.py:38 | Fully shard LoRAs | Advanced LoRA memory mgmt |
| `enable_kv_cache_events` | kv_events.py:18 | Enable KV cache events | Advanced KV monitoring |
| `enable_permute_local_kv` | kv_transfer.py:64 | Enable local KV permutation | KV transfer optimization |
| `torch_profiler_with_flops` | profiler.py:40 | Enable FLOPs calculation | Detailed profiling |
| `torch_profiler_record_shapes` | profiler.py:49 | Record tensor shapes | Debugging shape issues |
| `torch_profiler_with_memory` | profiler.py:52 | Record memory usage | Memory profiling |
| `ignore_frontend` | profiler.py:56 | Ignore frontend profiling | Backend-only profiling |
| `disable_padded_drafter_batch` | speculative.py:99 | Disable padded drafter | Speculative decoding tuning |
| `enable_log_requests` | arg_utils.py:2008 | Enable request logging | Debugging (privacy concern) |

---

## Complete Parameter List

### Quick Reference Summary

**Total Count**: 54 parameters with `False` default

**By Category**:
- Compilation: 11 parameters
- Cache: 3 parameters  
- Parallel: 10 parameters
- Model: 7 parameters
- Scheduler: 3 parameters
- Attention: 5 parameters
- Observability: 4 parameters
- Multimodal: 3 parameters
- Engine: 4 parameters
- Other: 8 parameters

---

## Usage Recommendations

### General Guidelines

When enabling these parameters:

1. **Read Documentation**: Understand the parameter's purpose and impact
2. **Test Individually**: Enable one parameter at a time to isolate effects
3. **Monitor Metrics**: Watch performance, memory usage, and output quality
4. **Benchmark**: Test on representative workloads
5. **Keep Backups**: Maintain working configurations for rollback

### Common Optimization Patterns

#### Production Deployment
```python
# Enable prefix caching for multi-turn conversations
enable_prefix_caching=True

# Enable chunked prefill for balanced latency/throughput
enable_chunked_prefill=True

# For large MoE models
enable_expert_parallel=True
enable_eplb=True  # if load imbalanced
```

#### FP8 Quantization (requires supported hardware)
```python
# Compilation optimizations
fuse_norm_quant=True
fuse_act_quant=True
fuse_attn_quant=True
eliminate_noops=True

# KV cache optimization
calculate_kv_scales=True  # if no pretrained scales
```

#### Development & Debugging
```python
# Disable optimizations for debugging
enforce_eager=True
disable_log_stats=False

# Enable detailed monitoring
kv_cache_metrics=True
cudagraph_metrics=True
enable_mfu_metrics=True
```

#### Multimodal Applications
```python
# Skip MM profiling for faster startup (use cautiously)
skip_mm_profiling=True

# Compile MM encoder (if supported)
compile_mm_encoder=True
```

### Performance vs. Stability Trade-offs

| Optimization Type | Performance Gain | Stability Risk | Recommendation |
|-------------------|------------------|----------------|----------------|
| Kernel Fusion (FP8) | High | Medium | Test thoroughly on target hardware |
| Async Scheduling | Medium | Low-Medium | Good for online serving |
| Expert Parallel | High (MoE) | Low | Safe for MoE models |
| CUDA Graph | High | Low | Usually safe, enabled by default |
| Custom All-Reduce | Medium | Low | Rarely need to disable |
| Profiling/Metrics | N/A (overhead) | Very Low | Enable for monitoring |

### Platform-Specific Considerations

**CUDA GPUs (NVIDIA)**:
- Most optimizations available
- FP8 support on H100, Ada generation
- FlashAttention available on Ampere+

**ROCm GPUs (AMD)**:
- Some CUDA optimizations ported
- FP8 support on MI300 series
- Check platform-specific documentation

**TPU**:
- Different optimization set
- Ray required for distributed
- Refer to TPU-specific guides

**CPU**:
- Limited optimization support
- Some features disabled on specific architectures (ARM, POWER, RISC-V)
- Focus on parallelism over kernel optimizations

---

## Resources

- **Official Documentation**: https://docs.vllm.ai/
- **GitHub Repository**: https://github.com/vllm-project/vllm
- **Configuration Source**: `vllm/config/` directory
- **Chinese Version**: See `optimization_parameters_false.md`

---

## Document Updates

This document is based on the vLLM codebase as of **2025-12-23**. As the code evolves, parameter behavior and defaults may change. Please refer to the latest code and documentation for up-to-date information.

---

## Contributing

If you find errors or have suggestions for this documentation, please:
1. Check the latest source code to verify accuracy
2. Open an issue or pull request on GitHub
3. Include specific file locations and line numbers

---

*Last Updated: 2025-12-23*
