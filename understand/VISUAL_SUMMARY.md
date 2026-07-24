# vLLM Optimization Parameters - Visual Summary

## Parameter Distribution by Category

```
Total: 54 Parameters with False Default

┌─────────────────────────────────────────────────────────────┐
│ Compilation Configuration (11)        ████████████████░░░░░ │ 20.4%
│ Parallel Configuration (10)           ██████████████░░░░░░░ │ 18.5%
│ Other Configuration (8)                ███████████░░░░░░░░░ │ 14.8%
│ Model Configuration (7)                █████████░░░░░░░░░░░ │ 13.0%
│ Attention Configuration (5)            ███████░░░░░░░░░░░░░ │  9.3%
│ Observability Configuration (4)        █████░░░░░░░░░░░░░░░ │  7.4%
│ Engine Arguments (4)                   █████░░░░░░░░░░░░░░░ │  7.4%
│ Cache Configuration (3)                ████░░░░░░░░░░░░░░░░ │  5.6%
│ Scheduler Configuration (3)            ████░░░░░░░░░░░░░░░░ │  5.6%
│ Multimodal Configuration (3)           ████░░░░░░░░░░░░░░░░ │  5.6%
└─────────────────────────────────────────────────────────────┘
```

## Parameters by Purpose

### Performance Optimization (23 params - 42.6%)
- Compilation optimizations (11)
- Cache optimizations (3)
- Parallelism optimizations (7)
- Attention optimizations (2)

### Monitoring & Debugging (8 params - 14.8%)
- Observability metrics (4)
- Profiler configuration (4)

### Model Behavior (11 params - 20.4%)
- Model configuration (7)
- Scheduler configuration (3)
- Multimodal processing (1)

### System Configuration (8 params - 14.8%)
- Engine arguments (4)
- Parallel configuration (4)

### Advanced Features (4 params - 7.4%)
- LoRA support (2)
- Structured outputs (2)

## Risk Level Distribution

```
Low Risk      ████████████████████████░░░░░░░░  60%  (32 params)
Medium Risk   ████████████░░░░░░░░░░░░░░░░░░░░  30%  (16 params)
High Risk     ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░  10%  ( 6 params)
```

**High Risk Parameters** (Security/Stability Concerns):
- `trust_remote_code`
- `enable_prompt_embeds`
- `enable_mm_embeds`
- `evaluate_guards`
- Experimental compilation features
- Advanced profiling features

**Medium Risk Parameters** (Compatibility/Stability):
- FP8 fusion optimizations
- Platform-specific attention backends
- Async scheduling
- Expert parallelism features

**Low Risk Parameters** (Monitoring/Optional Features):
- Metrics collection
- Logging controls
- Optional optimizations
- Helper features

## Hardware Support Matrix

```
Parameter Type          │ CUDA │ ROCm │ TPU │ CPU │
────────────────────────┼──────┼──────┼─────┼─────┤
Kernel Fusion (FP8)     │  ✅  │  ⚠️  │ ❌  │ ❌  │
FlashAttention          │  ✅  │  ✅  │ ❌  │ ❌  │
CUDA Graph              │  ✅  │  ✅  │ ❌  │ ❌  │
Expert Parallel         │  ✅  │  ✅  │ ✅  │ ✅  │
Data Parallel           │  ✅  │  ✅  │ ✅  │ ✅  │
Prefix Caching          │  ✅  │  ✅  │ ⚠️  │ ⚠️* │
Async Scheduling        │  ✅  │  ✅  │ ⚠️  │ ⚠️  │
Profiling/Metrics       │  ✅  │  ✅  │ ✅  │ ✅  │

Legend: ✅ Fully Supported | ⚠️ Partial/Experimental | ❌ Not Supported
* Some CPU architectures (ARM, POWER, RISC-V) have limitations
```

## Most Commonly Enabled Parameters

Based on typical production use cases:

### Top 10 Most Useful
1. `enable_prefix_caching` - Improves multi-turn conversation performance
2. `enable_chunked_prefill` - Balances latency and throughput
3. `enable_expert_parallel` - Essential for MoE models
4. `trust_remote_code` - Required for many HuggingFace models
5. `enable_lora` - Enables LoRA adapter support
6. `calculate_kv_scales` - Needed for FP8 KV cache without pretrained scales
7. `kv_cache_metrics` - Helpful for monitoring
8. `async_scheduling` - Reduces GPU idle time
9. `enable_eplb` - Balances load in expert parallel
10. `fuse_norm_quant` - Performance boost with FP8

### Rarely Used (Expert/Debug Only)
- `evaluate_guards` - Debug only
- `ray_workers_use_nsight` - Profiling only
- `enable_layerwise_nvtx_tracing` - Deep profiling
- `torch_profiler_*` parameters - Detailed profiling
- `disable_custom_all_reduce` - Troubleshooting only

## Performance Impact Spectrum

```
High Impact (>20% improvement possible)
├─ fuse_norm_quant + fuse_act_quant + fuse_attn_quant (FP8)
├─ enable_prefix_caching (multi-turn scenarios)
├─ enable_expert_parallel (MoE models)
└─ async_scheduling (online serving)

Medium Impact (5-20% improvement)
├─ enable_chunked_prefill
├─ enable_dbo (dual batch overlap)
├─ fuse_gemm_comms (high communication workloads)
└─ use_prefill_decode_attention (specific workloads)

Low Impact (<5% improvement)
├─ Most observability/metrics parameters (add overhead)
├─ cudagraph_copy_inputs (minor optimization)
└─ Various platform-specific tweaks

Negative Impact (overhead)
├─ Profiling parameters
├─ Detailed metrics collection
└─ Debug mode features
```

## Decision Tree: Which Parameters to Enable?

```
START
  │
  ├─ Using MoE model? ──YES──> enable_expert_parallel=True
  │                              ├─ Load imbalanced? ──YES──> enable_eplb=True
  │                              └─ NO ──> Continue
  │
  ├─ Have FP8 hardware? ──YES──> Enable FP8 fusion suite:
  │                                fuse_norm_quant, fuse_act_quant,
  │                                fuse_attn_quant, eliminate_noops
  │
  ├─ Multi-turn conversations? ──YES──> enable_prefix_caching=True
  │
  ├─ Need LoRA support? ──YES──> enable_lora=True
  │
  ├─ Online serving? ──YES──> async_scheduling=True (if compatible)
  │
  ├─ Need monitoring? ──YES──> Enable metrics:
  │                              kv_cache_metrics, cudagraph_metrics, etc.
  │
  ├─ Debugging issues? ──YES──> enforce_eager=True
  │                              + Enable relevant profiling
  │
  └─ Production deployment ──> Start with defaults, profile, then tune
```

## Configuration Templates

### 🚀 High Throughput (Batch Processing)
```python
enable_prefix_caching = True
enable_chunked_prefill = True
async_scheduling = True  # if supported
data_parallel_size = 4  # scale as needed
```

### ⚡ Low Latency (Real-time)
```python
enable_prefix_caching = True
enforce_eager = False  # Keep CUDA Graph
max_num_seqs = 256  # Lower batch size
```

### 💾 Memory Constrained
```python
enable_prefix_caching = True
cpu_offload_gb = 10  # Offload to CPU
gpu_memory_utilization = 0.8  # Conservative
```

### 🔬 FP8 Optimized (H100+)
```python
# Fusion suite
fuse_norm_quant = True
fuse_act_quant = True
fuse_attn_quant = True
eliminate_noops = True

# Cache
calculate_kv_scales = True
kv_cache_dtype = "fp8"
```

### 🔍 Debug Mode
```python
enforce_eager = True
disable_log_stats = False
kv_cache_metrics = True
cudagraph_metrics = True
enable_mfu_metrics = True
```

### 🎯 MoE Specialist
```python
enable_expert_parallel = True
enable_eplb = True
expert_placement_strategy = "round_robin"
all2all_backend = "allgather_reducescatter"
```

## Summary Statistics

- **Total parameters analyzed**: 54
- **Source files examined**: 13 config files + engine args
- **Code locations**: `vllm/config/` and `vllm/engine/arg_utils.py`
- **Categories**: 10 major categories
- **Documentation languages**: Chinese + English
- **Documentation size**: ~38KB (detailed) + 8KB (README)

---

*This visualization is based on the comprehensive analysis in the understand/ directory*
