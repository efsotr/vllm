# DPLB Implementation Summary

## Quick Overview

**DPLB (Data Parallel Load Balancing)** is vLLM's internal load balancing system for distributing inference requests across multiple data-parallel replicas of a model within a node or across multiple nodes.

## Key Components

### 1. DPLBAsyncMPClient
**Role**: Load balancer in API server processes  
**File**: `vllm/v1/engine/core_client.py`  
**Function**: Routes requests to least-loaded engine core

### 2. DPCoordinator
**Role**: Central coordinator for DP deployment  
**File**: `vllm/v1/engine/coordinator.py`  
**Function**: Collects and publishes engine stats

### 3. EngineCore
**Role**: Inference engine  
**File**: `vllm/v1/engine/core.py`  
**Function**: Processes model inference requests

## Load Balancing Algorithm

**Formula**: `score = waiting_count × 4 + running_count`

**Selection**: Choose engine with minimum score

**Features**:
- O(N) complexity - scans all engines
- Queue-aware - considers both waiting and running requests
- Multi-client aware - handles multiple API servers
- Predictive - updates local counts before coordinator sync

## Configuration Quick Reference

### Single Node (Internal LB)
```bash
vllm serve MODEL --data-parallel-size 4
```

### Multi-Node (Internal LB)
```bash
# Head node
vllm serve MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-address 10.99.48.128

# Worker node
vllm serve MODEL --headless \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-address 10.99.48.128
```

### Scale API Servers
```bash
vllm serve MODEL \
  --data-parallel-size 8 \
  --api-server-count 4
```

## Key Settings

| Setting | Purpose | Default |
|---------|---------|---------|
| `--data-parallel-size` | Total DP ranks | 1 |
| `--data-parallel-size-local` | Local DP ranks | Auto |
| `--data-parallel-start-rank` | Starting rank index | 0 |
| `--api-server-count` | Number of API servers | 1 |
| `--data-parallel-address` | Master node IP | 127.0.0.1 |
| `--data-parallel-rpc-port` | RPC port | 29550 |
| `--headless` | Run without API server | False |

## Request Flow

1. **Client** → HTTP request → **API Server**
2. **API Server** → DPLBAsyncMPClient selects engine
3. **Client** → ZMQ request → **EngineCore**
4. **EngineCore** → Processes inference
5. **EngineCore** → ZMQ response → **API Server**
6. **API Server** → HTTP response → **Client**

## Stats Update Flow

1. **EngineCores** → Report stats (waiting/running counts)
2. **DPCoordinator** → Collect and aggregate stats
3. **DPCoordinator** → Publish to all API servers (every ~100ms)
4. **DPLBAsyncMPClient** → Update local view
5. **DPLBAsyncMPClient** → Use in next routing decision

## Load Balancing Modes

### Internal LB (DPLB)
- Single HTTP endpoint
- Internal load balancing
- Coordinator-based stats
- **Default when DP > 1**

### External LB
- Multiple HTTP endpoints (one per rank)
- No internal load balancing
- External LB required
- **Enabled with `--data-parallel-rank`**

### Hybrid LB
- Multiple HTTP endpoints (one per node)
- Internal LB within nodes
- External LB across nodes
- **Enabled with `--data-parallel-hybrid-lb`**

## Performance Characteristics

### Strengths
✓ Simple and efficient O(N) algorithm  
✓ Queue-aware load distribution  
✓ Multi-API-server support  
✓ Predictive count updates  
✓ Deterministic behavior

### Limitations
✗ O(N) doesn't scale to very large DP (>16)  
✗ 100ms stats latency can cause temporary imbalance  
✗ No KV cache awareness  
✗ Simple scoring formula

### Future Improvements
- Power of Two Choices (P2C) algorithm for large DP
- KV cache aware routing for prefix caching
- Adaptive scoring weights
- Request size awareness

## Testing

### Test Files
- `tests/v1/distributed/test_internal_lb_dp.py`
- `tests/v1/distributed/test_external_lb_dp.py`
- `tests/v1/distributed/test_hybrid_lb_dp.py`

### Test Utilities
- `tests/v1/utils.py` - Prometheus metrics validation
- `check_request_balancing()` - Validates distribution

### Metrics
- `vllm:request_success_total{engine="N"}` - Per-engine request count
- Used to verify balanced distribution

## Common Use Cases

### 1. Single Node, Multiple GPUs
**Goal**: Maximize throughput on single machine  
**Config**: `--data-parallel-size N` where N = GPU count  
**Result**: All GPUs used, single endpoint, internal LB

### 2. Multi-Node Cluster
**Goal**: Scale across multiple machines  
**Config**: Set `--data-parallel-size` to total, `--data-parallel-size-local` per node  
**Result**: Single endpoint on head node, distributed processing

### 3. Separate API and Compute
**Goal**: Dedicated API and GPU nodes  
**Config**: API node with `--data-parallel-size-local 0`, compute node with engines  
**Result**: API node routes to remote engines only

### 4. High API Throughput
**Goal**: Handle many concurrent connections  
**Config**: Increase `--api-server-count`  
**Result**: Multiple API servers, same endpoint, better connection handling

## Monitoring

### Metrics Endpoint
`GET /metrics` - Prometheus format metrics

### Server Info Endpoint
`GET /server_info?config_format=json` - Configuration details

### Key Metrics to Monitor
- Request distribution across engines
- Time to first token (TTFT)
- Inter-token latency (ITL)
- Queue lengths per engine

## Recommendations

### DP Size 2-4
- Use 1-2 API servers
- Default settings work well
- Monitor for balanced distribution

### DP Size 4-8
- Use 2-4 API servers
- Consider dedicated coordinator node
- Monitor API server CPU usage

### DP Size 8-16
- Use 4-8 API servers
- Consider hybrid LB for multi-node
- Monitor network bandwidth

### DP Size >16
- Consider hybrid or external LB
- Watch for O(N) scan overhead
- Future P2C algorithm will help

## Related Features

### Expert Parallel (EP)
Different from DPLB - expert-level parallelism for MoE models  
**Config**: `--enable-expert-parallel --enable-eplb`

### Disaggregated Prefill
Separate prefill and decode instances  
**Docs**: `docs/features/disagg_prefill.md`

### KV Transfer
KV cache transfer between instances  
**Code**: `vllm/distributed/kv_transfer/`

## Code References

### Core Implementation
- `vllm/v1/engine/core_client.py` - Client and load balancer
- `vllm/v1/engine/coordinator.py` - Coordinator process
- `vllm/v1/engine/core.py` - Engine core
- `vllm/v1/engine/utils.py` - Utilities and setup

### Configuration
- `vllm/config/parallel.py` - ParallelConfig class
- `vllm/engine/arg_utils.py` - Argument parsing and validation

### Tests
- `tests/v1/distributed/` - Distribution tests
- `tests/v1/utils.py` - Test utilities

### Documentation
- `docs/serving/data_parallel_deployment.md` - User guide
- `docs/features/disagg_prefill.md` - Related feature

## Additional Resources

### Understanding Documents
1. `DPLB_Architecture.md` - Detailed architecture overview
2. `DPLB_Load_Balancing_Algorithm.md` - Algorithm deep dive
3. `DPLB_Configuration.md` - Complete configuration guide
4. `DPLB_Code_Reference.md` - Code-level reference

### Source Code Comments
The code is well-commented, especially:
- Load balancing algorithm (`core_client.py:1211-1235`)
- Coordinator logic (`coordinator.py:22-56`)
- Config validation (`arg_utils.py:1450-1524`)
