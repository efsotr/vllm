# DPLB Configuration and Settings

## Overview

This document describes all configuration options and settings related to Data Parallel Load Balancing (DPLB) within vLLM.

## Configuration Parameters

### Core Data Parallel Settings

#### 1. `--data-parallel-size`

**Type**: Integer  
**Default**: 1  
**Purpose**: Total number of data parallel ranks across all nodes

**Example**:
```bash
vllm serve MODEL --data-parallel-size 4
```

**Notes**:
- Each rank is a complete model replica
- Requires N GPUs where N = DP_SIZE × TP_SIZE
- When DP_SIZE > 1, enables data parallel coordination

#### 2. `--data-parallel-size-local`

**Type**: Integer  
**Default**: None (auto-inferred)  
**Purpose**: Number of DP ranks on the current node

**Example**:
```bash
# Head node with 2 local ranks
vllm serve MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2

# Worker node with 2 local ranks  
vllm serve MODEL --headless \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2
```

**Auto-inference** (from `arg_utils.py` lines 1460-1464):
```python
if self.data_parallel_size_local is None:
    # Infer data parallel size local for internal dplb:
    self.data_parallel_size_local = max(
        local_world_size // world_size_within_dp, 1
    )
```

**Special values**:
- `0`: API-only node (no engines)
- `None`: Auto-calculated based on available GPUs
- `N`: Explicitly set number of local engines

#### 3. `--data-parallel-start-rank`

**Type**: Integer  
**Default**: 0  
**Purpose**: Starting DP rank index for this node

**Example**:
```bash
# Worker node starting at rank 2
vllm serve MODEL --headless \
  --data-parallel-size 4 \
  --data-parallel-start-rank 2 \
  --data-parallel-size-local 2
```

**Notes**:
- Used in multi-node deployments
- Head node typically uses rank 0
- Worker nodes specify their starting rank

#### 4. `--data-parallel-rank`

**Type**: Integer  
**Default**: 0  
**Purpose**: Explicit DP rank assignment (enables external LB mode)

**Example**:
```bash
# Rank 0
vllm serve MODEL --data-parallel-size 2 --data-parallel-rank 0 --port 8000

# Rank 1
vllm serve MODEL --data-parallel-size 2 --data-parallel-rank 1 --port 8001
```

**Impact**:
- **Implicitly enables external LB mode**
- Each rank gets its own HTTP endpoint
- No internal load balancing
- Requires external load balancer

### Load Balancing Mode Settings

#### 5. `--data-parallel-external-lb`

**Type**: Boolean flag  
**Default**: False  
**Purpose**: Explicitly enable external load balancing mode

**Example**:
```bash
vllm serve MODEL \
  --data-parallel-size 2 \
  --data-parallel-external-lb
```

**Characteristics**:
- Each DP rank has separate endpoint
- No internal load balancing
- Client class: `DPAsyncMPClient`
- External LB required for traffic distribution

#### 6. `--data-parallel-hybrid-lb`

**Type**: Boolean flag  
**Default**: False  
**Purpose**: Enable hybrid load balancing mode

**Example**:
```bash
vllm serve MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-hybrid-lb
```

**Characteristics**:
- Per-node API servers
- Internal LB within each node
- External LB across nodes
- Cannot be used with `--headless`

**Validation** (from `arg_utils.py` lines 1488-1493):
```python
if self.data_parallel_hybrid_lb and data_parallel_size_local == 1:
    logger.warning(
        "data_parallel_size_local = 1, autoswitch to "
        "external lb."
    )
    self.data_parallel_hybrid_lb = False
```

### Network Configuration

#### 7. `--data-parallel-address`

**Type**: String (IP address)  
**Default**: "127.0.0.1"  
**Purpose**: IP address of the data parallel master (head node)

**Example**:
```bash
# Head node
vllm serve MODEL \
  --data-parallel-address 10.99.48.128

# Worker node
vllm serve MODEL --headless \
  --data-parallel-address 10.99.48.128
```

**Notes**:
- All nodes connect to this address
- Must be reachable from all worker nodes
- Used for ZMQ communication

#### 8. `--data-parallel-rpc-port`

**Type**: Integer  
**Default**: 29550  
**Purpose**: Port for data parallel RPC messaging

**Example**:
```bash
vllm serve MODEL \
  --data-parallel-rpc-port 13345
```

**Notes**:
- Used for coordinator-engine communication
- Must be same across all nodes in a deployment
- Default usually works unless port conflict exists

#### 9. `--data-parallel-master-port`

**Type**: Integer  
**Default**: 29500  
**Purpose**: Port of the data parallel master for process group init

**Example**:
```bash
vllm serve MODEL \
  --data-parallel-master-port 29500
```

### Backend Configuration

#### 10. `--data-parallel-backend`

**Type**: String (enum)  
**Values**: "mp" or "ray"  
**Default**: "mp"  
**Purpose**: Backend to use for data parallel deployment

**Multiprocessing (mp) Example**:
```bash
# Node 0
vllm serve MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2

# Node 1  
vllm serve MODEL --headless \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2
```

**Ray Example**:
```bash
# Single command from any node
vllm serve MODEL \
  --data-parallel-size 4 \
  --data-parallel-backend ray
```

**Ray Backend Advantages**:
- Single launch command
- Automatic node discovery
- No need for explicit address/port config
- Automatic resource allocation

**Ray-specific Environment Variable**:
```bash
# For multi-node DP groups
export VLLM_RAY_DP_PACK_STRATEGY="span"
```

### API Server Scaling

#### 11. `--api-server-count`

**Type**: Integer  
**Default**: 1  
**Purpose**: Number of API server processes to spawn

**Example**:
```bash
vllm serve MODEL \
  --data-parallel-size 4 \
  --api-server-count 4
```

**Impact**:
- Scales out API server capacity
- Each API server has own `DPLBAsyncMPClient`
- All share single HTTP port
- Helps prevent API server bottleneck at large DP sizes

**Recommendation**:
- Use 1 API server for small deployments (DP ≤ 4)
- Scale up for larger deployments (e.g., 4 API servers for DP=16)
- Can be adjusted per node in hybrid mode

### Deployment Mode

#### 12. `--headless`

**Type**: Boolean flag  
**Default**: False  
**Purpose**: Run without API server (engines only)

**Example**:
```bash
vllm serve MODEL --headless \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2
```

**Use Cases**:
- Worker nodes in multi-node internal LB deployment
- Headless engine server with separate API-only server
- Enables separation of API and compute resources

**Restrictions**:
- Cannot be used with `--data-parallel-hybrid-lb`
- Must have `data_parallel_size_local > 0`

## Configuration Validation

### Internal vs External LB Mode Detection

From `arg_utils.py` (lines 1465-1480):

```python
data_parallel_external_lb = (
    self.data_parallel_external_lb or self.data_parallel_rank is not None
)

if data_parallel_external_lb:
    assert self.data_parallel_rank is not None
    assert self.data_parallel_size_local in (1, None)
    data_parallel_size_local = 1
    self.data_parallel_hybrid_lb = False
elif self.data_parallel_size_local is not None:
    data_parallel_size_local = self.data_parallel_size_local
```

**Key Rules**:
1. Setting `--data-parallel-rank` enables external LB
2. External LB requires `data_parallel_size_local = 1`
3. External and hybrid LB are mutually exclusive

### Mode Inference Logic

From `arg_utils.py` (lines 1498-1524):

```python
if data_parallel_size_local == self.data_parallel_size:
    # Pure internal LB
    self.data_parallel_external_lb = False
    self.data_parallel_hybrid_lb = False
elif self.data_parallel_hybrid_lb:
    # Hybrid LB mode
    assert data_parallel_size_local is not None
    self.data_parallel_external_lb = False
elif self.data_parallel_rank is not None:
    # Pure external LB
    data_parallel_size_local = 1
else:
    # Default to pure internal LB
    data_parallel_size_local = self.data_parallel_size
```

## ParallelConfig Class

**Location**: `vllm/config/parallel.py` (lines 82-150)

### Key Fields

```python
@dataclass
class ParallelConfig:
    data_parallel_size: int = 1
    data_parallel_size_local: int = 1
    data_parallel_rank: int = 0
    data_parallel_rank_local: int | None = None
    data_parallel_master_ip: str = "127.0.0.1"
    data_parallel_rpc_port: int = 29550
    data_parallel_master_port: int = 29500
    data_parallel_backend: DataParallelBackend = "mp"
    data_parallel_external_lb: bool = False
    data_parallel_hybrid_lb: bool = False
```

### Validation Rules

From `parallel.py` (lines 286-289):

```python
if self.data_parallel_size_local > self.data_parallel_size:
    raise ValueError(
        f"data_parallel_size_local ({self.data_parallel_size_local}) "
        f"cannot be greater than data_parallel_size ({self.data_parallel_size})"
    )
```

## Common Configuration Patterns

### Pattern 1: Single-Node Internal LB

**Use Case**: All GPUs on one node

```bash
vllm serve MODEL \
  --data-parallel-size 4 \
  --tensor-parallel-size 2
```

**Result**: 8 GPUs used (4 DP × 2 TP), single endpoint, internal LB

### Pattern 2: Multi-Node Internal LB (Balanced)

**Use Case**: DP ranks evenly distributed across nodes

```bash
# Node 0 (10.99.48.128)
vllm serve MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345

# Node 1
vllm serve MODEL --headless \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345
```

**Result**: Single endpoint on Node 0, engines on both nodes

### Pattern 3: Multi-Node Internal LB (API Separate)

**Use Case**: Separate API and compute nodes

```bash
# API Node (10.99.48.128)
vllm serve MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 0 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345

# Engine Node
vllm serve MODEL --headless \
  --data-parallel-size 4 \
  --data-parallel-size-local 4 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345
```

**Result**: API-only node, all engines on second node

### Pattern 4: External LB (Single Node)

**Use Case**: Each rank is separate deployment

```bash
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve MODEL \
  --data-parallel-size 2 \
  --data-parallel-rank 0 \
  --port 8000

# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve MODEL \
  --data-parallel-size 2 \
  --data-parallel-rank 1 \
  --port 8001
```

**Result**: Two endpoints, external LB needed

### Pattern 5: Hybrid LB (Multi-Node)

**Use Case**: Per-node API servers with external cross-node LB

```bash
# Node 0
vllm serve MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-hybrid-lb \
  --port 8000

# Node 1
vllm serve MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-hybrid-lb \
  --port 8001
```

**Result**: Two endpoints (one per node), external LB for cross-node

### Pattern 6: Ray-based Internal LB

**Use Case**: Simplified multi-node deployment

```bash
vllm serve MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-backend ray
```

**Result**: Single command, Ray handles node allocation

## Environment Variables

### GPU Assignment

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
```

Used when running multiple instances on same node with external LB.

### Ray Configuration

```bash
export VLLM_RAY_DP_PACK_STRATEGY="span"
```

- **"pack"** (default): Fill nodes sequentially
- **"span"**: Spread across nodes for single DP group

### Development Mode

```bash
export VLLM_SERVER_DEV_MODE="1"
```

Enables development features in tests.

## Monitoring and Observability

### Prometheus Metrics

**Endpoint**: `/metrics`

**Key Metrics**:
- `vllm:request_success_total{engine="N"}`: Requests completed per engine
- `vllm:request_failure_total{engine="N"}`: Failed requests per engine
- `vllm:time_to_first_token_seconds`: TTFT latency
- `vllm:time_per_output_token_seconds`: Inter-token latency

### Server Info Endpoint

**Endpoint**: `/server_info?config_format=json`

**Returns**:
```json
{
  "vllm_config": {
    "parallel_config": {
      "data_parallel_size": 4,
      "data_parallel_size_local": 2,
      "data_parallel_rank": 0,
      "_api_process_count": 4,
      "_api_process_rank": 0
    }
  }
}
```

## Performance Tuning

### Recommendations by DP Size

| DP Size | API Servers | Notes |
|---------|-------------|-------|
| 1 | 1 | No DP, no load balancing |
| 2-4 | 1-2 | Single API server usually sufficient |
| 4-8 | 2-4 | Consider scaling API servers |
| 8-16 | 4-8 | API server count ≈ DP size / 2 |
| 16+ | 8+ | Consider hybrid LB to distribute API load |

### MoE-Specific Settings

For MoE models with Expert Parallel:

```bash
vllm serve MODEL \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --enable-eplb
```

**Impact**:
- Enables Expert Parallelism Load Balancing
- Requires DP coordination even for non-MoE load balancing
- Different from DPLB (data-level) - works at expert-level

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure `data-parallel-rpc-port` is not in use
2. **Network connectivity**: Verify all nodes can reach `data-parallel-address`
3. **GPU visibility**: Check CUDA_VISIBLE_DEVICES matches local config
4. **Rank mismatch**: Ensure start-rank + local-size doesn't exceed total size

### Debug Commands

```bash
# Check what GPUs are visible
nvidia-smi

# Test network connectivity
nc -zv <data-parallel-address> <data-parallel-rpc-port>

# View process tree
ps aux | grep vllm
```
