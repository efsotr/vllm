# DPLB (Data Parallel Load Balancing) Within a Node - Architecture

## Overview

vLLM supports Data Parallel (DP) deployment where model weights are replicated across separate instances/GPUs to process independent batches of requests. This document describes the internal load balancing (DPLB) implementation within a node.

## Key Concepts

### Data Parallel Modes

vLLM supports three distinct data parallel load balancing modes:

1. **Internal Load Balancing (Internal DPLB)** - The focus of this document
2. **External Load Balancing** - Each DP rank is a separate deployment with its own endpoint
3. **Hybrid Load Balancing** - Per-node API servers with local load balancing

### Internal Load Balancing Mode

In internal DPLB mode:
- A single HTTP endpoint is exposed
- Multiple DP engine cores run within the same node (or across multiple nodes)
- API server(s) distribute requests across DP engine cores
- Load balancing is done at the API server level based on engine state

## Architecture Components

### 1. EngineCore Processes

Each data parallel rank runs as a separate "core engine" process:
- **Location**: `vllm/v1/engine/core.py`
- **Purpose**: Executes model inference for assigned requests
- **Communication**: ZMQ sockets with front-end processes
- **Key Features**:
  - Independent KV cache per DP rank
  - Processes batches independently
  - Reports stats (waiting/running queue lengths) to coordinator

### 2. DPCoordinator Process

A dedicated coordinator process manages DP coordination:
- **Location**: `vllm/v1/engine/coordinator.py`
- **Purpose**: Coordinates multiple DP engine ranks and API servers
- **Key Responsibilities**:
  - Collects stats from each DP engine (waiting and running queue lengths)
  - Publishes stats to all front-end API servers for load-balancing decisions
  - Tracks current DP "request wave" number and engine running state
  - Broadcasts START_DP_WAVE messages to move engines from paused to running state

### 3. DPLBAsyncMPClient

The load-balancing client in API server processes:
- **Location**: `vllm/v1/engine/core_client.py` (lines 1178-1280)
- **Purpose**: Distributes incoming requests across DP engine cores
- **Key Features**:
  - Maintains local view of engine load state
  - Implements load balancing algorithm
  - Routes requests to least-loaded engine
  - Tracks in-flight requests for abort handling

### 4. API Server Processes

Multiple API server processes can be deployed:
- Configurable via `--api-server-count` parameter
- Each API server has its own DPLBAsyncMPClient
- All API servers share the same HTTP endpoint (single port)
- Transparent to users - single HTTP endpoint exposed

## Process Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Node                                 │
│                                                              │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │ API Server │  │ API Server │  │ API Server │  (1-N)    │
│  │  Process   │  │  Process   │  │  Process   │           │
│  │            │  │            │  │            │           │
│  │ DPLBClient │  │ DPLBClient │  │ DPLBClient │           │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘           │
│        │                │                │                   │
│        └────────────────┼────────────────┘                   │
│                         │                                    │
│                    ┌────▼────┐                               │
│                    │   DP    │                               │
│                    │Coordinator│                             │
│                    └────┬────┘                               │
│                         │                                    │
│        ┌────────────────┼────────────────┐                   │
│        │                │                │                   │
│   ┌────▼─────┐    ┌────▼─────┐    ┌────▼─────┐            │
│   │ Engine   │    │ Engine   │    │ Engine   │            │
│   │ Core 0   │    │ Core 1   │    │ Core N-1 │            │
│   │ (DP      │    │ (DP      │    │ (DP      │            │
│   │  Rank 0) │    │  Rank 1) │    │  Rank N-1)│            │
│   └──────────┘    └──────────┘    └──────────┘            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Communication Flow

### Request Processing Flow

1. **HTTP Request** → API Server receives request
2. **Load Balancing Decision** → DPLBAsyncMPClient selects engine based on load
3. **Request Routing** → Request sent to chosen EngineCore via ZMQ
4. **Processing** → EngineCore processes request
5. **Response** → Results returned via ZMQ to API Server
6. **HTTP Response** → API Server returns response to client

### Stats Update Flow

1. **Stats Collection** → Each EngineCore periodically sends stats (waiting/running counts)
2. **Coordinator Aggregation** → DPCoordinator collects stats from all engines
3. **Stats Publishing** → Coordinator publishes aggregated stats to all API servers
4. **Client Update** → Each DPLBAsyncMPClient updates its local view
5. **Load Balancing** → Next request uses updated stats for routing decision

### Request Wave Coordination

For MoE models, engines must synchronize:
- Engines alternate between global running/paused states
- "Request wave" number tracks running→paused transitions
- Coordinator broadcasts START_DP_WAVE when engines need to resume
- All-reduce operation synchronizes when all ranks become idle

## Configuration

### Single Node Example

```bash
vllm serve MODEL --data-parallel-size 4 --tensor-parallel-size 2
```
- Requires 8 GPUs (4 DP × 2 TP)
- Single endpoint
- Internal load balancing

### Multi-Node Example

**Head Node (with API server):**
```bash
vllm serve MODEL \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345
```

**Worker Node (headless):**
```bash
vllm serve MODEL --headless \
  --data-parallel-size 4 \
  --data-parallel-size-local 2 \
  --data-parallel-start-rank 2 \
  --data-parallel-address 10.99.48.128 \
  --data-parallel-rpc-port 13345
```

### Key Parameters

- `--data-parallel-size`: Total number of DP ranks
- `--data-parallel-size-local`: Number of DP ranks on current node
- `--data-parallel-start-rank`: Starting DP rank index for this node
- `--data-parallel-address`: IP address of head node
- `--data-parallel-rpc-port`: Port for DP coordination
- `--api-server-count`: Number of API server processes (default: 1)
- `--headless`: Run without API server (worker nodes only)

## References

### Key Source Files

1. **Engine Core**: `vllm/v1/engine/core.py`
2. **Coordinator**: `vllm/v1/engine/coordinator.py`
3. **Client/Load Balancer**: `vllm/v1/engine/core_client.py`
4. **Configuration**: `vllm/config/parallel.py`
5. **Utilities**: `vllm/v1/engine/utils.py`
6. **Argument Parsing**: `vllm/engine/arg_utils.py`

### Test Files

1. **Internal LB Tests**: `tests/v1/distributed/test_internal_lb_dp.py`
2. **External LB Tests**: `tests/v1/distributed/test_external_lb_dp.py`
3. **Hybrid LB Tests**: `tests/v1/distributed/test_hybrid_lb_dp.py`

### Documentation

1. **Data Parallel Deployment**: `docs/serving/data_parallel_deployment.md`
2. **Disaggregated Prefill**: `docs/features/disagg_prefill.md` (related concept)
