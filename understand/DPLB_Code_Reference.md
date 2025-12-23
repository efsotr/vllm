# DPLB Code Reference

## Overview

This document provides a detailed code-level reference for the Data Parallel Load Balancing (DPLB) implementation in vLLM.

## Core Classes and Methods

### 1. DPLBAsyncMPClient

**Location**: `vllm/v1/engine/core_client.py` (lines 1178-1280)

**Purpose**: Asyncio-compatible client for multi-engine data parallel deployment with internal load balancing.

#### Class Definition

```python
class DPLBAsyncMPClient(DPAsyncMPClient):
    """Asyncio-compatible client for multi-proc, multi-engine (data parallel)
    EngineCore. Load-balances between multiple engine processes."""
```

#### Key Methods

##### `__init__()`
**Lines**: 1182-1209

Initializes the load balancing client.

```python
def __init__(
    self,
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    client_addresses: dict[str, str] | None = None,
    client_count: int = 1,
    client_index: int = 0,
)
```

**Key Initialization**:
- `self.client_count`: Number of API server processes
- `self.reqs_in_flight`: Dict tracking which engine handles each request
- `self.eng_start_index`: Starting point for round-robin when engines equally loaded

##### `get_core_engine_for_request()`
**Lines**: 1211-1235

**Core load balancing logic** - selects which engine should process a request.

```python
def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
    # Engines are in rank order.
    if (eng_index := request.data_parallel_rank) is None:
        current_counts = self.lb_engines
        # TODO use P2C alg for larger DP sizes
        num_engines = len(current_counts)
        min_score = sys.maxsize
        eng_index = 0
        for i in range(num_engines):
            # Start from client_index to help with balancing when engines
            # are empty.
            idx = (self.eng_start_index + i) % num_engines
            waiting, running = current_counts[idx]
            score = waiting * 4 + running
            if score < min_score:
                min_score = score
                eng_index = idx
        # Increment local waiting count for better balancing between stats
        # updates from the coordinator (which happen every 100ms).
        current_counts[eng_index][0] += self.client_count

    chosen_engine = self.core_engines[eng_index]
    # Record which engine is chosen for this request, to handle aborts.
    self.reqs_in_flight[request.request_id] = chosen_engine
    return chosen_engine
```

**Algorithm**:
1. Check if request specifies target rank (bypass LB if so)
2. Iterate through all engines starting from `eng_start_index`
3. Calculate score: `waiting * 4 + running`
4. Select engine with minimum score
5. Increment local waiting count
6. Record engine assignment for abort handling

##### `abort_requests_async()`
**Lines**: 1256-1271

Routes abort requests to the correct engines.

```python
async def abort_requests_async(self, request_ids: list[str]) -> None:
    if not request_ids or self.resources.engine_dead:
        return

    if len(request_ids) == 1:
        # Fast-path common case.
        if engine := self.reqs_in_flight.get(request_ids[0]):
            await self._abort_requests(request_ids, engine)
        return

    by_engine = defaultdict[EngineIdentity, list[str]](list)
    for req_id in request_ids:
        if engine := self.reqs_in_flight.get(req_id):
            by_engine[engine].append(req_id)
    for engine, req_ids in by_engine.items():
        await self._abort_requests(req_ids, engine)
```

**Features**:
- Fast path for single request abort
- Groups multiple aborts by target engine
- Uses `reqs_in_flight` tracking

##### `process_engine_outputs()`
**Lines**: 1248-1254

Cleans up finished requests from tracking dict.

```python
@staticmethod
async def process_engine_outputs(
    self: "DPLBAsyncMPClient", outputs: EngineCoreOutputs
):
    if outputs.finished_requests and self.reqs_in_flight:
        for req_id in outputs.finished_requests:
            self.reqs_in_flight.pop(req_id, None)
```

### 2. DPAsyncMPClient

**Location**: `vllm/v1/engine/core_client.py` (lines 1022-1176)

**Purpose**: Base class for data parallel clients (used for external LB mode).

#### Key Methods

##### `__init__()`
**Lines**: 1026-1092

Sets up ZMQ sockets and handshakes with engines.

**Key Fields**:
- `self.core_engines`: List of engine identities
- `self.engine_ranks_managed`: Ranks this client manages
- `self.input_socket`: ZMQ ROUTER socket for requests
- `self.output_socket`: ZMQ PULL socket for responses

##### `_start_stats_listener()`
**Lines**: 1094-1176

Background task that listens for stats updates from coordinator.

```python
async def _start_stats_listener(self):
    # ... setup ...
    
    while not self.resources.engine_dead:
        try:
            with memoryview(await recv_async(self.resources.stats_socket)) as buf:
                if buf is None:
                    continue

                # Update local load-balancing state.
                counts, wave, running = msgspec.msgpack.decode(buf)
                self.current_wave = wave
                self.engines_running = running
                self.lb_engines = counts
```

**Purpose**:
- Receives periodic stats from DPCoordinator
- Updates `self.lb_engines` with latest queue counts
- Updates `self.current_wave` and `self.engines_running`
- Runs continuously in background

### 3. DPCoordinator

**Location**: `vllm/v1/engine/coordinator.py` (lines 22-100)

**Purpose**: Coordinator process for data parallel deployments.

#### Class Definition

```python
class DPCoordinator:
    """Coordinator process used for data-parallel deployments (DP>1).

    Intermediates between multiple DP engine rank processes and one or more
    front-end API server processes.

    * Collects stats from each DP engine (currently just waiting and running
      queue lengths), and publishes these to all front-ends for use in
      load-balancing decisions.

    * Keeps track of the current DP "request wave" number and running state
      of the engines.
    """
```

#### Key Methods

##### `__init__()`
**Lines**: 58-100

Spawns coordinator subprocess.

```python
def __init__(self, parallel_config: ParallelConfig):
    dp_size = parallel_config.data_parallel_size
    assert dp_size > 1, "Coordinator only used for data parallel"

    # ... address setup ...

    context = get_mp_context()
    self.proc: multiprocessing.Process = context.Process(
        target=DPCoordinatorProc.run_coordinator,
        name="VLLM_DP_Coordinator",
        kwargs={
            "engine_count": parallel_config.data_parallel_size,
            "front_publish_address": front_publish_address,
            "back_output_address": back_output_address,
            "back_publish_address": back_publish_address,
        },
        daemon=True,
    )
    self.proc.start()
```

##### `get_stats_publish_address()`
**Lines**: 96-97

Returns ZMQ address for stats subscription.

```python
def get_stats_publish_address(self) -> str:
    return self.stats_publish_address
```

### 4. DPCoordinatorProc

**Location**: `vllm/v1/engine/coordinator.py` (lines 103-286)

**Purpose**: The actual coordinator process implementation.

#### Key Methods

##### `run_coordinator()`
**Lines**: 106-286

Main coordinator loop (static method).

```python
@staticmethod
def run_coordinator(
    engine_count: int,
    front_publish_address: str,
    back_output_address: str,
    back_publish_address: str,
)
```

**Main Loop**:
```python
while True:
    # Poll for incoming messages from engines
    events = dict(poller.poll(timeout_ms))
    
    # Handle engine stats updates
    if events.get(back_output_socket) == zmq.POLLIN:
        # Receive stats from engine
        identity = back_output_socket.recv()
        buf = back_output_socket.recv()
        eng_index = int.from_bytes(identity, "little")
        
        # Decode stats
        eng_stats = EngineStats.from_packed(buf)
        
        # Store in engine_stats array
        engine_stats[eng_index] = eng_stats
        
        # Update tracking variables
        last_stats_step = eng_stats.step_counter
        last_stats_wave = eng_stats.current_wave
        stats_changed = True
    
    # Publish aggregated stats to front-end
    if stats_changed and some_condition:
        counts = [
            [stats.num_waiting_reqs, stats.num_running_reqs]
            for stats in engine_stats
        ]
        message = msgspec.msgpack.encode((counts, current_wave, engines_running))
        front_publish_socket.send(message)
        stats_changed = False
```

**Key Responsibilities**:
1. Poll for messages from engines
2. Collect stats from each engine
3. Aggregate stats
4. Publish to front-end API servers
5. Coordinate request waves for MoE

### 5. EngineCore

**Location**: `vllm/v1/engine/core.py`

**Purpose**: Core inference engine process.

#### Stats Reporting

Stats are reported via the output socket to coordinator:

```python
# From core.py
def send_stats():
    stats = EngineStats(
        num_waiting_reqs=len(self.waiting_queue),
        num_running_reqs=len(self.running_queue),
        step_counter=self.step_counter,
        current_wave=self.current_wave,
    )
    self.output_socket.send(self.identity)
    self.output_socket.send(stats.to_packed())
```

### 6. ParallelConfig

**Location**: `vllm/config/parallel.py` (lines 82-520)

**Purpose**: Configuration dataclass for parallel execution.

#### Key Fields

```python
@dataclass
class ParallelConfig:
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
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
    enable_expert_parallel: bool = False
    enable_eplb: bool = False
```

#### Validation Methods

##### `_verify_data_parallel_config()`
**Lines**: 286-292

```python
if self.data_parallel_size_local > self.data_parallel_size:
    raise ValueError(
        f"data_parallel_size_local ({self.data_parallel_size_local}) "
        f"cannot be greater than data_parallel_size ({self.data_parallel_size})"
    )
```

##### `_get_num_nodes()`
**Lines**: 424-425

```python
return self.data_parallel_size // self.data_parallel_size_local
```

### 7. EngineArgs

**Location**: `vllm/engine/arg_utils.py` (lines 52-820)

**Purpose**: Argument parsing and validation for engine configuration.

#### Data Parallel Arguments

```python
@dataclass
class EngineArgs:
    data_parallel_size: int = 1
    data_parallel_rank: int | None = None
    data_parallel_size_local: int | None = None
    data_parallel_start_rank: int = 0
    data_parallel_backend: str = "mp"
    data_parallel_address: str = "127.0.0.1"
    data_parallel_rpc_port: int = 29550
    data_parallel_master_port: int = 29500
    data_parallel_external_lb: bool = False
    data_parallel_hybrid_lb: bool = False
```

#### Configuration Creation

##### `create_parallel_config()`
**Lines**: 1406-1570

Complex method that:
1. Infers data parallel configuration from environment
2. Validates settings
3. Determines LB mode (internal/external/hybrid)
4. Creates ParallelConfig instance

**Key Logic** (lines 1460-1524):

```python
# Infer data parallel size local for internal dplb
if self.data_parallel_size_local is None:
    self.data_parallel_size_local = max(
        local_world_size // world_size_within_dp, 1
    )

# Determine if external LB
data_parallel_external_lb = (
    self.data_parallel_external_lb or self.data_parallel_rank is not None
)

# Configure based on mode
if data_parallel_external_lb:
    data_parallel_size_local = 1
    self.data_parallel_hybrid_lb = False
elif self.data_parallel_size_local is not None:
    data_parallel_size_local = self.data_parallel_size_local
else:
    # Default internal LB logic
    if data_parallel_size_local == self.data_parallel_size:
        self.data_parallel_external_lb = False
        self.data_parallel_hybrid_lb = False
    # ... more logic ...
```

### 8. EngineStats

**Location**: `vllm/v1/metrics/stats.py` (lines 168-174)

**Purpose**: Statistics structure reported by engines.

```python
@dataclass
class EngineStats:
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    
    # These are used for internal DP load-balancing.
    step_counter: int = 0
    current_wave: int = 0
```

Methods:
- `to_packed()`: Serialize to bytes
- `from_packed()`: Deserialize from bytes

## Client Factory Method

**Location**: `vllm/v1/engine/core_client.py` (lines 106-122)

```python
@staticmethod
def make_client(
    vllm_config: VllmConfig,
    executor_class: type[Executor],
    log_stats: bool,
    client_addresses: dict[str, str] | None = None,
    client_count: int = 1,
    client_index: int = 0,
) -> "EngineCoreClient":
    parallel_config = vllm_config.parallel_config
    if parallel_config.data_parallel_size > 1:
        if parallel_config.data_parallel_external_lb:
            # External load balancer - client per DP rank.
            return DPAsyncMPClient(*client_args)
        # Internal load balancer - client balances to all DP ranks.
        return DPLBAsyncMPClient(*client_args)
    return AsyncMPClient(*client_args)
```

**Decision Tree**:
- DP size = 1 → `AsyncMPClient` (no load balancing)
- DP size > 1 + external LB → `DPAsyncMPClient` (no internal LB)
- DP size > 1 + internal LB → `DPLBAsyncMPClient` (with load balancing)

## Engine Setup Utilities

**Location**: `vllm/v1/engine/utils.py`

### `make_core_engine_actor_manager()`
**Lines**: 760-900

Creates and manages engine core actors/processes.

**Key Logic** (lines 838-858):

```python
if dp_rank == 0:
    # Rank 0 holds Coordinator, so it handshakes with all Cores
    # in both external dplb and internal dplb mode.
    engines_to_handshake = [
        CoreEngine(index=i, local=(i < local_engine_count)) 
        for i in range(dp_size)
    ]
else:
    # Rank > 0 handshakes with just the local cores it is managing.
    assert local_engines_only, (
        "Attempting to launch core_engines from dp_rank > 0, but "
        "found internal DPLB, which is incompatible."
    )
    engines_to_handshake = [
        CoreEngine(index=i, local=True)
        for i in range(dp_rank, dp_rank + local_engine_count)
    ]
```

## ZMQ Communication Patterns

### Socket Types

1. **ROUTER** (API Server → Engine)
   - Used for request routing
   - Bidirectional, identity-based routing

2. **PULL** (API Server ← Engine)
   - Used for response collection
   - One-way, load balanced across receivers

3. **PUB** (Coordinator → API Servers)
   - Used for stats distribution
   - One-to-many broadcast

4. **SUB** (API Server ← Coordinator)
   - Used for stats reception
   - Subscribes to coordinator updates

### Address Format

```python
def get_engine_client_zmq_addr(
    local_only: bool,
    host: str,
    port: int | None = None
) -> str:
    if local_only:
        return f"ipc:///tmp/vllm-{uuid4().hex}"
    else:
        return f"tcp://{host}:{port or random_port()}"
```

- **Local**: IPC socket for single-node
- **Remote**: TCP socket for multi-node

## Test Utilities

**Location**: `tests/v1/utils.py`

### `get_prometheus_metrics()`
**Lines**: 12-66

Fetches and parses Prometheus metrics.

```python
def get_prometheus_metrics(server: RemoteOpenAIServer) -> dict[str, dict[str, float]]:
    response = requests.get(server.url_for("metrics"), timeout=10)
    # Parse Prometheus format
    # Returns: {"metric_name": {"{labels}": value}}
```

### `get_engine_request_counts()`
**Lines**: 69-91

Extracts per-engine request counts.

```python
def get_engine_request_counts(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    success_metrics = metrics.get("vllm:request_success_total", {})
    # Extract engine ID from labels
    # Returns: {"0": count0, "1": count1, ...}
```

### `check_request_balancing()`
**Lines**: 94-126

Validates load distribution across engines.

```python
def check_request_balancing(server: RemoteOpenAIServer, dp_size: int):
    metrics = get_prometheus_metrics(server)
    engine_counts = get_engine_request_counts(metrics)
    
    # Check that all engines received requests
    assert len(engines_with_requests) == dp_size
    
    # Verify reasonable balance
    total_requests = sum(engine_counts.values())
    for count in engine_counts.values():
        assert count > total_requests // (dp_size + 1)
```

## Key Data Structures

### EngineCoreRequest

```python
@dataclass
class EngineCoreRequest:
    request_id: str
    prompt: str
    # ... other fields ...
    data_parallel_rank: int | None = None  # Optional rank override
```

### EngineCoreOutputs

```python
@dataclass
class EngineCoreOutputs:
    request_outputs: list[RequestOutput]
    finished_requests: list[str]
```

### EngineIdentity

```python
EngineIdentity = bytes  # 2-byte engine rank identifier
```

Used as ZMQ socket identity for routing.

## Summary of Key Files

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `core_client.py` | Client and LB | `DPLBAsyncMPClient`, `DPAsyncMPClient` |
| `coordinator.py` | Coordination | `DPCoordinator`, `DPCoordinatorProc` |
| `core.py` | Engine | `EngineCore` |
| `parallel.py` | Config | `ParallelConfig` |
| `arg_utils.py` | Parsing | `EngineArgs`, `create_parallel_config()` |
| `utils.py` | Setup | `make_core_engine_actor_manager()` |
| `stats.py` | Stats | `EngineStats` |
| `tests/v1/utils.py` | Testing | `check_request_balancing()` |

## Call Flow Example

### Request Processing

1. **HTTP Request** arrives at API server
2. **FastAPI handler** creates `EngineCoreRequest`
3. **DPLBAsyncMPClient.get_core_engine_for_request()** selects engine
4. **_send_input()** sends via ZMQ ROUTER socket
5. **EngineCore** receives and processes
6. **EngineCore** sends response via output socket
7. **DPAsyncMPClient._process_outputs()** receives response
8. **FastAPI handler** returns HTTP response

### Stats Update

1. **EngineCore** periodically sends stats
2. **DPCoordinator** receives on output socket
3. **DPCoordinator** aggregates stats from all engines
4. **DPCoordinator** publishes via PUB socket
5. **DPAsyncMPClient._start_stats_listener()** receives on SUB socket
6. **Updates** `self.lb_engines` array
7. **Next request** uses updated stats for routing
