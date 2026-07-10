# DPLB Load Balancing Algorithm

## Overview

The Data Parallel Load Balancing (DPLB) algorithm in vLLM determines which DP engine core should process each incoming request. The goal is to distribute load evenly across all available engine cores while minimizing latency.

## Load Balancing Algorithm

### Implementation Location

- **File**: `vllm/v1/engine/core_client.py`
- **Class**: `DPLBAsyncMPClient`
- **Method**: `get_core_engine_for_request()` (lines 1211-1235)

### Algorithm Details

#### Core Logic

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

### Load Scoring Formula

**Score = waiting_count × 4 + running_count**

Where:
- `waiting_count`: Number of requests in the engine's waiting queue
- `running_count`: Number of requests currently being processed
- Weight of 4 for waiting requests reflects their higher priority impact

### Selection Strategy

1. **Iterate through all engines** starting from a client-specific offset
2. **Calculate score** for each engine based on its queue state
3. **Select minimum score** engine (least loaded)
4. **Update local count** to reflect the new assignment before coordinator update

### Key Features

#### 1. Client-Specific Starting Point

```python
self.eng_start_index = (
    len(self.core_engines) * self.client_index
) // client_count
```

When multiple API servers exist:
- Each starts scanning from a different engine index
- Helps distribute load when engines are empty/equally loaded
- Prevents all API servers from always selecting engine 0 first

#### 2. Local Count Tracking

```python
current_counts[eng_index][0] += self.client_count
```

Purpose:
- Compensate for coordinator update latency (~100ms)
- Prevents multiple API servers from selecting same engine simultaneously
- Improves balancing during burst traffic

#### 3. Manual Rank Override

If `request.data_parallel_rank` is specified:
- Request is routed to that specific rank
- Bypasses load balancing algorithm
- Used for special cases (e.g., prefix caching optimization)

## Stats Collection and Distribution

### Engine Stats Collection

Each EngineCore periodically reports:
- **Waiting queue length**: Requests queued but not yet processing
- **Running queue length**: Requests currently being processed
- **Step counter**: Current processing step number
- **Current wave**: Request wave number for MoE coordination

**Source**: `vllm/v1/metrics/stats.py` (lines 168-174)

```python
class EngineStats:
    num_running_reqs: int = 0
    num_waiting_reqs: int = 0
    
    # These are used for internal DP load-balancing.
    step_counter: int = 0
    current_wave: int = 0
```

### Coordinator Processing

**File**: `vllm/v1/engine/coordinator.py`

The DPCoordinator:
1. **Collects** stats from all engine ranks
2. **Aggregates** the data
3. **Publishes** to all API server processes via ZMQ PUB socket
4. **Update frequency**: Every 100ms (typical)

### Client Stats Update

**File**: `vllm/v1/engine/core_client.py` (lines 1142-1145)

```python
# Update local load-balancing state.
counts, wave, running = msgspec.msgpack.decode(buf)
self.current_wave = wave
self.engines_running = running
self.lb_engines = counts
```

Each API server:
- Subscribes to coordinator's stats updates
- Updates local view of engine loads
- Uses updated counts for next request routing decision

## Load Balancing Modes

### Internal Load Balancing (DPLB)

**Client Class**: `DPLBAsyncMPClient`

Features:
- Client manages **all** DP engine cores (local + remote)
- Active load balancing using queue-based scoring
- Single HTTP endpoint for all requests
- Coordinator-based stats distribution

Configuration:
```python
# Auto-enabled when:
parallel_config.data_parallel_size > 1
and not parallel_config.data_parallel_external_lb
and not parallel_config.data_parallel_hybrid_lb
```

### External Load Balancing

**Client Class**: `DPAsyncMPClient`

Features:
- Each client manages only **one** DP rank
- No internal load balancing
- Multiple HTTP endpoints (one per rank)
- External load balancer required

Configuration:
```bash
--data-parallel-size 4 --data-parallel-rank 0  # For rank 0
--data-parallel-size 4 --data-parallel-rank 1  # For rank 1
# etc.
```

### Hybrid Load Balancing

Features:
- Each node's API servers manage **local** DP ranks only
- Per-node load balancing
- Multiple HTTP endpoints (one per node)
- External load balancer distributes across nodes

Configuration:
```bash
--data-parallel-size 4 \
--data-parallel-size-local 2 \
--data-parallel-hybrid-lb
```

## Algorithm Analysis

### Time Complexity

- **O(N)** where N is the number of DP engine cores
- Linear scan through all engines to find minimum score
- Acceptable for typical DP sizes (2-16 ranks)

### Space Complexity

- **O(N)** for storing engine load counts
- **O(R)** for tracking in-flight requests, where R is concurrent request count

### Future Improvements

From code comment (line 1215):
```python
# TODO use P2C alg for larger DP sizes
```

**Power of Two Choices (P2C)** algorithm:
- Sample 2 random engines instead of scanning all
- Choose the less loaded of the two
- **O(1)** time complexity
- Good balance between randomness and load awareness
- Better for large DP deployments (>16 ranks)

## Request Abort Handling

When requests are aborted, they must be routed to the correct engine:

```python
# From lines 1256-1271
async def abort_requests_async(self, request_ids: list[str]) -> None:
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

Key points:
- Track which engine is processing each request
- Route abort to the correct engine
- Batch aborts by engine for efficiency

## Testing and Validation

### Test Implementation

**File**: `tests/v1/utils.py`

The `check_request_balancing()` function validates load distribution:

```python
def check_request_balancing(server: RemoteOpenAIServer, dp_size: int):
    metrics = get_prometheus_metrics(server)
    engine_counts = get_engine_request_counts(metrics)
    
    # Check that all engines received requests
    engines_with_requests = [
        engine for engine, count in engine_counts.items() if count > 0
    ]
    assert len(engines_with_requests) == dp_size
    
    # Verify reasonable balance
    total_requests = sum(engine_counts.values())
    for count in engine_counts.values():
        assert count > total_requests // (dp_size + 1)
```

### Metrics Used

**Prometheus Metric**: `vllm:request_success_total{engine="N"}`

- Tracks completed requests per engine
- Used to validate load distribution
- Accessible via `/metrics` endpoint

### Test Scenarios

From `tests/v1/distributed/test_internal_lb_dp.py`:

1. **Single request** - Basic functionality
2. **Burst traffic** - 200 concurrent requests
3. **Multiple bursts** - Sustained load over time
4. **Streaming requests** - Load balancing with streaming
5. **Multi-API server** - Tests with 1 and 4 API servers

## Performance Considerations

### Strengths

1. **Simple and efficient** - O(N) linear scan acceptable for typical sizes
2. **Predictable** - Deterministic algorithm, no randomness
3. **Adaptive** - Responds to actual queue lengths
4. **Multi-client aware** - Handles multiple API servers well

### Limitations

1. **Scale limit** - O(N) becomes expensive for very large DP sizes (>16)
2. **Stats latency** - 100ms coordinator update interval can cause temporary imbalance
3. **No KV cache awareness** - Doesn't consider prefix caching benefits
4. **Simple scoring** - Could incorporate more sophisticated metrics

### Optimization Opportunities

1. **Power of Two Choices (P2C)** - For larger DP deployments
2. **KV cache aware routing** - Route similar prompts to same engine
3. **Adaptive weights** - Adjust waiting/running weight ratio dynamically
4. **Request size awareness** - Consider token counts in scoring
5. **Historical performance** - Track per-engine throughput/latency
