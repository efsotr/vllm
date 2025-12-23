# Understanding DPLB (Data Parallel Load Balancing) in vLLM

## 目录 / Table of Contents

This folder contains comprehensive documentation about vLLM's Data Parallel Load Balancing (DPLB) implementation within a node.

## 文档列表 / Document List

### 1. [DPLB_Summary.md](DPLB_Summary.md)
**快速概览 / Quick Overview**

- 最佳的入门文档 / Best starting point
- DPLB 核心概念和组件 / Core concepts and components
- 配置快速参考 / Quick configuration reference
- 常见用例 / Common use cases

**适合阅读对象**: 所有用户 / For: All users

### 2. [DPLB_Architecture.md](DPLB_Architecture.md)
**架构详解 / Architecture Details**

- 系统架构图和组件说明 / System architecture diagrams and components
- 进程间通信流程 / Inter-process communication flows
- 请求处理流程 / Request processing flows
- 配置示例 / Configuration examples

**适合阅读对象**: 架构师、高级工程师 / For: Architects, Senior Engineers

### 3. [DPLB_Load_Balancing_Algorithm.md](DPLB_Load_Balancing_Algorithm.md)
**负载均衡算法 / Load Balancing Algorithm**

- 算法实现详解 / Algorithm implementation details
- 评分公式和选择策略 / Scoring formula and selection strategy
- 统计收集和分发 / Stats collection and distribution
- 性能分析和优化机会 / Performance analysis and optimization opportunities

**适合阅读对象**: 算法工程师、性能优化工程师 / For: Algorithm Engineers, Performance Engineers

### 4. [DPLB_Configuration.md](DPLB_Configuration.md)
**配置指南 / Configuration Guide**

- 所有配置参数详解 / All configuration parameters explained
- 配置模式和验证规则 / Configuration modes and validation rules
- 常见配置模式 / Common configuration patterns
- 性能调优建议 / Performance tuning recommendations

**适合阅读对象**: DevOps 工程师、系统管理员 / For: DevOps Engineers, System Administrators

### 5. [DPLB_Code_Reference.md](DPLB_Code_Reference.md)
**代码参考 / Code Reference**

- 核心类和方法详解 / Core classes and methods explained
- 代码级实现细节 / Code-level implementation details
- 关键数据结构 / Key data structures
- 调用流程示例 / Call flow examples

**适合阅读对象**: 开发者、贡献者 / For: Developers, Contributors

## 问题回答 / Questions Answered

### 现有的在一个 node 内的 dplb 是怎么做的？
### How is DPLB implemented within a node?

**答案见 / See**: 
- `DPLB_Architecture.md` - 整体架构 / Overall architecture
- `DPLB_Code_Reference.md` - 代码实现 / Code implementation

**关键点 / Key Points**:
1. **DPLBAsyncMPClient** 在 API 服务器中实现负载均衡
2. **DPCoordinator** 收集和分发引擎统计信息
3. **EngineCore** 处理实际的推理请求
4. 使用 ZMQ 进行进程间通信

### 有什么 load balance 方面的设置？
### What are the load balance settings?

**答案见 / See**: 
- `DPLB_Configuration.md` - 完整配置指南 / Complete configuration guide
- `DPLB_Summary.md` - 快速配置参考 / Quick configuration reference

**核心配置参数 / Core Configuration Parameters**:
- `--data-parallel-size`: 总 DP 副本数 / Total DP ranks
- `--data-parallel-size-local`: 本地 DP 副本数 / Local DP ranks
- `--api-server-count`: API 服务器进程数 / API server count
- `--data-parallel-address`: 主节点地址 / Master node address
- `--data-parallel-rpc-port`: RPC 通信端口 / RPC port

## DPLB 核心特性 / Core Features

### 1. 负载均衡算法 / Load Balancing Algorithm

**评分公式 / Scoring Formula**:
```
score = waiting_count × 4 + running_count
```

**选择策略 / Selection Strategy**:
- 遍历所有引擎 / Iterate through all engines
- 选择评分最低的引擎 / Select engine with minimum score
- 考虑等待队列和运行队列 / Consider waiting and running queues

### 2. 三种负载均衡模式 / Three Load Balancing Modes

1. **内部负载均衡 (Internal LB)** - 单一端点，内部分发
   - Single endpoint, internal distribution
   
2. **外部负载均衡 (External LB)** - 多端点，外部分发
   - Multiple endpoints, external distribution
   
3. **混合负载均衡 (Hybrid LB)** - 节点内内部，节点间外部
   - Internal per-node, external cross-node

### 3. 统计更新机制 / Stats Update Mechanism

- **频率 / Frequency**: 约每 100ms
- **内容 / Content**: 等待队列长度、运行队列长度
- **传输 / Transport**: ZMQ PUB/SUB 模式

### 4. 多 API 服务器支持 / Multi-API Server Support

- 可配置多个 API 服务器进程
- 每个 API 服务器独立进行负载均衡
- 从不同起始点开始扫描引擎
- 提高并发连接处理能力

## 使用场景 / Use Cases

### 单节点部署 / Single Node Deployment
```bash
vllm serve MODEL --data-parallel-size 4
```
适合: 单机多卡场景 / For: Single machine with multiple GPUs

### 多节点部署 / Multi-Node Deployment
```bash
# 主节点 / Head node
vllm serve MODEL --data-parallel-size 4 --data-parallel-size-local 2

# 工作节点 / Worker node
vllm serve MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 --data-parallel-start-rank 2
```
适合: 集群环境 / For: Cluster environment

### API 和计算分离 / API and Compute Separation
```bash
# API 节点 / API node
vllm serve MODEL --data-parallel-size 4 --data-parallel-size-local 0

# 计算节点 / Compute node
vllm serve MODEL --headless --data-parallel-size 4 --data-parallel-size-local 4
```
适合: 专用 API 和 GPU 节点 / For: Dedicated API and GPU nodes

## 性能考虑 / Performance Considerations

### 优势 / Strengths
✓ 简单高效的 O(N) 算法  
✓ 基于队列的负载感知  
✓ 支持多 API 服务器  
✓ 预测性计数更新  

### 限制 / Limitations
✗ O(N) 在大规模 DP (>16) 时不够高效  
✗ 100ms 统计延迟可能导致短暂不平衡  
✗ 不考虑 KV 缓存状态  
✗ 简单的评分公式  

### 建议 / Recommendations

| DP 大小 | API 服务器数 | 说明 |
|---------|-------------|------|
| 2-4 | 1-2 | 默认配置即可 |
| 4-8 | 2-4 | 考虑增加 API 服务器 |
| 8-16 | 4-8 | API 服务器数约为 DP 大小的一半 |
| 16+ | 8+ | 考虑混合负载均衡模式 |

## 监控和观察 / Monitoring and Observability

### Prometheus 指标 / Prometheus Metrics
- `vllm:request_success_total{engine="N"}` - 每个引擎的请求计数
- `vllm:time_to_first_token_seconds` - TTFT 延迟
- `vllm:time_per_output_token_seconds` - 每个输出 token 的时间

### 端点 / Endpoints
- `/metrics` - Prometheus 格式指标
- `/server_info?config_format=json` - 配置信息

## 相关特性 / Related Features

- **Expert Parallel (EP)**: MoE 模型的专家级并行
- **Disaggregated Prefill**: 分离 prefill 和 decode 实例
- **KV Transfer**: 实例间 KV 缓存传输

## 源代码位置 / Source Code Locations

### 核心实现 / Core Implementation
- `vllm/v1/engine/core_client.py` - 客户端和负载均衡器
- `vllm/v1/engine/coordinator.py` - 协调器进程
- `vllm/v1/engine/core.py` - 引擎核心
- `vllm/v1/engine/utils.py` - 工具和设置

### 配置 / Configuration
- `vllm/config/parallel.py` - ParallelConfig 类
- `vllm/engine/arg_utils.py` - 参数解析和验证

### 测试 / Tests
- `tests/v1/distributed/test_internal_lb_dp.py` - 内部负载均衡测试
- `tests/v1/distributed/test_external_lb_dp.py` - 外部负载均衡测试
- `tests/v1/distributed/test_hybrid_lb_dp.py` - 混合负载均衡测试

### 文档 / Documentation
- `docs/serving/data_parallel_deployment.md` - 用户指南
- `docs/features/disagg_prefill.md` - 相关特性

## 阅读建议 / Reading Recommendations

### 入门 / Beginner
1. 先读 **DPLB_Summary.md** 了解整体概念
2. 再读 **DPLB_Architecture.md** 理解架构
3. 参考 **DPLB_Configuration.md** 进行配置

### 进阶 / Advanced
1. 研究 **DPLB_Load_Balancing_Algorithm.md** 理解算法
2. 深入 **DPLB_Code_Reference.md** 学习实现细节
3. 查看源代码进行验证

### 开发者 / Developer
1. 从 **DPLB_Code_Reference.md** 开始
2. 对照源代码理解实现
3. 参考测试文件学习用法

## 贡献 / Contributing

如果发现文档有误或需要补充，欢迎提交 PR。

If you find errors or want to contribute, PRs are welcome.

## 版本 / Version

文档基于 vLLM 主分支代码创建 (2025-12-23)

Documentation created based on vLLM main branch code (2025-12-23)
