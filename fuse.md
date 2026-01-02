本设计文档旨在建立一个**同构映射（Isomorphic Mapping）**系统，将 DeepSeek 的推理能力无损地投影至阿里云 DashScope 协议空间。我们将此系统划分为两个正交的维度：**规格说明书（The Abstract Specification）**定义了系统的行为边界与状态转换规则；**工程实施方案（The Concrete Implementation）**则定义了该规格在物理层面的构建算子与容错机制。

以下是整理合并后的最终文档：

---

# DeepSeek 接入 DashScope 协议网关：规格定义与工程实现

## 第一部分：技术规格说明书 (The Abstract Specification)

**版本**：1.0.0 | **密级**：CONFIDENTIAL | **状态**：Final Draft
**适用范围**：DeepSeek 研发团队、阿里云百炼网关研发团队

本规格书基于“Generational 顶尖计算机理论科学家”视角构建，将接口集成视为两个复杂分布式系统之间的**状态同步（State Synchronization）与语义映射（Semantic Mapping）**。

### 1. 核心设计公理 (Axiomatic Design Principles)

系统的正确性由以下三条公理保证：

1. **接口同构性（Interface Isomorphism）**：DeepSeek 网关必须在行为上表现为 DashScope 协议的一个严格子集。对于下游客户端（百炼网关）而言，异构模型的调用必须是透明且无感的。
2. **状态可观测性（State Observability）**：系统必须满足全链路可观测性。特别是在 SSE（Server-Sent Events）流式传输中，每一帧数据包（Packet）必须携带完整的元数据（Usage Metrics），以满足精确计量（Metering）的严苛要求。
3. **错误确定性（Error Determinism）**：错误空间必须是有限且闭合的。所有的运行时异常必须被映射到预定义的、已知的状态空间中，杜绝未定义的异常传播（Undefined Behavior）。

### 2. 协议定义 (Protocol Definition)

#### 2.1 通信规约

* **传输层**：HTTP/1.1 或 HTTP/2。
* **应用层**：严格遵循 JSON 格式。
* **端点 (Endpoint)**：`POST /api/v1/services/aigc/text-generation/generation` (由 Partner 网关提供)。
* **兼容性约束**：输入/输出格式必须严格向 DashScope 标准收敛。废弃过时的 `prompt` 字段，统一采用 `messages` 字段以支持多轮对话及复杂系统指令。

#### 2.2 请求头 (Request Headers)

| Header 字段 | 约束 | 值定义 | 说明 |
| --- | --- | --- | --- |
| `Content-Type` | **Mandatory** | `application/json` | 数据载荷格式 |
| `Authorization` | **Mandatory** | `Bearer {API_KEY}` | 标准鉴权凭证 |
| `X-DashScope-SSE` | Optional | `enable` | **流式触发器**：若存在且值为 `enable`，Server **必须** 以 `text/event-stream` 格式响应。 |

#### 2.3 领域模型 (Domain Models)

**2.3.1 Model Domain**

```json
{ "model": "deepseek-v3" } // 或 "deepseek-r1"

```

**2.3.2 Input Domain (Context State)**
`messages` 列表必须保持时间正序，且 `role` 枚举值符合标准集合。

```json
"input": {
    "messages": [
        { "role": "system", "content": "You are a helpful assistant." },
        { "role": "user", "content": "分析黎曼猜想。" }
    ]
}

```

**2.3.3 Parameters Domain (Control State)**
| 字段 | 类型 | 默认值 | 约束与语义 |
| :--- | :--- | :--- | :--- |
| `result_format` | String | `message` | **必须支持**。指定返回数据结构为 message 格式。 |
| `incremental_output` | Boolean | `false` | **流式控制**：若为 `true`，返回增量 delta；若为 `false`，返回全量 buffer。**R1模型强制建议为 true**。 |
| `enable_thinking` | Boolean | `false` | **DeepSeek 特性**。开启思维链，此时 `incremental_output` 必须为 `true`。 |
| `thinking_budget` | Integer | 1024 | **DeepSeek 特性**。限制思考过程 Token 长度。 |
| `enable_search` | Boolean | `false` | 开启联网搜索（会引起 Input Token 膨胀）。 |

*(注：常规参数如 `temperature`, `top_p`, `seed` 等遵循标准定义)*

### 3. 响应拓扑 (Response Topology)

无论是同步还是流式，响应结构必须保持**拓扑一致性**。

#### 3.1 SSE 流式响应 (Streaming Response) - 核心约束

**设计法则**：为了确保百炼网关能够精确计费，Partner 网关**必须**在每一个 SSE 包中透出当前的 `usage` 信息。这与标准 OpenAI 协议（仅在尾包透出）不同，是本系统的**强一致性约束**。

**Payload 结构示例**：

```json
data: {
    "output": {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "正在推导公式...", // Thinking Phase
                    "role": "assistant"
                },
                "finish_reason": "null"
            }
        ]
    },
    "usage": { // 必须随每一包返回全量累计值
        "total_tokens": 55,
        "input_tokens": 50,
        "output_tokens": 5,
        "output_tokens_details": { "reasoning_tokens": 5 }
    },
    "request_id": "uuid..."
}

```

#### 3.2 计量字段 (Usage Metrics)

流式返回的每一包 `usage` 必须包含以下字段的**全量累计值**（Accumulated Value）：

* `input_tokens`: 输入上下文长度（包含 Search 注入的内容）。
* `output_tokens`: 模型已生成的总长度。
* `total_tokens`: 。

**推论**：若发生网络中断，Client 将依据收到的**最后一个有效 SSE 包**中的 `usage` 进行结算。

### 4. 异常处理与状态映射 (Error Mapping)

所有的内部异常必须坍缩（Collapse）为以下标准错误码。

| HTTP Status | DashScope Code | 语义描述 |
| --- | --- | --- |
| 400 | `InvalidParameter` | 参数非法 (e.g., `temperature > 2`) |
| 400 | `DataInspectionFailed` | 内容安全拦截 |
| 429 | `Throttling.*` | 速率或配额限流 |
| 500 | `InternalError` | 上游推理服务故障 |

---

## 第二部分：工程实施方案 (Engineering Implementation)

本章节定义了如何构建上述规格的物理实现。工程目标是构建一个鲁棒的、高并发的**协议适配器（Protocol Adapter）**。

### 1. 架构设计：双模运行时 (Dual-Mode Runtime)

系统需支持两种正交的运行模式，通过入口点分离实现。

* **生产模式 (Production Mode)**: 通过 `main` 入口启动，作为高性能 Proxy 服务，承载实际流量。
* **兼容测试模式 (Compatibility Test Mode)**: 必须保留并暴露以下历史接口与类，作为遗留测试套件（Legacy Test Suite）的**兼容性锚点（Compatibility Anchors）**：
* `create_mock_server`
* `create_app`
* `run_server`
* `MockServer` (Class)
* *设计意图*：在进入测试模式后，系统应退化为原有的硬编码 Mock 行为，确保现有 CI/CD 流程的稳定性（Regression Safety）。



### 2. 核心组件设计 (Component Design)

#### 2.1 接口适配层 (Adapter Layer)

* **职责**：实现数据结构的双向转换。
* **实现**：利用 Pydantic 模型严格定义 DashScope 与 OpenAI/DeepSeek 的数据结构。
* **功能**：执行从“历史遗留格式”到“现代 Chat Completion 格式”的**无损转换**。这一层必须屏蔽字段差异，保证内核逻辑的纯粹性。

#### 2.2 流式状态机 (Streaming State Machine)

* **职责**：管理生成过程中的状态流转，特别是 DeepSeek 特有的 `reasoning_content` 与常规 `content` 的切换。
* **实现**：将推理内容和 Token 计量逻辑封装在独立的**生成器（Generator）**中。
* **约束**：确保 SSE 协议的每一帧输出都严格符合 DashScope 规范，杜绝格式畸形。

#### 2.3 严格的客户端契约执行 (Strict Contract Enforcement)

系统必须严格遵循客户端的意愿（Intent），禁止任何隐式的行为假设：

* **IF** `incremental_output=True` (Streaming):
* 激活 SSE 通道。
* 保留思维链（Chain of Thought）的实时推送。


* **IF** `incremental_output=False` (Default):
* 激活标准 JSON 通道。
* **强制关闭流式**：即使上游模型支持流式，Proxy 也必须在内存中缓冲全量响应后一次性返回。
* *目的*：防止不支持流式的旧版 SDK 发生 Crash。



#### 2.4 并发控制与死锁规避 (Concurrency & Deadlock Prevention)

* **问题模型**：测试代码（Consumer）在 `queue.get()` 阻塞等待，而服务器（Producer）处理完毕后未发送终止信号，导致“生产者-消费者”死锁。
* **解决方案**：
* **主动消费策略**：Proxy 的 Response Queue 必须被主动消费。
* **扁平化存储**：Request Queue 存储“扁平化”的 Payload（字典或基础类型），而非复杂的封装对象，以消除序列化过程中的 `KeyError` 风险。
* **非阻塞/超时机制**：在队列操作中引入超时熔断机制。



### 3. 可观测性工程 (Observability Engineering)

* **调试模式 (Debug Mode)**：当从测试兼容模式启动时，系统必须自动进入 Debug 模式。
* **结构化日志 (Structured Logging)**：
* 记录详细的 Request/Response Payload。
* **On-the-fly 状态监控**：引入 Epoch 时钟，周期性打印系统内部状态（如：当前并发处理请求数 `pending_requests`），确保系统不是黑盒。



### 4. 依赖栈与配置 (Stack & Configuration)

* **Runtime**: Python 3.x, 使用 `uv` 进行包管理。
* **Core Logic**: 基于 `openai-python` SDK 构建上游交互逻辑。
* **Web Framework**: FastAPI (隐含在 `create_app` 上下文中)。
* **Environment Configuration**:
* **模型映射 (Model Mapping)**: 提供开放接口函数，默认映射 `Qwen/Qwen3-8B` 用于测试回落。
* **上游服务配置**:
```python
SILICON_FLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICON_FLOW_API_KEY = os.getenv("SILICON_FLOW_API_KEY", "sk-your-siliconflow-key-here")

```




* **测试集成**: 开发者将直接复制粘贴代码至 `tests/mock_server.py` 运行，因此代码必须自包含且无复杂的外部文件依赖。

### 5. 异常边界防护 (Exception Boundary Protection)

在编码实现时，必须显式防御以下常见运行时错误（Runtime Errors）：

1. **签名不匹配**: `TypeError: create_mock_server() takes 0 positional arguments but 1 was given`。
* *对策*：严格审查函数签名，确保位置参数与关键字参数的兼容性。


2. **属性缺失**: `AttributeError: 'FastAPI' object has no attribute 'responses'`。
* *对策*：确保对象初始化完整，避免在对象未完全构造时访问动态属性。


3. **循环/错误导入**: `ImportError: cannot import name 'MockServer'`.
* *对策*：保持 `tests/mock_server.py` 的命名空间整洁，避免自身递归导入或与标准库命名冲突。确保 `MockServer` 类定义在模块顶层。


4. **向后兼容性断言失败**:
* *对策*：Output Structure 必须同时包含 `text` (遗留字段) 和 `choices` (新标准字段)，以满足旧版 SDK 的校验逻辑。
