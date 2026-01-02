这是一个关于 **DeepSeek 模型接入阿里云百炼（Bailian）平台的标准化技术规格说明书**。

基于“Generational 顶尖计算机理论科学家”的视角，本设计文档不局限于简单的字段罗列，而是将接口集成视为两个复杂分布式系统之间的**状态同步（State Synchronization）与语义映射（Semantic Mapping）**。

设计核心遵循以下三条公理：

1. **接口同构性（Interface Isomorphism）**：DeepSeek 网关必须在行为上表现为 DashScope 协议的一个严格子集，确保下游客户端（百炼网关）对异构模型的无感调用。
2. **状态可观测性（State Observability）**：特别是在 SSE（Server-Sent Events）流式传输中，每一帧数据都必须包含完整的元数据（Usage），以满足精确的计量（Metering）需求。
3. **错误确定性（Error Determinism）**：错误必须被映射到有限的、已知的状态空间中，杜绝未定义的异常传播。

---

# DeepSeek 模型接入百炼平台技术规格说明书 (DeepSeek Integration Specification)

**版本**：1.0.0
**密级**：CONFIDENTIAL
**状态**：Draft
**受众**：DeepSeek 研发团队、阿里云百炼网关研发团队

---

## 1. 架构设计原则 (Architectural Principles)

本规范定义了百炼网关（Client）与 DeepSeek 服务网关（Server）之间的交互协议。鉴于 DeepSeek-V3/R1 等模型具有“思考（Thinking）”与“通用生成”的双重特性，本协议采用 **DashScope Message Protocol** 作为唯一规范载体。

### 1.1 协议选择与约束

通信层采用 **HTTP/1.1** 或 **HTTP/2**，应用层数据交换格式严格遵循 **JSON**。

* **传输模式**：支持 **同步（Synchronous）** 与 **SSE 流式（Streaming）** 两种模式。
* **兼容性公理**：Server 端必须实现 DashScope 定义的输入/输出格式。虽然历史上存在 `prompt` 字段（Text-Generation），但本规范要求统一收敛至 `messages` 字段（Chat-Generation），以支持多轮对话及复杂的系统指令。

---

## 2. 接口定义 (Interface Definition)

**Endpoint**: `POST /api/v1/services/aigc/text-generation/generation` (示例路径，实际由 Partner 网关提供)

### 2.1 请求头 (Request Headers)

系统交互必须包含以下头部信息，用于鉴权与协议协商。

| Header 字段 | 必选 | 值定义/约束 | 说明 |
| --- | --- | --- | --- |
| `Content-Type` | **Yes** | `application/json` | 数据载荷格式 |
| `Authorization` | **Yes** | `Bearer {API_KEY}` | 标准 Bearer Token 鉴权 |
| `X-DashScope-SSE` | No | `enable` | **关键**：若存在此 Header 且值为 `enable`，Server **必须** 以 `text/event-stream` 格式返回。 |

### 2.2 请求体 (Request Body)

请求体由 `model`、`input` 和 `parameters` 三个核心域（Domain）组成。

#### 2.2.1 Model Domain

```json
{
  "model": "deepseek-v3" // 或 "deepseek-r1"
}

```

#### 2.2.2 Input Domain (Context State)

`input` 对象承载对话的上下文状态。

```json
"input": {
    "messages": [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "分析一下黎曼猜想。"
        }
    ]
}

```

* **约束**：`role` 枚举值集合 。
* **顺序**：`messages` 列表必须保持时间正序。

#### 2.2.3 Parameters Domain (Control State)

`parameters` 对象控制生成过程的随机性、长度及特定模型行为（如思考）。

| 字段 | 类型 | 必选 | 默认值 | 约束与说明 |
| --- | --- | --- | --- | --- |
| `result_format` | String | No | `message` | **必须支持**。指定返回数据结构为 message 格式。 |
| `max_tokens` | Integer | No | Model Max | 限制生成的最大 Token 数（不包含思考 Token）。 |
| `temperature` | Float | No | 1.0 | 采样温度，范围 。 |
| `top_p` | Float | No | 0.8 | 核采样阈值，范围 。 |
| `top_k` | Integer | No | 100 | 候选集大小。 |
| `seed` | Integer | No | 1234 | 随机种子，用于确定性生成。 |
| `incremental_output` | Boolean | No | `false` | **关键**：流式模式下，若为 `true`，则返回增量 delta；若为 `false`，返回全量 buffer。**R1模型建议强制为 true**。 |
| `enable_thinking` | Boolean | No | `false` | **DeepSeek 特有**。是否开启思维链。开启时，`incremental_output` 必须为 `true`。 |
| `thinking_budget` | Integer | No | 1024 | **DeepSeek 特有**。限制思考过程的 Token 长度。 |
| `enable_search` | Boolean | No | `false` | 是否开启联网搜索。 |

---

## 3. 响应定义 (Response Definition)

设计要求：无论是同步还是流式，响应结构必须保持**拓扑一致性**。

### 3.1 同步响应 (Non-Streaming Response)

当 `X-DashScope-SSE` 未开启时返回。

```json
{
    "output": {
        "text": null, // 兼容字段，Message 模式下通常为 null
        "finish_reason": "stop", // 枚举：null, stop, length
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "黎曼猜想是关于黎曼ζ函数零点分布的猜想...",
                    "reasoning_content": "用户询问黎曼猜想，我需要调用数学知识库..." // 仅在 enable_thinking=true 时出现
                }
            }
        ]
    },
    "usage": {
        "total_tokens": 150,
        "input_tokens": 50,
        "output_tokens": 100,
        "output_tokens_details": {
             "reasoning_tokens": 20, // 思考过程消耗
             "text_tokens": 80       // 最终回答消耗
        }
    },
    "request_id": "uuid-v4-string"
}

```

### 3.2 SSE 流式响应 (Streaming Response) - 核心设计

**关键设计法则**：为了确保百炼网关能够精确计费，Partner 网关**必须**在每一个 SSE 包（Packet）中透出当前的 `usage` 信息。这与标准的 OpenAI 协议（通常仅在最后一句透出）不同，是本系统的**强约束**。

**SSE 包结构示例**：

```json
// Packet N (Thinking Phase)
data: {
    "output": {
        "choices": [
            {
                "message": {
                    "content": "",
                    "reasoning_content": "正在检索...",
                    "role": "assistant"
                },
                "finish_reason": "null"
            }
        ]
    },
    "usage": {
        "total_tokens": 55,
        "input_tokens": 50,
        "output_tokens": 5,
        "output_tokens_details": {
            "reasoning_tokens": 5
        }
    },
    "request_id": "..."
}

// ... 中间若干包 ...

// Packet M (Content Phase)
data: {
    "output": {
        "choices": [
            {
                "message": {
                    "content": "黎曼猜想",
                    "reasoning_content": "",
                    "role": "assistant"
                },
                "finish_reason": "null"
            }
        ]
    },
    "usage": { // 注意：Usage 必须随包返回累计值
        "total_tokens": 120,
        "input_tokens": 50,
        "output_tokens": 70
    },
    "request_id": "..."
}

```

#### 3.2.1 计量字段 (Usage Metrics)

流式返回的每一包 `usage` 必须包含以下字段的**全量累计值**（Accumulated Value）：

* `input_tokens`: 输入上下文长度（包含搜索注入的内容）。
* `output_tokens`: 模型已生成的总长度。
* `total_tokens`: 。

**推论**：若发生网络中断或用户 Cancel，百炼网关将依据收到的**最后一个有效 SSE 包**中的 `usage` 字段进行计费结算。

---

## 4. 错误码映射 (Error Code Mapping)

Partner 网关必须建立内部错误与 DashScope 标准错误码的**满射（Surjection）**关系。所有的内部异常都必须坍缩为以下标准错误码之一。

| HTTP Status | DashScope Code | 触发场景描述 |
| --- | --- | --- |
| 400 | `InvalidParameter` | 参数格式错误，如 `temperature > 2` 或 `thinking_budget` 非法。 |
| 400 | `DataInspectionFailed` | 触发 DeepSeek 内容安全拦截（输入/输出）。 |
| 401 | `InvalidApiKey` | API Key 无效或过期。 |
| 429 | `Throttling.RateQuota` | 触发 QPS/RPM 限流。 |
| 429 | `Throttling.AllocationQuota` | 触发 Token 消耗速率限流。 |
| 404 | `ModelNotFound` | 请求的模型名称不存在（如拼写错误）。 |
| 500 | `InternalError` | 下游模型推理服务不可用或超时。 |
| 500 | `InternalError.Algo` | 推理过程中的计算错误。 |

**响应体格式**：

```json
{
    "code": "InvalidParameter",
    "message": "The thinking_budget parameter must be a positive integer.",
    "request_id": "..."
}

```

---

## 5. 边界条件与特殊逻辑 (Edge Cases & Implementation Notes)

1. **Thinking 模式互斥性**：
* 当 `enable_thinking = true` 时，Partner 必须校验 `incremental_output` 是否为 `true`。如果客户端未传或传 `false`，建议报错或强制置为 `true`（依据 Postel's Law，建议强制置为 `true` 并返回）。
* **不支持续写**：Thinking 模式下，通常不支持 prefix 续写，需在文档中明确。


2. **搜索（Search）与 Token 膨胀**：
* 若 `enable_search = true`，模型可能会注入搜索结果到 Prompt。此时 `usage.input_tokens` 会显著大于用户原始输入的长度。Partner 必须在 `usage` 中如实反映这一变化，**计费以 Partner 返回的 input_tokens 为准**。


3. **SSE 格式严格性**：
* SSE 必须以 `data: ` 开头，以 `\n\n` 结尾。
* 必须发送 `[DONE]` 标记或通过 `finish_reason: stop` 显式结束流。


4. **超时机制**：
* 百炼网关对首包（TTFT）和包间间隔（Inter-token latency）有严格要求。建议 Partner 侧设置合理的 Keep-Alive 和超时断开逻辑。



---

## 6. 附录：接口校验清单 (Checklist)

* [ ] **协议对齐**：输入支持 `input.messages`，输出支持 `output.choices[].message`。
* [ ] **Usage 透出**：SSE 流式每一包均包含 `usage` 对象，且数值准确。
* [ ] **Thinking 支持**：正确解析 `enable_thinking` 并通过 `reasoning_content` 返回思考内容。
* [ ] **错误码映射**：所有 4xx/5xx 错误均已映射到 DashScope 定义的 Code 空间。
* [ ] **计费凭证**：确认 `usage` 字段包含了 Search 带来的 Token 增量。
