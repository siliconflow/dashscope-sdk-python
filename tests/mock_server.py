import httpx
import os
import json
import time
import uuid
import logging
import threading
import multiprocessing
from typing import List, Optional, Dict, Any, Union, AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices, ConfigDict
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError

# --- [System Configuration] ---

logging.basicConfig(
    level=logging.DEBUG,  # Switched to INFO for production noise reduction
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(process)d | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DeepSeekProxy")

# Upstream Base URL
SILICON_FLOW_BASE_URL = os.getenv(
    "SILICON_FLOW_BASE_URL", "https://api.siliconflow.cn/v1"
)

# MOCK/TEST ONLY: This key is never used in the production generation path
_MOCK_ENV_API_KEY = os.getenv("SILICON_FLOW_API_KEY")

MODEL_MAPPING = {
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "default": "deepseek-ai/DeepSeek-V3",
}
DUMMY_KEY = "dummy-key"

# --- [Shared State] ---


class ServerState:
    _instance = None

    def __init__(self):
        self.request_queue: Optional[multiprocessing.Queue] = None
        self.response_queue: Optional[multiprocessing.Queue] = None
        self.active_requests: int = 0
        self.lock = threading.Lock()
        self.is_mock_mode = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ServerState()
        return cls._instance

    def set_queues(self, req_q, res_q):
        self.request_queue = req_q
        self.response_queue = res_q
        self.is_mock_mode = True
        logger.warning("!!! Server running in MOCK MODE via Queue Injection !!!")

    def increment_request(self):
        with self.lock:
            self.active_requests += 1

    def decrement_request(self):
        with self.lock:
            self.active_requests -= 1

    @property
    def snapshot(self):
        with self.lock:
            return {
                "active_requests": self.active_requests,
                "is_mock_mode": self.is_mock_mode,
            }


SERVER_STATE = ServerState.get_instance()

# --- [Pydantic Models] ---


class Message(BaseModel):
    role: str
    content: Optional[str] = ""  # tool_calls 时 content 可能为空
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

    model_config = ConfigDict(extra="allow")


class InputData(BaseModel):
    messages: Optional[List[Message]] = None
    prompt: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None


class Parameters(BaseModel):
    result_format: str = "message"
    incremental_output: bool = False
    n: Optional[int] = 1
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = None
    seed: Optional[int] = 1234
    max_tokens: Optional[int] = Field(
        None, validation_alias=AliasChoices("max_tokens", "max_length")
    )
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    repetition_penalty: Optional[float] = 1.0

    # OpenAI原生格式
    stop: Optional[Union[str, List[str]]] = None
    # DashScope兼容格式
    stop_words: Optional[List[Dict[str, Any]]] = None
    enable_thinking: bool = False
    thinking_budget: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # === response_format 字段] ===
    response_format: Optional[Dict[str, Any]] = None

    # 显式开启从属性名读取
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class GenerationRequest(BaseModel):
    model: str
    input: InputData
    parameters: Optional[Parameters] = Field(default_factory=Parameters)


# --- [DeepSeek Proxy Logic] ---


class DeepSeekProxy:
    def __init__(self, api_key: str, extra_headers: Optional[Dict[str, str]] = None):
        # We instantiate a new client per request to ensure isolation of user credentials
        if extra_headers is None:
            extra_headers = {}
        if "x-api-key" in extra_headers and extra_headers["x-api-key"]:
            api_key = extra_headers["x-api-key"]

        kv = {"api_key": api_key} if api_key != DUMMY_KEY else {}
        HOUR = 60 * 60.0
        self.client = AsyncOpenAI(
            base_url=SILICON_FLOW_BASE_URL,
            timeout=httpx.Timeout(
                connect=10.0, read=(2 * HOUR), write=600.0, pool=10.0
            ),
            default_headers=extra_headers,  # 透传 Header
            **kv,
        )

    def _get_mapped_model(self, request_model: str) -> str:
        return MODEL_MAPPING.get(request_model, MODEL_MAPPING["default"])

    def _convert_input_to_messages(self, input_data: InputData) -> List[Dict[str, str]]:
        if input_data.messages:
            return [m.model_dump() for m in input_data.messages]

        messages = []
        if input_data.history:
            for item in input_data.history:
                if "user" in item:
                    messages.append({"role": "user", "content": item["user"]})
                if "bot" in item:
                    messages.append({"role": "assistant", "content": item["bot"]})

        if input_data.prompt:
            messages.append({"role": "user", "content": input_data.prompt})
        return messages

    async def generate(self, req_data: GenerationRequest, initial_request_id: str):
        # 0. Pre-computation Validations
        params = req_data.parameters

        # --- Validate 'n' parameter ---
        if params.n is not None:
            if not (1 <= params.n <= 4):
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: Range of n should be [1, 4]",
                    },
                )

        # === [thinking_budget 校验] ===
        if params.thinking_budget is not None:
            # 必须大于 0 (具体限制取决于上游，但负数肯定是无效的)
            if params.thinking_budget <= 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        # 保持与代码库中其他错误一致的格式
                        "message": "<400> InternalError.Algo.InvalidParameter: thinking_budget should be greater than 0",
                    },
                )
        # ===================================

        # === [新增 response_format 校验逻辑] ===
        if params.response_format:
            rf_type = params.response_format.get("type")

            # 校验 type 值是否合法
            # 允许的值通常为 json_object 或 text (根据测试用例报错信息)
            if rf_type and rf_type not in ["json_object", "text"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": f"<400> InternalError.Algo.InvalidParameter: 'response_format.type' Invalid value: '{rf_type}'. Supported values are: 'json_object' and 'text'.",
                    },
                )

            # 校验 json_object 下不能包含 json_schema
            if rf_type == "json_object" and "json_schema" in params.response_format:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: Unknown parameter: 'response_format.json_schema'. 'response_format.json_schema' cannot be provided when 'response_format.type' is 'json_object'.",
                    },
                )
        # -----------------------------------------------

        # 0. 提前拦截无效的 Tool Call 链 (Strict Validation)
        # 必须在 _convert_input_to_messages 之前执行，且依赖 req_data.input.messages 的原始结构
        if req_data.input.messages:
            msgs = req_data.input.messages
            for idx, msg in enumerate(msgs):
                # 检查是否是发起调用的 assistant 消息
                # 注意：这里依赖 Pydantic 模型中已定义 tool_calls 或者是 extra="allow"
                has_tool_calls = getattr(msg, "tool_calls", None)

                if msg.role == "assistant" and has_tool_calls:
                    # 检查下一条消息
                    next_idx = idx + 1
                    if next_idx < len(msgs):
                        next_msg = msgs[next_idx]
                        # 规则：Assistant call 之后必须紧接 tool 消息
                        if next_msg.role != "tool":
                            logger.warning(
                                f"Interceptor caught invalid tool chain at index {next_idx}"
                            )
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "code": "InvalidParameter",
                                    # 精确匹配错误格式
                                    "message": f'<400> InternalError.Algo.InvalidParameter: An assistant message with "tool_calls" must be followed by tool messages responding to each "tool_call_id". The following tool_call_ids did not have response messages: message[{next_idx}].role',
                                },
                            )

                # --- Check if Tool message is preceded by Assistant (NEW LOGIC) ---
                if msg.role == "tool":
                    is_orphan = False
                    if idx == 0:
                        is_orphan = True
                    else:
                        prev_msg = msgs[idx - 1]
                        # Must be assistant AND have tool_calls
                        if prev_msg.role != "assistant" or not getattr(
                            prev_msg, "tool_calls", None
                        ):
                            is_orphan = True

                    if is_orphan:
                        return JSONResponse(
                            status_code=400,
                            content={
                                "code": "InvalidParameter",
                                "message": '<400> InternalError.Algo.InvalidParameter: messages with role "tool" must be a response to a preceeding message with "tool_calls".',
                            },
                        )
                # -----------------------------------------------------

        # Validation: Tools require message format
        if params.tools and params.result_format != "message":
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": "When 'tools' are provided, 'result_format' must be 'message'.",
                },
            )

        # Validation: R1 + Tools constraint
        is_r1 = "deepseek-r1" in req_data.model or params.enable_thinking
        if is_r1 and params.tool_choice and isinstance(params.tool_choice, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": "DeepSeek R1 does not support specific tool_choice binding (dict) while thinking logic is active.",
                },
            )

        target_model = self._get_mapped_model(req_data.model)
        messages = self._convert_input_to_messages(req_data.input)

        openai_params = {
            "model": target_model,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": params.incremental_output or params.enable_thinking,
        }

        # === [将合法的 response_format 加入请求参数] ===
        if params.response_format:
            openai_params["response_format"] = params.response_format
        # ----------------------------------------------------

        if params.frequency_penalty is not None:
            openai_params["frequency_penalty"] = params.frequency_penalty

        if params.presence_penalty is not None:
            openai_params["presence_penalty"] = params.presence_penalty

        extra_body = {}
        # 不是 OpenAI 标准参数，必须放入 extra_body
        # === [校验与修正逻辑] ===

        # 1. 校验 repetition_penalty
        # DashScope 要求 > 0.0，否则报错 InvalidParameter
        if params.repetition_penalty is not None:
            if params.repetition_penalty <= 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: Repetition_penalty should be greater than 0.0",
                    },
                )
            extra_body["repetition_penalty"] = params.repetition_penalty

        # 2. 校验 top_k
        if params.top_k is not None:
            # 2.1 负数校验
            # 此时 Pydantic 已经确保它是 int，这里检查数值
            if params.top_k < 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: Parameter top_k be greater than or equal to 0",
                    },
                )

            # 2.2 上限截断
            # SiliconFlow 限制 top_k 为 [1, 100]。
            # 如果用户传 0，通常对应 disable (即 -1) 或由模型决定，这里映射为 -1 比较稳妥
            # 如果用户传 > 100 (如 1025)，必须截断为 100，否则上游报错
            if params.top_k == 0:
                extra_body["top_k"] = -1  # Disable
            elif params.top_k > 100:
                extra_body["top_k"] = 100  # Clamp to max supported by upstream
            else:
                extra_body["top_k"] = params.top_k

        # === [将 thinking_budget 加入 extra_body] ===
        if params.enable_thinking and params.thinking_budget is not None:
            # 确保只有在开启思考时才透传 budget，且前面已经校验过 >0
            # 如果上游明确支持 thinking_budget 字段：
            extra_body["thinking_budget"] = params.thinking_budget
        # ===============================================

        if extra_body:
            openai_params["extra_body"] = extra_body

        if params.tools:
            openai_params["tools"] = params.tools
            if params.tool_choice:
                if isinstance(params.tool_choice, str) and params.tool_choice == "null":
                    return JSONResponse(
                        status_code=400,
                        content={
                            "code": "InvalidParameter",
                            "message": "<400> InternalError.Algo.InvalidParameter: Invalid tool_choice. It must be 'none', 'auto', or a valid object, not the string 'null'.",
                        },
                    )
                openai_params["tool_choice"] = params.tool_choice

        if params.max_tokens is not None:
            openai_params["max_tokens"] = params.max_tokens
            logger.debug(
                f"[Request] Truncation enabled: max_tokens={params.max_tokens}"
            )
        else:
            logger.debug(
                "[Request] No max_tokens found in parameters, model will generate full response"
            )

        if params.seed:
            openai_params["seed"] = params.seed

        # === 处理 Stop Words 兼容性 ===
        # 优先使用 OpenAI 原生 stop 参数
        if params.stop:
            openai_params["stop"] = params.stop
        # 如果没有 stop 但有 stop_words (DashScope 格式)，则进行转换
        elif params.stop_words:
            # 提取所有 mode="exclude" (默认) 的 stop_str
            # 注意：DashScope 的 stop_words 是 list[dict] 结构
            stop_list = []
            for sw in params.stop_words:
                # 仅处理 exclude 模式或未指定模式的词
                if sw.get("mode", "exclude") == "exclude" and "stop_str" in sw:
                    stop_list.append(sw["stop_str"])

            if stop_list:
                openai_params["stop"] = stop_list

        # --- 生成 Curl 命令 (过滤掉非法 Header) ---
        # 1. 基础 Header
        curl_headers = [
            f"-H 'Authorization: Bearer $SILICONFLOW_API_KEY'",
            "-H 'Content-Type: application/json'",
        ]

        # 2. 补充透传的 Header (过滤掉 Omit 对象和不需要的字段)
        # default_headers 包含了初始化时传入的 extra_headers
        skip_keys = {"authorization", "content-type", "content-length", "host"}
        for k, v in self.client.default_headers.items():
            # 过滤掉 OpenAI 内部的 Omit 对象 和 系统自动生成的头
            if k.lower() not in skip_keys and not str(v).startswith("<openai."):
                curl_headers.append(f"-H '{k}: {v}'")

        # 3. 组装命令
        curl_cmd = (
            f"curl -X POST {SILICON_FLOW_BASE_URL}/chat/completions \\\n  "
            + " \\\n  ".join(curl_headers)
            + f" \\\n  -d '{json.dumps(openai_params, ensure_ascii=False)}'"
        )

        print(f"\n--- [Generated Curl] ---\n{curl_cmd}\n------------------------\n")
        # ----------------------------------------------------

        try:
            if openai_params["stream"]:
                raw_resp = await self.client.chat.completions.with_raw_response.create(
                    **openai_params
                )
                trace_id = raw_resp.headers.get(
                    "X-SiliconCloud-Trace-Id", initial_request_id
                )

                return StreamingResponse(
                    self._stream_generator(raw_resp.parse(), trace_id),
                    media_type="text/event-stream",
                    headers={"X-SiliconCloud-Trace-Id": trace_id},
                )
            else:
                raw_resp = await self.client.chat.completions.with_raw_response.create(
                    **openai_params
                )
                trace_id = raw_resp.headers.get(
                    "X-SiliconCloud-Trace-Id", initial_request_id
                )
                return self._format_unary_response(raw_resp.parse(), trace_id)

        except APIError as e:
            logger.error(
                f"[request id: {initial_request_id}] Upstream API Error: {str(e)}"
            )
            error_code = "InternalError"
            if isinstance(e, RateLimitError):
                error_code = "Throttling.RateQuota"
            elif isinstance(e, AuthenticationError):
                error_code = "InvalidApiKey"

            return JSONResponse(
                status_code=e.status_code or 500,
                content={
                    "code": error_code,
                    "message": str(e),
                    "request_id": initial_request_id,
                },
            )

    async def _stream_generator(
        self, stream, request_id: str
    ) -> AsyncGenerator[str, None]:
        accumulated_usage = {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "output_tokens_details": {"text_tokens": 0, "reasoning_tokens": 0},
        }
        finish_reason = "null"

        full_text = ""
        full_reasoning = ""

        async for chunk in stream:
            if chunk.usage:
                accumulated_usage["total_tokens"] = chunk.usage.total_tokens
                accumulated_usage["input_tokens"] = chunk.usage.prompt_tokens
                accumulated_usage["output_tokens"] = chunk.usage.completion_tokens
                details = getattr(chunk.usage, "completion_tokens_details", None)
                if details:
                    accumulated_usage["output_tokens_details"]["reasoning_tokens"] = (
                        getattr(details, "reasoning_tokens", 0)
                    )
                    accumulated_usage["output_tokens_details"]["text_tokens"] = (
                        accumulated_usage["output_tokens"]
                        - accumulated_usage["output_tokens_details"]["reasoning_tokens"]
                    )

            delta = chunk.choices[0].delta if chunk.choices else None
            delta_content = delta.content if delta and delta.content else ""
            delta_reasoning = (
                (getattr(delta, "reasoning_content", "") or "") if delta else ""
            )

            # ✅ 累积完整内容
            if delta_content:
                full_text += delta_content
            if delta_reasoning:
                full_reasoning += delta_reasoning

            tool_calls = None
            if delta and delta.tool_calls:
                tool_calls = [tc.model_dump() for tc in delta.tool_calls]

            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            # ✅ 关键：stop 包输出“完整累积内容”，避免最后一包是空导致聚合为空
            if finish_reason != "null":
                content_to_send = full_text
                reasoning_to_send = full_reasoning
            else:
                content_to_send = delta_content
                reasoning_to_send = delta_reasoning

            message_body = {
                "role": "assistant",
                "content": content_to_send,
                "reasoning_content": reasoning_to_send,
            }
            if tool_calls:
                message_body["tool_calls"] = tool_calls

            response_body = {
                "output": {
                    "choices": [
                        {"message": message_body, "finish_reason": finish_reason}
                    ]
                },
                "usage": accumulated_usage,
                "request_id": request_id,
            }
            yield f"data: {json.dumps(response_body, ensure_ascii=False)}\n\n"

            if finish_reason != "null":
                break

    def _format_unary_response(self, completion, request_id: str):
        choice = completion.choices[0]
        msg = choice.message
        usage_data = {
            "total_tokens": completion.usage.total_tokens,
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
            "output_tokens_details": {"text_tokens": 0, "reasoning_tokens": 0},
        }
        details = getattr(completion.usage, "completion_tokens_details", None)
        if details:
            usage_data["output_tokens_details"]["reasoning_tokens"] = getattr(
                details, "reasoning_tokens", 0
            )
            usage_data["output_tokens_details"]["text_tokens"] = (
                usage_data["output_tokens"]
                - usage_data["output_tokens_details"]["reasoning_tokens"]
            )

        message_body = {
            "role": msg.role,
            "content": msg.content,
            "reasoning_content": getattr(msg, "reasoning_content", ""),
        }
        if msg.tool_calls:
            message_body["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]

        response_body = {
            "output": {
                "choices": [
                    {"message": message_body, "finish_reason": choice.finish_reason}
                ]
            },
            "usage": usage_data,
            "request_id": request_id,
        }

        return JSONResponse(
            content=response_body, headers={"X-SiliconCloud-Trace-Id": request_id}
        )


# --- [FastAPI App & Lifecycle] ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = threading.Event()

    def epoch_clock():
        while not stop_event.is_set():
            time.sleep(5)
            state = SERVER_STATE.snapshot
            if state["active_requests"] > 0 or state["is_mock_mode"]:
                logger.info(
                    f"[Monitor] Active: {state['active_requests']} | "
                    f"Mode: {'MOCK' if state['is_mock_mode'] else 'PRODUCTION'}"
                )

    monitor_thread = threading.Thread(target=epoch_clock, daemon=True)
    monitor_thread.start()
    yield
    stop_event.set()


def _prepare_proxy_and_headers(
    request: Request, authorization: Optional[str]
) -> tuple[DeepSeekProxy, str]:
    """Helper to extract API key, filter headers, and instantiate the proxy."""
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    api_key = DUMMY_KEY

    if SERVER_STATE.is_mock_mode:
        api_key = _MOCK_ENV_API_KEY or "mock-key"
    elif authorization:
        if not authorization.startswith("Bearer "):
            logger.warning(
                f"Rejected request {request_id}: Invalid Authorization Format"
            )
            raise HTTPException(
                status_code=401,
                detail="Invalid Authorization header format. Expected 'Bearer <token>'",
            )
        api_key = authorization.replace("Bearer ", "")

    logger.debug(f"using API Key: {api_key[:8]}... for request {request_id}")

    unsafe_headers = {
        "host",
        "content-length",
        "content-type",
        "authorization",
        "connection",
        "upgrade",
        "accept-encoding",
        "transfer-encoding",
    }
    forward_headers = {
        k: v for k, v in request.headers.items() if k.lower() not in unsafe_headers
    }

    proxy = DeepSeekProxy(api_key=api_key, extra_headers=forward_headers)
    return proxy, request_id


def create_app() -> FastAPI:
    app = FastAPI(title="DeepSeek-DashScope Proxy", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        # 获取第一个错误详情
        err = exc.errors()[0]
        error_msg = err.get("msg", "Invalid parameter")
        loc = err.get("loc", [])
        param_name = loc[-1] if loc else "unknown"

        # 当 content 字段接收到非字符串类型（如 list）时，Pydantic 会报 "valid string" 错误
        # 此时需返回 "unsupported input format" 以匹配测试用例
        if param_name == "content":
            # 检查是否是因为传了 List 导致的字符串类型错误
            # Pydantic v2 的错误消息通常包含 "Input should be a valid string"
            if "valid string" in error_msg or "str" in error_msg:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: An unknown error occurred due to an unsupported input format.",
                    },
                )

        logger.error(f"Validation Error: {exc.errors()}")

        # 默认错误格式
        return JSONResponse(
            status_code=400,
            content={
                "code": "InvalidParameter",
                "message": f"<400> InternalError.Algo.InvalidParameter: Parameter {param_name} check failed: {error_msg}",
            },
        )

    @app.middleware("http")
    async def request_tracker(request: Request, call_next):
        SERVER_STATE.increment_request()
        start_time = time.time()
        try:
            response = await call_next(request)
            return response
        finally:
            SERVER_STATE.decrement_request()
            duration = (time.time() - start_time) * 1000
            # Reduced log noise for healthy requests, kept for errors or slow ones if needed
            if duration > 1000:
                logger.warning(
                    f"{request.method} {request.url.path} - SLOW {duration:.2f}ms"
                )

    @app.get("/health_check")
    async def health_check():
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "DeepSeek-Proxy",
                "mode": "mock" if SERVER_STATE.is_mock_mode else "production",
            },
        )

    @app.post("/api/v1/services/aigc/text-generation/generation")
    async def generation(
        request: Request,
        body: GenerationRequest = None,
        authorization: Optional[str] = Header(None),
    ):
        proxy, request_id = _prepare_proxy_and_headers(request, authorization)

        # Parse Body if not injected
        if not body:
            try:
                raw_json = await request.json()
                body = GenerationRequest(**raw_json)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        # --- [Auto-enable stream mode based on header] ---
        accept_header = request.headers.get("accept", "")
        if "text/event-stream" in accept_header and body.parameters:
            logger.info("SSE client detected, forcing incremental_output=True")
            body.parameters.incremental_output = True

        # --- [Mock Handling] ---
        if SERVER_STATE.is_mock_mode:
            if body:
                # Shadow Traffic Logic (Optional validation against upstream)
                try:
                    # We only perform shadow traffic if a key is actually available
                    if _MOCK_ENV_API_KEY:
                        shadow_proxy = DeepSeekProxy(api_key=_MOCK_ENV_API_KEY)
                        # Fire and forget (or await if validation is strict)
                        # await shadow_proxy.generate(body, f"shadow-{request_id}")
                except Exception:
                    pass  # Swallow shadow errors

            try:
                raw_body = await request.json()
                SERVER_STATE.request_queue.put(raw_body)
                response_data = SERVER_STATE.response_queue.get(timeout=10)
                response_json = (
                    json.loads(response_data)
                    if isinstance(response_data, str)
                    else response_data
                )
                status_code = response_json.pop("status_code", 200)
                return JSONResponse(content=response_json, status_code=status_code)
            except Exception as e:
                return JSONResponse(
                    status_code=500, content={"code": "MockError", "message": str(e)}
                )

        # --- [Production Handling] ---
        # Forward request to upstream
        return await proxy.generate(body, request_id)

    @app.post("/siliconflow/models/{model_path:path}")
    async def dynamic_path_generation(
        model_path: str, request: Request, authorization: Optional[str] = Header(None)
    ):
        proxy, request_id = _prepare_proxy_and_headers(request, authorization)

        # 2. Parse, Inject Model, and Validate
        try:
            payload = await request.json()
            payload["model"] = model_path  # Force set model from URL
            body = GenerationRequest(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Request: {e}")

        # 3. Handle SSE
        if "text/event-stream" in request.headers.get("accept", "") and body.parameters:
            body.parameters.incremental_output = True

        # 4. Generate
        return await proxy.generate(body, request_id)

    @app.api_route("/{path_name:path}", methods=["GET", "POST", "DELETE", "PUT"])
    async def catch_all(path_name: str, request: Request):
        # Catch-all only valid in Mock Mode
        if SERVER_STATE.is_mock_mode:
            try:
                body = None
                if request.method in ["POST", "PUT"]:
                    try:
                        body = await request.json()
                    except:
                        pass
                req_record = {
                    "path": f"/{path_name}",
                    "method": request.method,
                    "headers": dict(request.headers),
                    "body": body,
                }
                SERVER_STATE.request_queue.put(req_record)
                response_data = SERVER_STATE.response_queue.get(timeout=5)
                response_json = (
                    json.loads(response_data)
                    if isinstance(response_data, str)
                    else response_data
                )
                status_code = response_json.pop("status_code", 200)
                return JSONResponse(content=response_json, status_code=status_code)
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"error": "Mock Catch-All Error", "detail": str(e)},
                )

        return JSONResponse(
            status_code=404,
            content={"code": "NotFound", "message": "Proxy endpoint not implemented"},
        )

    return app


def run_server_process(req_q, res_q, host="0.0.0.0", port=8000):
    if req_q and res_q:
        SERVER_STATE.set_queues(req_q, res_q)
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


class MockServer:
    def __init__(self) -> None:
        self.requests = multiprocessing.Queue()
        self.responses = multiprocessing.Queue()
        self.proc = None


def create_mock_server(*args, **kwargs):
    mock_server = MockServer()
    proc = multiprocessing.Process(
        target=run_server_process,
        args=(mock_server.requests, mock_server.responses, "0.0.0.0", 8089),
    )
    proc.start()
    mock_server.proc = proc
    time.sleep(1.5)
    logger.info("Mock Server started on port 8089")
    if args and hasattr(args[0], "addfinalizer"):

        def stop_server():
            if proc.is_alive():
                proc.terminate()
                proc.join()

        args[0].addfinalizer(stop_server)
    return mock_server


def run_server(host="0.0.0.0", port=8000):
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
