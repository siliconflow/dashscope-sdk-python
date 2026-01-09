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

MAX_NUM_MSG_CURL_DUMP = 5  # Controls the max messages shown in curl logs

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
    content: Optional[str] = ""  # content can be empty when using tool_calls
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

    # OpenAI native format
    stop: Optional[Union[str, List[str]]] = None
    # DashScope compatible format
    stop_words: Optional[List[Dict[str, Any]]] = None
    enable_thinking: bool = False
    thinking_budget: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    # === [response_format field] ===
    response_format: Optional[Dict[str, Any]] = None

    # Explicitly enable reading from attribute name
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
            default_headers=extra_headers,  # Pass through headers
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

        # === [thinking_budget validation] ===
        if params.thinking_budget is not None:
            if params.thinking_budget <= 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: thinking_budget should be greater than 0",
                    },
                )
        # ===================================

        # === [response_format validation] ===
        if params.response_format:
            rf_type = params.response_format.get("type")
            if rf_type and rf_type not in ["json_object", "text"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": f"<400> InternalError.Algo.InvalidParameter: 'response_format.type' Invalid value: '{rf_type}'. Supported values are: 'json_object' and 'text'.",
                    },
                )

            if rf_type == "json_object" and "json_schema" in params.response_format:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: Unknown parameter: 'response_format.json_schema'. 'response_format.json_schema' cannot be provided when 'response_format.type' is 'json_object'.",
                    },
                )
        # -----------------------------------------------

        # 0. Preemptively intercept invalid Tool Call chains
        if req_data.input.messages:
            msgs = req_data.input.messages
            for idx, msg in enumerate(msgs):
                has_tool_calls = getattr(msg, "tool_calls", None)

                if msg.role == "assistant" and has_tool_calls:
                    next_idx = idx + 1
                    if next_idx < len(msgs):
                        next_msg = msgs[next_idx]
                        # Rule: Assistant call must be immediately followed by tool messages
                        if next_msg.role != "tool":
                            logger.warning(
                                f"Interceptor caught invalid tool chain at index {next_idx}"
                            )
                            return JSONResponse(
                                status_code=400,
                                content={
                                    "code": "InvalidParameter",
                                    "message": f'<400> InternalError.Algo.InvalidParameter: An assistant message with "tool_calls" must be followed by tool messages responding to each "tool_call_id". The following tool_call_ids did not have response messages: message[{next_idx}].role',
                                },
                            )

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

        if params.response_format:
            openai_params["response_format"] = params.response_format

        if params.frequency_penalty is not None:
            openai_params["frequency_penalty"] = params.frequency_penalty

        if params.presence_penalty is not None:
            openai_params["presence_penalty"] = params.presence_penalty

        extra_body = {}
        # Not an OpenAI standard parameter, must be put into extra_body
        # === [Validation and Correction Logic] ===

        # 1. Validate repetition_penalty
        # DashScope requires > 0.0, otherwise InvalidParameter error
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

        # 2. Validate top_k
        if params.top_k is not None:
            # 2.1 Negative number validation
            # Pydantic has ensured it is an int, checking value here
            if params.top_k < 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: Parameter top_k be greater than or equal to 0",
                    },
                )

            # 2.2 Upper limit truncation
            # SiliconFlow limits top_k to [1, 100].
            # If user passes 0, usually corresponds to disable (i.e. -1) or model decision; mapping to -1 here is safer.
            # If user passes > 100 (e.g., 1025), must truncate to 100, otherwise upstream errors.
            if params.top_k == 0:
                extra_body["top_k"] = -1
            elif params.top_k > 100:
                extra_body["top_k"] = 100
            else:
                extra_body["top_k"] = params.top_k

        # === [Add thinking_budget to extra_body] ===
        if params.enable_thinking and params.thinking_budget is not None:
            # Ensure budget is passed only when thinking is enabled, and validated >0 previously
            # If upstream explicitly supports thinking_budget field:
            extra_body["thinking_budget"] = params.thinking_budget

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

        # === Handle Stop Words Compatibility ===
        # Prioritize OpenAI native stop parameter
        if params.stop:
            openai_params["stop"] = params.stop
        # If no stop but stop_words exists (DashScope format), convert it
        elif params.stop_words:
            # Extract all stop_str with mode='exclude' (default)
            # Note: DashScope's stop_words is a list[dict] structure
            stop_list = []
            for sw in params.stop_words:
                # Only handle exclude mode or unspecified mode words
                if sw.get("mode", "exclude") == "exclude" and "stop_str" in sw:
                    stop_list.append(sw["stop_str"])

            if stop_list:
                openai_params["stop"] = stop_list

        # --- Generate Curl Command (Filter illegal Headers & Truncate Log) ---
        # 1. Base Header
        curl_headers = [
            '-H "Authorization: Bearer ${SILICONFLOW_API_KEY}"',
            "-H 'Content-Type: application/json'",
        ]

        # 2. Supplement pass-through Headers
        skip_keys = {"authorization", "content-type", "content-length", "host"}
        for k, v in self.client.default_headers.items():
            if k.lower() not in skip_keys and not str(v).startswith("<openai."):
                curl_headers.append(f"-H '{k}: {v}'")

        # 3. Prepare Logging Payload (Truncate if message too long)
        log_payload = openai_params.copy()
        msg_list = log_payload.get("messages", [])
        if len(msg_list) > MAX_NUM_MSG_CURL_DUMP:
            # Truncation display: keep only first N items, add explanation
            truncated_msgs = msg_list[:MAX_NUM_MSG_CURL_DUMP]
            truncated_msgs.append(
                {
                    "_log_truncation": f"... {len(msg_list) - MAX_NUM_MSG_CURL_DUMP} messages omitted for brevity ..."
                }
            )
            log_payload["messages"] = truncated_msgs

        # 4. Assemble command
        curl_cmd = (
            f"curl -X POST {SILICON_FLOW_BASE_URL}/chat/completions \\\n  "
            + " \\\n  ".join(curl_headers)
            + f" \\\n  -d '{json.dumps(log_payload, ensure_ascii=False)}'"
        )

        logger.debug(f"[Curl Command]\n{curl_cmd}")
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

            # ✅ Accumulate full content
            if delta_content:
                full_text += delta_content
            if delta_reasoning:
                full_reasoning += delta_reasoning

            tool_calls = None
            if delta and delta.tool_calls:
                tool_calls = [tc.model_dump() for tc in delta.tool_calls]

            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            # ✅ Key: stop packet outputs "full accumulated content" to avoid empty aggregation if last packet is empty
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

    # get x-api-key
    x_api_key = request.headers.get("x-api-key")

    if SERVER_STATE.is_mock_mode:
        api_key = _MOCK_ENV_API_KEY or "mock-key"

    # [Modification]: If x-api-key exists, override authorization header
    elif x_api_key:
        api_key = x_api_key
        logger.debug(
            f"Request {request_id}: Using x-api-key for authorization override."
        )

    # Only check Authorization Header if x-api-key is not present
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
        "authorization",  # Authorization has been regenerated by AsyncOpenAI client through api_key parameter, so it is filtered out here
        "connection",
        "upgrade",
        "accept-encoding",
        "transfer-encoding",
    }

    # x-api-key 不在 unsafe_headers 中，因此会被保留在 forward_headers 中发送给上游
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
        # Get first error detail
        err = exc.errors()[0]
        error_msg = err.get("msg", "Invalid parameter")
        loc = err.get("loc", [])
        param_name = loc[-1] if loc else "unknown"

        # When content field receives non-string type (e.g. list), Pydantic reports "valid string" error
        # Return "unsupported input format" to match test cases
        if param_name == "content":
            # Check if string type error caused by passing List
            # Pydantic v2 error messages usually contain "Input should be a valid string"
            if "valid string" in error_msg or "str" in error_msg:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: An unknown error occurred due to an unsupported input format.",
                    },
                )

        logger.error(f"Validation Error: {exc.errors()}")

        # Default error format
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

        # --- [Auto-enable stream mode based on headers] ---
        # Check traditional Accept Header or DashScope specific X-DashScope-SSE Header
        accept_header = request.headers.get("accept", "")
        dashscope_sse = request.headers.get("x-dashscope-sse", "").lower()

        if (
            "text/event-stream" in accept_header or dashscope_sse == "enable"
        ) and body.parameters:
            logger.debug(
                f"SSE detected (Accept: {accept_header}, X-DashScope-SSE: {dashscope_sse}), enabling stream"
            )
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
        # Check traditional Accept Header or DashScope specific X-DashScope-SSE Header
        accept_header = request.headers.get("accept", "")
        dashscope_sse = request.headers.get("x-dashscope-sse", "").lower()

        if (
            "text/event-stream" in accept_header or dashscope_sse == "enable"
        ) and body.parameters:
            logger.debug(
                f"SSE detected (Accept: {accept_header}, X-DashScope-SSE: {dashscope_sse}), enabling stream"
            )
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
