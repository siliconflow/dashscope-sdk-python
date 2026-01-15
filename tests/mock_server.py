import os
import json
import time
import uuid
import logging
import threading
import multiprocessing
from typing import List, Optional, Dict, Any, Union, AsyncGenerator, Tuple
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, AliasChoices, ConfigDict, ValidationError
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(process)d | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DeepSeekProxy")

# --- Constants & Environment Variables ---
SILICON_FLOW_BASE_URL = os.getenv(
    "SILICON_FLOW_BASE_URL", "https://api.siliconflow.cn/v1"
)
_MOCK_ENV_API_KEY = os.getenv("SILICON_FLOW_API_KEY")

MODEL_MAPPING = {
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "default": "deepseek-ai/DeepSeek-V3",
    "pre-siliconflow/deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "pre-siliconflow/deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "pre-siliconflow/deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    "pre-siliconflow/deepseek-r1": "deepseek-ai/DeepSeek-R1",
}

DUMMY_KEY = "dummy-key"
MAX_NUM_MSG_CURL_DUMP = 5


# --- Server State Management ---
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


# --- Pydantic Models ---
class Message(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    partial: Optional[bool] = None

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

    prefix: Optional[str] = None

    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None

    stop: Optional[Union[str, List[Union[str, int]]]] = None

    stop_words: Optional[List[Dict[str, Any]]] = None
    enable_thinking: bool = False
    thinking_budget: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

    response_format: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())


class GenerationRequest(BaseModel):
    model: str
    input: Optional[InputData] = None
    parameters: Optional[Parameters] = Field(default_factory=Parameters)


# --- DeepSeek Proxy Logic ---
class DeepSeekProxy:
    def __init__(self, api_key: str, extra_headers: Optional[Dict[str, str]] = None):
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
            default_headers=extra_headers,
            **kv,
        )

    def _get_mapped_model(self, request_model: str) -> str:
        return MODEL_MAPPING.get(request_model, MODEL_MAPPING["default"])

    def _convert_input_to_messages(
        self, input_data: InputData
    ) -> List[Dict[str, Any]]:
        if input_data.messages:
            return [m.model_dump(exclude_none=True) for m in input_data.messages]
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

    def _find_earliest_stop(
        self, text: str, stop_sequences: List[str]
    ) -> Tuple[int, Optional[str]]:
        """
        Helper to find the earliest occurrence of any stop sequence in text.
        Returns (index, stop_sequence_found). If not found, index is -1.
        """
        min_index = -1
        found_stop = None
        for stop_seq in stop_sequences:
            if not stop_seq:
                continue
            idx = text.find(stop_seq)
            if idx != -1:
                if min_index == -1 or idx < min_index:
                    min_index = idx
                    found_stop = stop_seq
        return min_index, found_stop

    async def generate(
        self,
        req_data: GenerationRequest,
        initial_request_id: str,
        force_stream: bool = False,
        skip_model_exist_check: bool = False,
    ):
        # Model existence check
        if not skip_model_exist_check and req_data.model not in MODEL_MAPPING:
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": "Model not exist.",
                },
            )

        # Input content validation
        has_input = req_data.input is not None
        has_content = False
        if has_input:
            if (
                req_data.input.messages is not None
                and req_data.input.prompt is not None
            ):
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": '<400> InternalError.Algo.InvalidParameter: Only one of the parameters "prompt" and "messages" can be present',
                    },
                )

            has_content = (
                bool(req_data.input.messages)
                or bool(req_data.input.prompt)
                or bool(req_data.input.history)
            )

            if req_data.input.prompt == "" and not req_data.input.messages:
                has_content = False

        if not has_input or not has_content:
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": '<400> InternalError.Algo.InvalidParameter: Either "prompt" or "messages" must exist and cannot both be none',
                },
            )

        params = req_data.parameters

        # Logprobs check
        if params.logprobs:
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": "<400> InternalError.Algo.InvalidParameter: The parameters `logprobs` is not supported.",
                },
            )

        # Max Tokens check
        if params.max_tokens is not None and params.max_tokens < 1:
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": f"<400> InternalError.Algo.InvalidParameter: Range of max_tokens should be [1, 2147483647]",
                },
            )

        # N (Completions) check
        if params.n is not None:
            if not (1 <= params.n <= 4):
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: Range of n should be [1, 4]",
                    },
                )

        # Top P check
        if params.top_p is not None:
            if params.top_p <= 0 or params.top_p > 1.0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": f"<400> InternalError.Algo.InvalidParameter: Range of top_p should be (0.0, 1.0], but got {params.top_p}",
                    },
                )

        # Temperature check
        if params.temperature is not None:
            if params.temperature < 0 or params.temperature > 2:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": f"<400> InternalError.Algo.InvalidParameter: Temperature should be in [0, 2], but got {params.temperature}",
                    },
                )

        # Thinking Budget check
        if params.thinking_budget is not None:
            if params.thinking_budget <= 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: thinking_budget should be greater than 0",
                    },
                )

        # Seed check
        if params.seed is not None:
            if not (0 <= params.seed <= 9223372036854775807):
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: Range of seed should be [0, 9223372036854775807]",
                    },
                )

        # Response Format check
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

        # Tool Calls chain logic validation
        if req_data.input.messages:
            msgs = req_data.input.messages
            for idx, msg in enumerate(msgs):
                has_tool_calls = getattr(msg, "tool_calls", None)

                if msg.role == "assistant" and has_tool_calls:
                    next_idx = idx + 1
                    if next_idx < len(msgs):
                        next_msg = msgs[next_idx]
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

        if params.tools and params.result_format != "message":
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": "When 'tools' are provided, 'result_format' must be 'message'.",
                },
            )

        is_r1 = "deepseek-r1" in req_data.model or params.enable_thinking
        if is_r1 and params.tool_choice and isinstance(params.tool_choice, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": "DeepSeek R1 does not support specific tool_choice binding (dict) while thinking logic is active.",
                },
            )

        if "deepseek-r1" in req_data.model and params.enable_thinking:
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": "Value error, current model does not support parameter `enable_thinking`.",
                },
            )

        # Stop parameter extraction
        proxy_stop_list: List[str] = []
        if params.stop:
            if isinstance(params.stop, str):
                proxy_stop_list.append(params.stop)
            elif isinstance(params.stop, list):
                for s in params.stop:
                    if isinstance(s, str):
                        proxy_stop_list.append(s)

        if params.stop_words:
            for sw in params.stop_words:
                if sw.get("mode", "exclude") == "exclude" and "stop_str" in sw:
                    proxy_stop_list.append(sw["stop_str"])

        # Deduplicate
        proxy_stop_list = list(set(proxy_stop_list))

        # --- Request Parameters Assembly ---
        target_model = self._get_mapped_model(req_data.model)
        messages = self._convert_input_to_messages(req_data.input)

        if params.enable_thinking:
            for msg in messages:
                if msg.get("partial"):
                    return JSONResponse(
                        status_code=400,
                        content={
                            "code": "InvalidParameter",
                            "message": "<400> InternalError.Algo.InvalidParameter: Partial mode is not supported when enable_thinking is true",
                        },
                    )

        should_stream = (
            params.incremental_output or params.enable_thinking or force_stream
        )

        openai_params = {
            "model": target_model,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": should_stream,
            # Note: "stop" is intentionally omitted here, handled by proxy
        }

        if params.response_format:
            openai_params["response_format"] = params.response_format

        if params.frequency_penalty is not None:
            openai_params["frequency_penalty"] = params.frequency_penalty
        if params.presence_penalty is not None:
            openai_params["presence_penalty"] = params.presence_penalty

        extra_body = {}
        if params.prefix is not None:
            extra_body["prefix"] = params.prefix
        if params.repetition_penalty is not None:
            if params.repetition_penalty <= 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: Repetition_penalty should be greater than 0.0",
                    },
                )
            clamped_penalty = min(params.repetition_penalty, 2.0)
            if clamped_penalty != params.repetition_penalty:
                logger.warning(
                    f"Repetition penalty {params.repetition_penalty} exceeds maximum allowed value 2.0, clamping to 2.0"
                )
            extra_body["repetition_penalty"] = clamped_penalty

        if params.top_k is not None:
            if params.top_k < 0:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: Parameter top_k be greater than or equal to 0",
                    },
                )

            if params.top_k == 0:
                extra_body["top_k"] = -1
            elif params.top_k > 100:
                extra_body["top_k"] = 100
            else:
                extra_body["top_k"] = params.top_k

        extra_body["enable_thinking"] = params.enable_thinking

        if params.enable_thinking and params.thinking_budget is not None:
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

        if params.seed:
            openai_params["seed"] = params.seed

        # --- Debug Logging ---
        logger.debug(f"[Stop Sequences] Proxy handling stop for: {proxy_stop_list}")

        curl_headers = [
            '-H "Authorization: Bearer ${SILICONFLOW_API_KEY}"',
            "-H 'Content-Type: application/json'",
        ]
        skip_keys = {"authorization", "content-type", "content-length", "host"}
        for k, v in self.client.default_headers.items():
            if k.lower() not in skip_keys and not str(v).startswith("<openai."):
                curl_headers.append(f"-H '{k}: {v}'")

        log_payload = openai_params.copy()
        msg_list = log_payload.get("messages", [])
        if len(msg_list) > MAX_NUM_MSG_CURL_DUMP:
            truncated_msgs = msg_list[:MAX_NUM_MSG_CURL_DUMP]
            truncated_msgs.append(
                {
                    "_log_truncation": f"... {len(msg_list) - MAX_NUM_MSG_CURL_DUMP} messages omitted for brevity ..."
                }
            )
            log_payload["messages"] = truncated_msgs

        curl_cmd = (
            f"curl -X POST {SILICON_FLOW_BASE_URL}/chat/completions \\\n  "
            + " \\\n  ".join(curl_headers)
            + f" \\\n  -d '{json.dumps(log_payload, ensure_ascii=False)}'"
        )
        logger.debug(f"[Curl Command]\n{curl_cmd}")

        # --- Execution ---
        try:
            is_r1_model = "deepseek-r1" in req_data.model
            if openai_params["stream"]:
                raw_resp = await self.client.chat.completions.with_raw_response.create(
                    **openai_params
                )
                trace_id = raw_resp.headers.get(
                    "X-SiliconCloud-Trace-Id", initial_request_id
                )

                return StreamingResponse(
                    self._stream_generator(
                        raw_resp.parse(),
                        trace_id,
                        is_incremental=params.incremental_output,
                        stop_sequences=proxy_stop_list,
                        is_r1_model=is_r1_model,
                    ),
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
                return self._format_unary_response(
                    raw_resp.parse(),
                    trace_id,
                    stop_sequences=proxy_stop_list,
                    is_r1_model=is_r1_model,
                )

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
        self,
        stream,
        request_id: str,
        is_incremental: bool,
        stop_sequences: List[str],
        is_r1_model: bool = False,
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
        accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}

        # --- Stop Logic Buffer State ---
        content_buffer = ""
        max_stop_len = max([len(s) for s in stop_sequences]) if stop_sequences else 0
        stop_triggered = False

        async for chunk in stream:
            if stop_triggered:
                continue

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

            if is_r1_model:
                accumulated_usage["output_tokens_details"].pop("text_tokens", None)

            delta = chunk.choices[0].delta if chunk.choices else None

            # --- Reasoning Content Handling ---
            delta_reasoning = (
                (getattr(delta, "reasoning_content", "") or "") if delta else ""
            )
            if delta_reasoning:
                full_reasoning += delta_reasoning
                # Output reasoning content regardless of incremental mode
                if is_incremental:
                    yield self._build_stream_response(
                        content="",
                        reasoning_content=delta_reasoning,
                        tool_calls=None,
                        finish_reason="null",
                        usage=accumulated_usage,
                        request_id=request_id,
                    )
                else:
                    # In non-incremental mode, output accumulated reasoning and text
                    yield self._build_stream_response(
                        content=full_text,
                        reasoning_content=full_reasoning,
                        tool_calls=None,
                        finish_reason="null",
                        usage=accumulated_usage,
                        request_id=request_id,
                    )

            # --- Text Content Handling ---
            delta_content = delta.content if delta and delta.content else ""
            content_to_yield = ""

            if delta_content:
                if not stop_sequences:
                    content_to_yield = delta_content
                    full_text += delta_content
                else:
                    content_buffer += delta_content
                    earliest_idx, _ = self._find_earliest_stop(
                        content_buffer, stop_sequences
                    )

                    if earliest_idx != -1:
                        stop_triggered = True
                        finish_reason = "stop"
                        final_chunk = content_buffer[:earliest_idx]
                        content_to_yield = final_chunk
                        full_text += final_chunk
                        content_buffer = ""
                    else:
                        if len(content_buffer) > max_stop_len:
                            safe_chars = len(content_buffer) - max_stop_len
                            chunk_safe = content_buffer[:safe_chars]
                            content_to_yield = chunk_safe
                            full_text += chunk_safe
                            content_buffer = content_buffer[safe_chars:]

            # --- Tool Calls Handling ---
            current_tool_calls_payload = None
            if delta and delta.tool_calls:
                if is_incremental:
                    current_tool_calls_payload = [
                        tc.model_dump() for tc in delta.tool_calls
                    ]
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in accumulated_tool_calls:
                        accumulated_tool_calls[idx] = {
                            "index": idx,
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name or "",
                                "arguments": "",
                            },
                        }
                    else:
                        if tc.id:
                            accumulated_tool_calls[idx]["id"] = tc.id
                        if tc.function.name:
                            accumulated_tool_calls[idx]["function"][
                                "name"
                            ] = tc.function.name
                    if tc.function.arguments:
                        accumulated_tool_calls[idx]["function"][
                            "arguments"
                        ] += tc.function.arguments

            # Check upstream finish reason
            upstream_finish = (
                chunk.choices[0].finish_reason
                if (chunk.choices and chunk.choices[0].finish_reason)
                else "null"
            )
            if upstream_finish != "null":
                finish_reason = upstream_finish

            # --- Yield Content Logic ---

            # Check if there is actual content to push
            has_content_update = (
                content_to_yield
                or current_tool_calls_payload
                or (stop_triggered and finish_reason == "stop")
            )

            if has_content_update:
                if is_incremental:
                    # Incremental mode: push only delta
                    yield self._build_stream_response(
                        content=content_to_yield,
                        reasoning_content="",  # Reasoning handled above
                        tool_calls=current_tool_calls_payload,
                        finish_reason=(finish_reason if stop_triggered else "null"),
                        usage=accumulated_usage,
                        request_id=request_id,
                    )
                else:
                    # Non-incremental mode: push full text
                    # Note: Full tool calls are usually sent at the end in non-incremental mode
                    yield self._build_stream_response(
                        content=full_text,
                        reasoning_content=full_reasoning,
                        tool_calls=None,
                        finish_reason=(finish_reason if stop_triggered else "null"),
                        usage=accumulated_usage,
                        request_id=request_id,
                    )

            if stop_triggered:
                break

        # --- End of Stream Handling ---

        # Flush leftover buffer if not stopped
        if not stop_triggered and content_buffer and stop_sequences:
            full_text += content_buffer
            # Flush remaining buffer based on mode
            if is_incremental:
                yield self._build_stream_response(
                    content=content_buffer,
                    reasoning_content="",
                    tool_calls=None,
                    finish_reason="null",
                    usage=accumulated_usage,
                    request_id=request_id,
                )
            else:
                yield self._build_stream_response(
                    content=full_text,
                    reasoning_content=full_reasoning,
                    tool_calls=None,
                    finish_reason="null",
                    usage=accumulated_usage,
                    request_id=request_id,
                )

        # Final Finish Handling
        if not is_incremental:
            # Final packet for non-incremental mode: includes finish_reason and complete Tool Calls
            if stop_sequences:
                earliest_idx, _ = self._find_earliest_stop(full_text, stop_sequences)
                if earliest_idx != -1:
                    full_text = full_text[:earliest_idx]
                    finish_reason = "stop"

            final_tool_calls = None
            if accumulated_tool_calls:
                final_tool_calls = sorted(
                    accumulated_tool_calls.values(), key=lambda x: x["index"]
                )

            yield self._build_stream_response(
                content=full_text,
                reasoning_content=full_reasoning,
                tool_calls=final_tool_calls,
                finish_reason=finish_reason,
                usage=accumulated_usage,
                request_id=request_id,
            )
        else:
            # Incremental mode: send empty end packet if not triggered by stop
            if not stop_triggered and finish_reason != "null":
                yield self._build_stream_response(
                    content="",
                    reasoning_content="",
                    tool_calls=None,
                    finish_reason=finish_reason,
                    usage=accumulated_usage,
                    request_id=request_id,
                )

    def _build_stream_response(
        self, content, reasoning_content, tool_calls, finish_reason, usage, request_id
    ):
        """Helper to format SSE data line"""
        message_body = {
            "role": "assistant",
            "content": content,
            "reasoning_content": reasoning_content,
        }
        if tool_calls:
            message_body["tool_calls"] = tool_calls

        response_body = {
            "output": {
                "choices": [{"message": message_body, "finish_reason": finish_reason}]
            },
            "usage": usage,
            "request_id": request_id,
        }
        return f"data: {json.dumps(response_body, ensure_ascii=False)}\n\n"

    def _format_unary_response(
        self,
        completion,
        request_id: str,
        stop_sequences: List[str],
        is_r1_model: bool = False,
    ):
        choice = completion.choices[0]
        msg = choice.message

        # --- Stop Logic (Unary) ---
        final_content = msg.content
        finish_reason = choice.finish_reason

        if final_content and stop_sequences:
            earliest_idx, _ = self._find_earliest_stop(final_content, stop_sequences)
            if earliest_idx != -1:
                final_content = final_content[:earliest_idx]
                finish_reason = "stop"
                logger.debug(
                    f"[Stop Logic] Truncated unary response at index {earliest_idx}"
                )

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

        if is_r1_model:
            usage_data["output_tokens_details"].pop("text_tokens", None)

        message_body = {
            "role": msg.role,
            "content": final_content,  # Uses the potentially truncated content
            "reasoning_content": getattr(msg, "reasoning_content", ""),
        }
        if msg.tool_calls:
            message_body["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]

        response_body = {
            "output": {
                "choices": [{"message": message_body, "finish_reason": finish_reason}]
            },
            "usage": usage_data,
            "request_id": request_id,
        }

        return JSONResponse(
            content=response_body, headers={"X-SiliconCloud-Trace-Id": request_id}
        )


# --- FastAPI Application Lifecycle ---
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
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    api_key = DUMMY_KEY

    x_api_key = request.headers.get("x-api-key")

    if SERVER_STATE.is_mock_mode:
        api_key = _MOCK_ENV_API_KEY or "mock-key"

    elif x_api_key:
        api_key = x_api_key
        logger.debug(
            f"Request {request_id}: Using x-api-key for authorization override."
        )

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
    @app.exception_handler(ValidationError)  # Catch errors during manual parsing
    async def validation_exception_handler(request, exc):
        try:
            # Compatible with two ways of retrieving errors
            errors = exc.errors() if hasattr(exc, "errors") else []
        except Exception:
            errors = [{"msg": str(exc), "loc": [], "type": "unknown"}]

        if not errors:
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": "Unknown validation error",
                },
            )

        err = errors[0]
        error_msg = err.get("msg", "Invalid parameter")
        loc = err.get("loc", [])
        param_name = loc[-1] if loc else "unknown"
        path_str = ".".join([str(x) for x in loc if x != "body"])
        err_type = err.get("type")
        input_value = err.get("input")

        # Keep original logic below

        if err_type == "int_parsing":
            if isinstance(input_value, str):
                if param_name in ["max_tokens", "max_length"]:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "code": "InvalidParameter",
                            "message": f"<400> InternalError.Algo.InvalidParameter: Input should be a valid integer, unable to parse string as an integer: {path_str}",
                        },
                    )
                else:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "code": "InvalidParameter",
                            "message": f"<400> InternalError.Algo.InvalidParameter: Input should be a valid integer: {path_str}",
                        },
                    )

        if err_type == "int_from_float":
            if param_name in ["max_tokens", "max_length"]:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": f"<400> InternalError.Algo.InvalidParameter: Input should be a valid integer, got a number with a fractional part: {path_str}",
                    },
                )
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": f"<400> InternalError.Algo.InvalidParameter: Input should be a valid integer: {path_str}",
                    },
                )

        if "stop" in loc:
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": "<400> InternalError.Algo.InvalidParameter: Input should be a valid list: parameters.stop.list[any] & Input should be a valid string: parameters.stop.str",
                },
            )

        if "model" in loc and err_type == "missing":
            return JSONResponse(
                status_code=400,
                content={
                    "code": "BadRequest.EmptyModel",
                    "message": 'Required parameter "model" missing from request.',
                },
            )

        if param_name == "content" or (len(loc) > 1 and loc[-2] == "content"):
            if (
                "valid string" in error_msg
                or "str" in error_msg
                or err_type == "string_type"
                or err_type == "list_type"
                or err_type == "dict_type"
            ):
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "InvalidParameter",
                        "message": "<400> InternalError.Algo.InvalidParameter: An unknown error occurred due to an unsupported input format.",
                    },
                )

        if "response_format" in loc and err_type == "dict_type":
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": f"<400> InternalError.Algo.InvalidParameter: Unknown format of response_format, response_format should be a dict, includes 'type' and an optional key 'json_schema'. The response_format type from user is {type(input_value)}.",
                },
            )

        type_msg_map = {
            "float_parsing": "Input should be a valid number, unable to parse string as a number",
            "bool_parsing": "Input should be a valid boolean, unable to interpret input",
            "string_type": "Input should be a valid string",
        }

        if err_type in type_msg_map:
            return JSONResponse(
                status_code=400,
                content={
                    "code": "InvalidParameter",
                    "message": f"<400> InternalError.Algo.InvalidParameter: {type_msg_map[err_type]}: {path_str}",
                },
            )

        # Fallback logging
        logger.error(f"Validation Error: {errors}")

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

        if not body:
            try:
                raw_json = await request.json()
                body = GenerationRequest(**raw_json)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        accept_header = request.headers.get("accept", "")
        dashscope_sse = request.headers.get("x-dashscope-sse", "").lower()

        force_stream = False
        if (
            "text/event-stream" in accept_header or dashscope_sse == "enable"
        ) and body.parameters:
            logger.debug(
                f"SSE detected (Accept: {accept_header}, X-DashScope-SSE: {dashscope_sse}), enabling stream transport"
            )
            force_stream = True

        if SERVER_STATE.is_mock_mode:
            if body:
                try:
                    if _MOCK_ENV_API_KEY:
                        shadow_proxy = DeepSeekProxy(api_key=_MOCK_ENV_API_KEY)
                except Exception:
                    pass

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

        return await proxy.generate(body, request_id, force_stream=force_stream)

    @app.post("/siliconflow/models/{model_path:path}")
    async def dynamic_path_generation(
        model_path: str, request: Request, authorization: Optional[str] = Header(None)
    ):
        proxy, request_id = _prepare_proxy_and_headers(request, authorization)

        try:
            payload = await request.json()
            payload["model"] = model_path
            body = GenerationRequest(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Request: {e}")

        accept_header = request.headers.get("accept", "")
        dashscope_sse = request.headers.get("x-dashscope-sse", "").lower()

        force_stream = False
        if (
            "text/event-stream" in accept_header or dashscope_sse == "enable"
        ) and body.parameters:
            logger.debug(
                f"SSE detected (Accept: {accept_header}, X-DashScope-SSE: {dashscope_sse}), enabling stream transport"
            )
            force_stream = True

        return await proxy.generate(
            body, request_id, force_stream=force_stream, skip_model_exist_check=True
        )

    @app.api_route("/{path_name:path}", methods=["GET", "POST", "DELETE", "PUT"])
    async def catch_all(path_name: str, request: Request):
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


# --- Mock Server Utilities ---
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
