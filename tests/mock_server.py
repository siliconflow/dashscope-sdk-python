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
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError

# --- [System Configuration] ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(process)d | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("DeepSeekProxy")

SILICON_FLOW_BASE_URL = os.getenv("SILICON_FLOW_BASE_URL", "https://api.siliconflow.cn/v1")
SILICON_FLOW_API_KEY = os.getenv("SILICON_FLOW_API_KEY")

MODEL_MAPPING = {
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "default": "deepseek-ai/DeepSeek-V3"
}

# --- [Shared State for Mock Mode] ---

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
        logger.info("Server transitioned to MOCK MODE via Queue Injection.")

    def increment_request(self):
        with self.lock:
            self.active_requests += 1

    def decrement_request(self):
        with self.lock:
            self.active_requests -= 1

    @property
    def snapshot(self):
        """Returns a consistent snapshot of the state."""
        with self.lock:
            return {
                "active_requests": self.active_requests,
                "is_mock_mode": self.is_mock_mode
            }

SERVER_STATE = ServerState.get_instance()

# --- [Pydantic Models] ---

class Message(BaseModel):
    role: str
    content: str

class InputData(BaseModel):
    messages: Optional[List[Message]] = None
    prompt: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

class Parameters(BaseModel):
    result_format: str = "message"
    incremental_output: bool = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 0.8
    top_k: Optional[int] = None
    seed: Optional[int] = 1234
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    enable_thinking: bool = False
    thinking_budget: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None

class GenerationRequest(BaseModel):
    model: str
    input: InputData
    parameters: Optional[Parameters] = Field(default_factory=Parameters)

# --- [DeepSeek Proxy Logic] ---

class DeepSeekProxy:
    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=SILICON_FLOW_API_KEY if SILICON_FLOW_API_KEY else "dummy_key",
            base_url=SILICON_FLOW_BASE_URL
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
        params = req_data.parameters

        if params.tools and params.result_format != "message":
            return JSONResponse(
                status_code=400,
                content={"code": "InvalidParameter", "message": "When 'tools' are provided, 'result_format' must be 'message'."}
            )

        is_r1 = "deepseek-r1" in req_data.model or params.enable_thinking
        if is_r1 and params.tool_choice and isinstance(params.tool_choice, dict):
             return JSONResponse(
                status_code=400,
                content={"code": "InvalidParameter", "message": "DeepSeek R1 does not support specific tool_choice binding (dict) while thinking logic is active."}
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

        if params.tools:
            openai_params["tools"] = params.tools
            if params.tool_choice:
                openai_params["tool_choice"] = params.tool_choice

        if params.max_tokens: openai_params["max_tokens"] = params.max_tokens
        if params.stop: openai_params["stop"] = params.stop
        if params.seed: openai_params["seed"] = params.seed

        try:
            if openai_params["stream"]:
                # Fetch raw response for headers in stream mode (awaited)
                raw_resp = await self.client.chat.completions.with_raw_response.create(**openai_params)
                trace_id = raw_resp.headers.get("X-SiliconCloud-Trace-Id", initial_request_id)

                # raw_resp.parse() returns the AsyncStream
                return StreamingResponse(
                    self._stream_generator(raw_resp.parse(), trace_id),
                    media_type="text/event-stream",
                    headers={"X-SiliconCloud-Trace-Id": trace_id}  # <--- Added Header Propagation
                )
            else:
                # Standard response (awaited)
                raw_resp = await self.client.chat.completions.with_raw_response.create(**openai_params)
                trace_id = raw_resp.headers.get("X-SiliconCloud-Trace-Id", initial_request_id)
                return self._format_unary_response(raw_resp.parse(), trace_id)

        except APIError as e:
            logger.error(f"Upstream API Error: {str(e)}")
            error_code = "InternalError"
            if isinstance(e, RateLimitError): error_code = "Throttling.RateQuota"
            elif isinstance(e, AuthenticationError): error_code = "InvalidApiKey"

            return JSONResponse(
                status_code=e.status_code or 500,
                content={"code": error_code, "message": str(e), "request_id": initial_request_id}
            )

    async def _stream_generator(self, stream, request_id: str) -> AsyncGenerator[str, None]:
        accumulated_usage = {
            "total_tokens": 0, "input_tokens": 0, "output_tokens": 0,
            "output_tokens_details": {"text_tokens": 0, "reasoning_tokens": 0}
        }
        finish_reason = "null"

        async for chunk in stream:
            if chunk.usage:
                accumulated_usage["total_tokens"] = chunk.usage.total_tokens
                accumulated_usage["input_tokens"] = chunk.usage.prompt_tokens
                accumulated_usage["output_tokens"] = chunk.usage.completion_tokens
                details = getattr(chunk.usage, "completion_tokens_details", None)
                if details:
                    accumulated_usage["output_tokens_details"]["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0)
                    accumulated_usage["output_tokens_details"]["text_tokens"] = accumulated_usage["output_tokens"] - accumulated_usage["output_tokens_details"]["reasoning_tokens"]

            delta = chunk.choices[0].delta if chunk.choices else None
            content = delta.content if delta and delta.content else ""
            reasoning = getattr(delta, "reasoning_content", "") if delta else ""

            tool_calls = None
            if delta and delta.tool_calls:
                tool_calls = [tc.model_dump() for tc in delta.tool_calls]

            if chunk.choices and chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

            message_body = {"role": "assistant", "content": content, "reasoning_content": reasoning}
            if tool_calls: message_body["tool_calls"] = tool_calls

            response_body = {
                "output": {"choices": [{"message": message_body, "finish_reason": finish_reason}]},
                "usage": accumulated_usage,
                "request_id": request_id
            }
            yield f"data: {json.dumps(response_body, ensure_ascii=False)}\n\n"
            if finish_reason != "null": break

    def _format_unary_response(self, completion, request_id: str):
        choice = completion.choices[0]
        msg = choice.message
        usage_data = {
            "total_tokens": completion.usage.total_tokens,
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
            "output_tokens_details": {"text_tokens": 0, "reasoning_tokens": 0}
        }
        details = getattr(completion.usage, "completion_tokens_details", None)
        if details:
             usage_data["output_tokens_details"]["reasoning_tokens"] = getattr(details, "reasoning_tokens", 0)
             usage_data["output_tokens_details"]["text_tokens"] = usage_data["output_tokens"] - usage_data["output_tokens_details"]["reasoning_tokens"]

        message_body = {
            "role": msg.role,
            "content": msg.content,
            "reasoning_content": getattr(msg, "reasoning_content", "")
        }
        if msg.tool_calls:
            message_body["tool_calls"] = [tc.model_dump() for tc in msg.tool_calls]

        response_body = {
            "output": {"choices": [{"message": message_body, "finish_reason": choice.finish_reason}]},
            "usage": usage_data,
            "request_id": request_id
        }

        return JSONResponse(
            content=response_body,
            headers={"X-SiliconCloud-Trace-Id": request_id}
        )

# --- [FastAPI App & Lifecycle] ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    stop_event = threading.Event()
    def epoch_clock():
        while not stop_event.is_set():
            time.sleep(2)
            state = SERVER_STATE.snapshot
            if state["active_requests"] > 0 or state["is_mock_mode"]:
                 logger.info(
                    f"[Epoch Clock] Active Requests: {state['active_requests']} | "
                    f"Mode: {'MOCK' if state['is_mock_mode'] else 'PROXY'}"
                )
    monitor_thread = threading.Thread(target=epoch_clock, daemon=True)
    monitor_thread.start()
    yield
    stop_event.set()

def create_app() -> FastAPI:
    app = FastAPI(title="DeepSeek-DashScope Proxy", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
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
            logger.info(f"{request.method} {request.url.path} - {duration:.2f}ms")

    @app.get("/health_check")
    async def health_check():
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "DeepSeek-DashScope-Proxy",
                "timestamp": time.time(),
                "mode": "mock" if SERVER_STATE.is_mock_mode else "production"
            }
        )

    @app.post("/api/v1/services/aigc/text-generation/generation")
    async def generation(request: Request, body: GenerationRequest = None):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))

        # Instantiate Proxy Per Request
        proxy = DeepSeekProxy()

        if not body:
            try:
                raw_json = await request.json()
                body = GenerationRequest(**raw_json)
            except Exception as e:
                if not SERVER_STATE.is_mock_mode:
                    raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        if SERVER_STATE.is_mock_mode:
            if body:
                logger.info(f"[Shadow] Validating request against upstream...")
                try:
                    # Async generate call on the local instance
                    await proxy.generate(body, f"shadow-{request_id}")
                except Exception as e:
                    logger.error(f"[Shadow] Validation Exception: {str(e)}")

            try:
                raw_body = await request.json()
                SERVER_STATE.request_queue.put(raw_body)
                response_data = SERVER_STATE.response_queue.get(timeout=10)
                response_json = json.loads(response_data) if isinstance(response_data, str) else response_data
                status_code = response_json.pop("status_code", 200)
                return JSONResponse(content=response_json, status_code=status_code)
            except Exception as e:
                logger.critical(f"[Mock] DEADLOCK/ERROR: {e}")
                return JSONResponse(status_code=500, content={"code": "MockError", "message": f"Mock Server Error: {str(e)}"})

        return await proxy.generate(body, request_id)

    @app.api_route("/{path_name:path}", methods=["GET", "POST", "DELETE", "PUT"])
    async def catch_all(path_name: str, request: Request):
        if SERVER_STATE.is_mock_mode:
            try:
                body = None
                if request.method in ["POST", "PUT"]:
                    try: body = await request.json()
                    except: pass
                req_record = {"path": f"/{path_name}", "method": request.method, "headers": dict(request.headers), "body": body}
                SERVER_STATE.request_queue.put(req_record)
                response_data = SERVER_STATE.response_queue.get(timeout=5)
                response_json = json.loads(response_data) if isinstance(response_data, str) else response_data
                status_code = response_json.pop("status_code", 200)
                return JSONResponse(content=response_json, status_code=status_code)
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": "Mock Catch-All Error", "detail": str(e)})

        return JSONResponse(status_code=404, content={"code": "NotFound", "message": "Proxy endpoint not implemented"})

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
    proc = multiprocessing.Process(target=run_server_process, args=(mock_server.requests, mock_server.responses, "0.0.0.0", 8089))
    proc.start()
    mock_server.proc = proc
    time.sleep(1.5)
    logger.info("Mock Server (Proxy Mode) started on port 8089")
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
