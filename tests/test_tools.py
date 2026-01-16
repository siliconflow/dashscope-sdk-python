import pytest
import json
import requests
import os
from typing import Generator, List, Dict, Any, Optional
from dataclasses import dataclass

# --- 1. CONFIGURATION & CONSTANTS ---

# GATEWAY_URL = "https://api-bailian.siliconflow.cn/api/v1/services/aigc/text-generation/generation"
GATEWAY_URL = "http://localhost:8000/api/v1/services/aigc/text-generation/generation"
API_KEY = os.getenv("SILICONFLOW_API_KEY", "test_api_key")

TOOL_VECTOR_WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

# --- 2. CORE UTILITIES & DATA STRUCTURES ---


@dataclass
class SSEFrame:
    """Formal representation of a Server-Sent Event frame for validation."""

    id: str
    output: Dict[str, Any]
    usage: Dict[str, Any]
    request_id: str

    @property
    def text_content(self) -> str:
        """Helper to safely extract standard content."""
        choices = self.output.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("content", "")

    @property
    def reasoning_content(self) -> str:
        """Helper to safely extract reasoning content (for R1 models)."""
        choices = self.output.get("choices", [])
        if not choices:
            return ""
        return choices[0].get("message", {}).get("reasoning_content", "")


def parse_sse_stream(response: requests.Response) -> Generator[SSEFrame, None, None]:
    """
    Parses the raw SSE stream, enforcing protocol strictness.
    Yields structured frames for assertion logic.
    """
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data:"):
                json_str = decoded_line[5:].strip()
                try:
                    data = json.loads(json_str)
                    # Handle cases where usage might be missing in some frames if strictly required
                    usage_data = data.get("usage", {})

                    yield SSEFrame(
                        id=data.get("output", {})
                        .get("choices", [{}])[0]
                        .get("id", "unknown"),
                        output=data.get("output", {}),
                        usage=usage_data,
                        request_id=data.get("request_id", ""),
                    )
                except json.JSONDecodeError:
                    continue


# --- SUITE A: INVARIANT & PREDICATE VERIFICATION ---


def test_invariant_format_constraint():
    """
    Predicate A: If P_tools is not empty, P_result_format must be 'message'.
    """
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-v3",
        "input": {"messages": [{"role": "user", "content": "What's the weather?"}]},
        "parameters": {
            "tools": TOOL_VECTOR_WEATHER,
            "result_format": "text",  # <--- INTENTIONAL VIOLATION
        },
    }
    response = requests.post(GATEWAY_URL, headers=headers, json=payload)

    assert response.status_code == 400
    error_data = response.json()
    assert "code" in error_data
    assert "result_format" in str(error_data).lower()


def test_invariant_r1_orthogonality():
    """
    Predicate B: DeepSeek R1 'Thinking Mode' is orthogonal to 'Forced Tool Choice (Dict)'.
    """
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-r1",
        "input": {
            "messages": [{"role": "user", "content": "Analyze the weather logic."}]
        },
        "parameters": {
            "enable_thinking": True,
            "tools": TOOL_VECTOR_WEATHER,
            "tool_choice": {  # <--- INTENTIONAL VIOLATION: Specific Dict
                "type": "function",
                "function": {"name": "get_current_weather"},
            },
        },
    }
    response = requests.post(GATEWAY_URL, headers=headers, json=payload)

    assert response.status_code == 400
    assert "InvalidParameter" in response.json().get("code", "")


# --- SUITE B: PROTOCOL ISOMORPHISM (SSE TELEMETRY) ---


def test_telemetry_continuity_sse():
    """
    Theorem: The 'usage' object must be persisted in EVERY SSE frame.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "enable",
    }
    payload = {
        "model": "deepseek-v3",
        "input": {"messages": [{"role": "user", "content": "Call the tool."}]},
        "parameters": {
            "tools": TOOL_VECTOR_WEATHER,
            "result_format": "message",
            "incremental_output": True,
        },
    }

    response = requests.post(GATEWAY_URL, headers=headers, json=payload, stream=True)
    assert response.status_code == 200

    frame_count = 0
    for frame in parse_sse_stream(response):
        frame_count += 1
        assert frame.usage is not None
        assert "total_tokens" in frame.usage

    assert frame_count > 0


# --- SUITE C: TOOL INVOCATION & CONFIGURATION TESTS ---


def test_unary_tool_invocation_structure():
    """
    Validates standard unary responses maintain tool structures when tools enabled.
    """
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-v3",
        "input": {"messages": [{"role": "user", "content": "Check weather in Tokyo"}]},
        "parameters": {
            "tools": TOOL_VECTOR_WEATHER,
            "result_format": "message",
            "incremental_output": False,
            "tool_choice": "auto",
        },
    }
    response = requests.post(GATEWAY_URL, headers=headers, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "output" in data
    assert "choices" in data["output"]
    assert len(data["output"]["choices"]) > 0

    choice = data["output"]["choices"][0]
    assert "message" in choice
    assert "usage" in data


def test_tool_choice_none_suppression():
    """
    Validates that tool_choice='none' is accepted and processes without error.
    """
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-v3",
        "input": {"messages": [{"role": "user", "content": "What's the weather?"}]},
        "parameters": {
            "tools": TOOL_VECTOR_WEATHER,
            "tool_choice": "none",
            "result_format": "message",
        },
    }
    response = requests.post(GATEWAY_URL, headers=headers, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["output"]["choices"][0]["finish_reason"] is not None


# --- SUITE D: INCREMENTAL OUTPUT BEHAVIOR ---


def assert_stream_accumulation(frames: List[SSEFrame], check_reasoning: bool = False):
    """
    Validates 'Accumulated' behavior (incremental_output=False).
    Theorem: For any frame N > 0, Content(N) must start with Content(N-1).
    """
    previous_content = ""
    for i, frame in enumerate(frames):
        current_content = (
            frame.reasoning_content if check_reasoning else frame.text_content
        )

        # Skip empty frames
        if not current_content and not previous_content:
            continue

        assert current_content.startswith(previous_content), (
            f"Frame {i} violation: Output is not accumulated.\n"
            f"Previous: {previous_content!r}\n"
            f"Current:  {current_content!r}"
        )
        previous_content = current_content


def assert_stream_deltas(frames: List[SSEFrame], check_reasoning: bool = False):
    """
    Validates 'Delta' behavior (incremental_output=True).
    Theorem: Content(N) is independent of Content(N-1); it should not simply be a superset.
    """
    accumulation_detected = False
    previous_content = ""

    for i, frame in enumerate(frames):
        current_content = (
            frame.reasoning_content if check_reasoning else frame.text_content
        )
        if not current_content:
            continue

        # Heuristic: If content strictly grows and contains previous, it's likely accumulation
        if (
            previous_content
            and current_content.startswith(previous_content)
            and len(current_content) > len(previous_content)
        ):
            accumulation_detected = True
            break

        previous_content = current_content

    assert (
        not accumulation_detected
    ), "Stream appears to be accumulating full text, expected Deltas."


def test_incremental_output_false_explicit():
    """
    Case 1: Explicitly set incremental_output=False.
    Expectation: The response contains the FULL accumulated text in every frame.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "enable",
    }
    payload = {
        "model": "deepseek-r1",
        "input": {"messages": [{"role": "user", "content": "Count to 5"}]},
        "parameters": {
            "result_format": "message",
            "incremental_output": False,  # <--- EXPLICIT FALSE
        },
    }

    response = requests.post(GATEWAY_URL, headers=headers, json=payload, stream=True)
    assert response.status_code == 200
    frames = list(parse_sse_stream(response))
    assert len(frames) > 0
    assert_stream_accumulation(frames, check_reasoning=True)


def test_incremental_output_default_behavior():
    """
    Case 2: incremental_output param is MISSING.
    Expectation: Defaults to False (Accumulated behavior).
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "enable",
    }
    payload = {
        "model": "deepseek-r1",
        "input": {"messages": [{"role": "user", "content": "Who are you?"}]},
        "parameters": {
            "result_format": "message"
            # "incremental_output" is OMITTED
        },
    }

    response = requests.post(GATEWAY_URL, headers=headers, json=payload, stream=True)
    assert response.status_code == 200
    frames = list(parse_sse_stream(response))
    assert len(frames) > 0
    assert_stream_accumulation(frames, check_reasoning=True)


def test_incremental_output_true_contrast():
    """
    Case 3: Explicitly set incremental_output=True.
    Expectation: The response contains only DELTAS (chunks).
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "X-DashScope-SSE": "enable",
    }
    payload = {
        "model": "deepseek-r1",
        "input": {"messages": [{"role": "user", "content": "Hello"}]},
        "parameters": {
            "result_format": "message",
            "incremental_output": True,  # <--- EXPLICIT TRUE
        },
    }

    response = requests.post(GATEWAY_URL, headers=headers, json=payload, stream=True)
    assert response.status_code == 200
    frames = list(parse_sse_stream(response))
    assert len(frames) > 0
    assert_stream_deltas(frames, check_reasoning=True)


# --- SUITE E: BASIC GENERATION PARAMETERS ---


def test_deepseek_v3_with_temperature_and_logit_bias():
    """
    Test DeepSeek-V3 model with temperature and logit_bias parameters.
    Validates that advanced generation parameters are properly handled.
    """
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-ai/DeepSeek-V3",
        "input": {"prompt": "Hello"},
        "parameters": {
            "temperature": 0.7,
            "logit_bias": {"12345": 5.0, "67890": -100.0},
        },
    }

    response = requests.post(GATEWAY_URL, headers=headers, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "output" in data
    assert "choices" in data["output"]
    assert len(data["output"]["choices"]) > 0

    choice = data["output"]["choices"][0]
    assert "text" in choice or "message" in choice
    assert "usage" in data
