import pytest
import json
import requests
from typing import Generator, List, Dict, Any
from dataclasses import dataclass

# --- CONSTANTS & CONFIGURATION ---
# Note: Ensure your MockServer/Proxy is running on this port
GATEWAY_URL = "http://localhost:8000/api/v1/services/aigc/text-generation/generation"
API_KEY = "sk-test-vector-integrity"

# Define the Tool Schema Vector
TOOL_VECTOR_WEATHER = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["location"]
            }
        }
    }
]

@dataclass
class SSEFrame:
    """Formal representation of a Server-Sent Event frame for validation."""
    id: str
    output: Dict[str, Any]
    usage: Dict[str, Any]
    request_id: str

def parse_sse_stream(response: requests.Response) -> Generator[SSEFrame, None, None]:
    """
    Parses the raw SSE stream, enforcing protocol strictness.
    Yields structured frames for assertion logic.
    """
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith('data:'):
                json_str = decoded_line[5:].strip()
                try:
                    data = json.loads(json_str)
                    yield SSEFrame(
                        id=data.get('output', {}).get('choices', [{}])[0].get('id', 'unknown'),
                        output=data.get('output', {}),
                        usage=data.get('usage', {}),
                        request_id=data.get('request_id', '')
                    )
                except json.JSONDecodeError:
                    continue

# --- SUITE A: INVARIANT & PREDICATE VERIFICATION ---

def test_invariant_format_constraint():
    """
    Predicate A: If P_tools is not empty, P_result_format must be 'message'.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-v3",
        "input": {"messages": [{"role": "user", "content": "What's the weather?"}]},
        "parameters": {
            "tools": TOOL_VECTOR_WEATHER,
            "result_format": "text"  # <--- INTENTIONAL VIOLATION
        }
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
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-r1",
        "input": {"messages": [{"role": "user", "content": "Analyze the weather logic."}]},
        "parameters": {
            "enable_thinking": True,
            "tools": TOOL_VECTOR_WEATHER,
            "tool_choice": {  # <--- INTENTIONAL VIOLATION: Specific Dict
                "type": "function",
                "function": {"name": "get_current_weather"}
            }
        }
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
        "X-DashScope-SSE": "enable"
    }

    payload = {
        "model": "deepseek-v3",
        "input": {"messages": [{"role": "user", "content": "Call the tool."}]},
        "parameters": {
            "tools": TOOL_VECTOR_WEATHER,
            "result_format": "message",
            "incremental_output": True
        }
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
    Validates that standard unary (non-streaming) responses maintain proper tool structures
    when tools are enabled but not necessarily forced.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-v3",
        "input": {"messages": [{"role": "user", "content": "Check weather in Tokyo"}]},
        "parameters": {
            "tools": TOOL_VECTOR_WEATHER,
            "result_format": "message",
            "incremental_output": False,
            "tool_choice": "auto"
        }
    }
    response = requests.post(GATEWAY_URL, headers=headers, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "output" in data
    assert "choices" in data["output"]
    assert len(data["output"]["choices"]) > 0

    # Check structure integrity
    choice = data["output"]["choices"][0]
    assert "message" in choice
    assert "usage" in data

def test_tool_choice_none_suppression():
    """
    Validates that tool_choice='none' is accepted by the gateway and processed without error.
    This ensures the explicit suppression logic path is valid.
    """
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-v3",
        "input": {"messages": [{"role": "user", "content": "What's the weather?"}]},
        "parameters": {
            "tools": TOOL_VECTOR_WEATHER,
            "tool_choice": "none",
            "result_format": "message"
        }
    }
    response = requests.post(GATEWAY_URL, headers=headers, json=payload)
    assert response.status_code == 200

    # Even if mocked, the structure must be valid
    data = response.json()
    assert data["output"]["choices"][0]["finish_reason"] is not None
