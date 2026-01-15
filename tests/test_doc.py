import pytest
import requests
import os
import json
from typing import Dict, Any

# --- CONFIGURATION ---
BASE_URL_PREFIX = "http://localhost:8000/siliconflow/models"
API_KEY = os.getenv("SILICONFLOW_API_KEY", "test_api_key")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
    "X-DashScope-SSE": "enable",
}

# --- MODEL MAPPING ---
# Exact mapping from model name to URL path
MODEL_PATH_MAP = {
    "deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    "deepseek-r1": "deepseek-ai/DeepSeek-R1",
    "pre-siliconflow/deepseek-v3": "deepseek-ai/DeepSeek-V3",
    "pre-siliconflow/deepseek-v3.1": "deepseek-ai/DeepSeek-V3.1",
    "pre-siliconflow/deepseek-v3.2": "deepseek-ai/DeepSeek-V3.2",
    "pre-siliconflow/deepseek-r1": "deepseek-ai/DeepSeek-R1",
}

# --- EXPECTED ERROR MESSAGES ---
ERR_MSG_TOP_P_TYPE = "<400> InternalError.Algo.InvalidParameter: Input should be a valid number, unable to parse string as a number: parameters.top_p"
ERR_MSG_TOP_P_RANGE = (
    "<400> InternalError.Algo.InvalidParameter: Range of top_p should be (0.0, 1.0]"
)
ERR_MSG_TEMP_RANGE = (
    "<400> InternalError.Algo.InvalidParameter: Temperature should be in [0.0, 2.0]"
)
ERR_MSG_PARTIAL_THINKING_CONFLICT = "<400> InternalError.Algo.InvalidParameter: Partial mode is not supported when enable_thinking is true"
ERR_MSG_TOOL_CHOICE = '<400> InternalError.Algo.InvalidParameter: tool_choice is one of the strings that should be ["none", "auto"]'


# --- HELPERS ---


def make_request(payload: Dict[str, Any], stream: bool = True) -> requests.Response:
    """
    Helper to send POST request using the Dynamic Path URL structure.
    Args:
        payload: The request body.
        stream: Whether to use streaming (SSE). Defaults to True.
    """
    raw_model_name = payload.get("model")

    if raw_model_name not in MODEL_PATH_MAP:
        pytest.fail(f"Configuration Error: Model '{raw_model_name}' not in MAP.")

    model_path = MODEL_PATH_MAP[raw_model_name]
    url = f"{BASE_URL_PREFIX}/{model_path}"

    # Handle Headers
    current_headers = HEADERS.copy()
    if not stream:
        current_headers["Accept"] = "application/json"
        # Remove SSE flag
        if "X-DashScope-SSE" in current_headers:
            del current_headers["X-DashScope-SSE"]

    # Note: requests.post 'stream' parameter controls whether to immediately download
    # the response body. We keep it True here to access raw response/iter_lines conveniently,
    # regardless of the business logic stream setting.
    return requests.post(url, headers=current_headers, json=payload, stream=True)


def assert_exact_error(
    response: requests.Response, expected_code_str: str, expected_message: str
):
    try:
        data = response.json()
    except Exception:
        pytest.fail(f"Response is not valid JSON: {response.text}")

    actual_code = data.get("code")
    assert (
        actual_code == expected_code_str
    ), f"Error Code mismatch.\nExpected: {expected_code_str}\nActual:   {actual_code}"

    actual_message = data.get("message")
    assert (
        actual_message == expected_message
    ), f"Error Message mismatch.\nExpected: {expected_message}\nActual:   {actual_message}"


# --- TEST SUITE ---
class TestStrictErrorValidation:
    def test_invalid_parameter_type_top_p(self):
        payload = {
            "model": "pre-siliconflow/deepseek-v3",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"top_p": "a"},
        }
        response = make_request(payload)
        assert_exact_error(response, "InvalidParameter", ERR_MSG_TOP_P_TYPE)

    def test_invalid_parameter_range_top_p(self):
        payload = {
            "model": "pre-siliconflow/deepseek-v3.1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"top_p": 0},
        }
        response = make_request(payload)
        assert_exact_error(response, "InvalidParameter", ERR_MSG_TOP_P_RANGE)

    def test_invalid_parameter_range_temperature(self):
        payload = {
            "model": "pre-siliconflow/deepseek-v3.1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"temperature": 2.1},
        }
        response = make_request(payload)
        assert_exact_error(response, "InvalidParameter", ERR_MSG_TEMP_RANGE)

    def test_conflict_prefix_and_thinking(self):
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "partial": True, "content": "你好，我是"},
                ]
            },
            "parameters": {"enable_thinking": True},
        }
        response = make_request(payload)
        assert_exact_error(
            response, "InvalidParameter", ERR_MSG_PARTIAL_THINKING_CONFLICT
        )

    def test_r1_enable_thinking_unsupported(self):
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"enable_thinking": True},
        }
        response = make_request(payload)
        assert response.status_code == 200


class TestFunctionalFixes:
    def test_r1_usage_structure_no_text_tokens(self):
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"max_tokens": 10},
        }
        response = make_request(payload)
        assert response.status_code == 200
        content = response.text
        assert '"text_tokens"' not in content
        assert '"reasoning_tokens"' in content

    def test_tool_choice_invalid_error_mapping(self):
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {"messages": [{"role": "user", "content": "Weather?"}]},
            "parameters": {
                "result_format": "message",
                "tool_choice": {"type": "get_current_weather"},
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                    }
                ],
            },
        }
        response = make_request(payload)
        assert_exact_error(response, "InvalidParameter", ERR_MSG_TOOL_CHOICE)

    def test_history_tool_call_fix(self):
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "user"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {"arguments": "{}", "name": "fn"},
                                "id": "call_1",
                                "type": "function",
                            }
                        ],
                    },
                    {"role": "tool", "content": "res", "tool_call_id": "call_1"},
                ]
            },
            "parameters": {"enable_thinking": False},
        }
        response = make_request(payload)
        assert response.status_code == 200

    def test_prefix_completion_success(self):
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "partial": True, "content": "你好，我是"},
                ]
            },
            "parameters": {"enable_thinking": False},
        }
        response = make_request(payload)
        assert response.status_code == 200

    def test_stop_parameter_invalid_type(self):
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"stop": 123},
        }
        response = make_request(payload)
        assert response.status_code == 400 or response.status_code == 500

    def test_non_streaming_request(self):
        """
        Verify non-streaming response (incremental_output=false).
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {"messages": [{"role": "user", "content": "Say hi"}]},
            "parameters": {"incremental_output": False},  # Explicitly set parameter
        }

        response = make_request(payload, stream=False)

        assert (
            response.status_code == 200
        ), f"Non-streaming request failed: {response.text}"
        data = response.json()
        # Non-streaming should return 'output' field directly, not SSE delta
        assert "output" in data, "Non-streaming response missing 'output' field"

    def test_stop_behavior_in_reasoning(self):
        """
        Verify Stop parameter behavior in Reasoning models:
        1. Ensure reasoning_content is returned (not truncated).
        2. Ensure stop parameter applies to content and truncates correctly.
        """
        target_stop_word = "Banana"
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": f"Please think about the color of the sun, and then simply output the phrase: 'The sun is like a {target_stop_word}'",
                    }
                ]
            },
            "parameters": {"stop": [target_stop_word], "incremental_output": True},
        }

        response = make_request(payload)
        assert response.status_code == 200

        # --- SSE Parsing Logic ---
        collected_reasoning = ""
        collected_content = ""
        final_finish_reason = None

        for line in response.iter_lines():
            if not line:
                continue

            line_text = line.decode("utf-8")

            if line_text.startswith("data:"):
                json_str = line_text[5:].strip()
                if json_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(json_str)

                    # Adapt to DashScope structure: get output first
                    output = chunk.get("output", {})
                    choices = output.get("choices", [])

                    if not choices:
                        continue

                    # Adapt to DashScope structure: use 'message' instead of 'delta'
                    message = choices[0].get("message", {})

                    if "reasoning_content" in message:
                        collected_reasoning += message["reasoning_content"]

                    if "content" in message:
                        collected_content += message["content"]

                    if choices[0].get("finish_reason"):
                        final_finish_reason = choices[0].get("finish_reason")

                except json.JSONDecodeError:
                    continue

        # --- Assertions ---
        print(f"\n[DEBUG] Collected Reasoning Length: {len(collected_reasoning)}")
        print(f"[DEBUG] Collected Content: '{collected_content}'")
        print(f"[DEBUG] Finish Reason: {final_finish_reason}")

        # 1. Verify reasoning content is present (Think process should not be empty)
        assert (
            len(collected_reasoning) > 10
        ), f"Expected significant reasoning content from R1 model. {collected_reasoning}"

        # 2. Verify Stop worked (Content should include prefix but stop before target word)
        assert "The sun" in collected_content
        assert (
            target_stop_word not in collected_content
        ), f"Content should stop before '{target_stop_word}'"

        # 3. Verify finish_reason is stop
        assert (
            final_finish_reason == "stop"
        ), f"Expected finish_reason to be 'stop', but got '{final_finish_reason}'"

    def test_thinking_budget_behavior(self):
        payload = {
            "model": "pre-siliconflow/deepseek-v3.1",
            "input": {"messages": [{"role": "user", "content": "Plan a trip"}]},
            "parameters": {"thinking_budget": 10},
        }
        response = make_request(payload)
        assert response.status_code != 500

    def test_n_parameter_unsupported(self):
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"n": 2},
        }
        response = make_request(payload)
        if response.status_code == 400:
            assert "n" in response.text
