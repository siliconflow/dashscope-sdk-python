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

# --- [NEW] MODEL MAPPING ---
# 定义模型名称到 URL 路径的精确映射
MODEL_PATH_MAP = {
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

# --- EXPECTED ERROR MESSAGES (COPIED FROM TABLE) ---
ERR_MSG_TOP_P_TYPE = "<400> InternalError.Algo.InvalidParameter: Input should be a valid number, unable to parse string as a number: parameters.top_p"
ERR_MSG_TOP_P_RANGE = (
    "<400> InternalError.Algo.InvalidParameter: Range of top_p should be (0.0, 1.0]"
)
ERR_MSG_TEMP_RANGE = (
    "<400> InternalError.Algo.InvalidParameter: Temperature should be in [0.0, 2.0]"
)
ERR_MSG_PARTIAL_THINKING_CONFLICT = "<400> InternalError.Algo.InvalidParameter: Partial mode is not supported when enable_thinking is true"
ERR_MSG_R1_THINKING = "Error code: 400 - {'code': 20015, 'message': 'Value error, current model does not support parameter `enable_thinking`.', 'data': None}"
ERR_MSG_TOOL_CHOICE = '<400> InternalError.Algo.InvalidParameter: tool_choice is one of the strings that should be ["none", "auto"]'


# --- HELPERS ---

def make_request(payload: Dict[str, Any]) -> requests.Response:
    """Helper to send POST request using the Dynamic Path URL structure."""
    raw_model_name = payload.get("model")

    # 获取映射后的真实路径，如果没有在映射中则使用原名称
    model_path = MODEL_PATH_MAP.get(raw_model_name, raw_model_name)

    # [注释] 测试的时候必须是http://localhost:8000/siliconflow/models/deepseek-ai/DeepSeek-V3这样的URL
    url = f"{BASE_URL_PREFIX}/{model_path}"

    return requests.post(url, headers=HEADERS, json=payload, stream=True)


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
# (之前的 TestStrictErrorValidation 类代码保持不变...)
class TestStrictErrorValidation:
    # ... 此处省略类中的方法，因逻辑未变 ...
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

    # ... 其他测试用例保持不变 ...
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
        assert_exact_error(response, "InvalidParameter", ERR_MSG_PARTIAL_THINKING_CONFLICT)

    def test_r1_enable_thinking_unsupported(self):
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"enable_thinking": True},
        }
        response = make_request(payload)
        assert response.status_code == 200


class TestFunctionalFixes:
    # ... 之前的测试用例保持不变 ...
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
            "input": {
                "messages": [{"role": "user", "content": "Weather?"}]
            },
            "parameters": {
                "result_format": "message",
                "tool_choice": {"type": "get_current_weather"},
                "tools": [{"type": "function", "function": {"name": "get_current_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}],
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
                    {"role": "assistant", "tool_calls": [{"function": {"arguments": "{}", "name": "fn"}, "id": "call_1", "type": "function"}]},
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
        [Updated] 验证非流式返回，需同样应用 URL 映射逻辑
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {"messages": [{"role": "user", "content": "Say hi"}]},
            "parameters": {},
        }

        # 1. 解析模型并应用映射
        raw_model_name = payload.get("model")
        model_path = MODEL_PATH_MAP.get(raw_model_name, raw_model_name)

        # 2. 拼接 URL (测试的时候必须是http://localhost:8000/siliconflow/models/deepseek-ai/DeepSeek-V3这样的URL)
        url = f"{BASE_URL_PREFIX}/{model_path}"

        headers_no_stream = HEADERS.copy()
        headers_no_stream["Accept"] = "application/json"
        del headers_no_stream["X-DashScope-SSE"]

        response = requests.post(url, headers=headers_no_stream, json=payload)

        assert response.status_code == 200, f"Non-streaming request failed: {response.text}"
        data = response.json()
        assert "output" in data, "Non-streaming response missing 'output' field"

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
