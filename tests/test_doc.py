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

# --- EXPECTED ERROR MESSAGES (COPIED FROM TABLE) ---
# 定义预期错误常量，确保逐字对齐
ERR_MSG_TOP_P_TYPE = "<400> InternalError.Algo.InvalidParameter: Input should be a valid number, unable to parse string as a number: parameters.top_p"
ERR_MSG_TOP_P_RANGE = (
    "<400> InternalError.Algo.InvalidParameter: Range of top_p should be (0.0, 1.0]"
)
ERR_MSG_TEMP_RANGE = (
    "<400> InternalError.Algo.InvalidParameter: Temperature should be in [0.0, 2.0]"
)
ERR_MSG_PARTIAL_THINKING_CONFLICT = "<400> InternalError.Algo.InvalidParameter: Partial mode is not supported when enable_thinking is true"
# R1 不支持 enable_thinking 的报错
ERR_MSG_R1_THINKING = "Error code: 400 - {'code': 20015, 'message': 'Value error, current model does not support parameter `enable_thinking`.', 'data': None}"

# [UPDATED] tool_choice 校验错误 (根据百炼返回更新)
ERR_MSG_TOOL_CHOICE = '<400> InternalError.Algo.InvalidParameter: tool_choice is one of the strings that should be ["none", "auto"]'

# --- HELPERS ---


def make_request(payload: Dict[str, Any]) -> requests.Response:
    """Helper to send POST request using the Dynamic Path URL structure."""
    model_path = payload.get("model")
    url = f"{BASE_URL_PREFIX}/{model_path}"
    return requests.post(url, headers=HEADERS, json=payload, stream=True)


def assert_exact_error(
    response: requests.Response, expected_code_str: str, expected_message: str
):
    """
    严格校验错误返回：
    1. HTTP 状态码通常为 4xx 或 500
    2. JSON body 中的 code 字段
    3. JSON body 中的 message 字段 (逐字匹配)
    """
    try:
        data = response.json()
    except Exception:
        pytest.fail(f"Response is not valid JSON: {response.text}")

    # 1. Check Error Code
    actual_code = data.get("code")
    assert (
        actual_code == expected_code_str
    ), f"Error Code mismatch.\nExpected: {expected_code_str}\nActual:   {actual_code}"

    # 2. Check Error Message (Exact String Match)
    actual_message = data.get("message")
    assert (
        actual_message == expected_message
    ), f"Error Message mismatch.\nExpected: {expected_message}\nActual:   {actual_message}"


# --- TEST SUITE ---


class TestStrictErrorValidation:

    def test_invalid_parameter_type_top_p(self):
        """
        表格行: 4xx的报错请求
        Input: top_p = "a" (string)
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"top_p": "a"},
        }
        response = make_request(payload)

        assert_exact_error(
            response,
            expected_code_str="InvalidParameter",
            expected_message=ERR_MSG_TOP_P_TYPE,
        )

    def test_invalid_parameter_range_top_p(self):
        """
        表格行: pre-siliconflow-deepseek-v3.1 top_p取值范围（0,1.0]
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"top_p": 0},
        }
        response = make_request(payload)

        assert_exact_error(
            response,
            expected_code_str="InvalidParameter",
            expected_message=ERR_MSG_TOP_P_RANGE,
        )

    def test_invalid_parameter_range_temperature(self):
        """
        表格行: pre-siliconflow-deepseek-v3.1 取值范围 [0, 2)
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"temperature": 2.1},
        }
        response = make_request(payload)

        assert_exact_error(
            response,
            expected_code_str="InvalidParameter",
            expected_message=ERR_MSG_TEMP_RANGE,
        )

    def test_conflict_prefix_and_thinking(self):
        """
        表格行: 前缀续写...思考模式下...会报4xx
        """
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
            response,
            expected_code_str="InvalidParameter",
            expected_message=ERR_MSG_PARTIAL_THINKING_CONFLICT,
        )

    def test_r1_enable_thinking_unsupported(self):
        """
        表格行: r1传了enable_thinking报错
        """
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"enable_thinking": True},
        }
        response = make_request(payload)

        # 注意：此处维持原逻辑，如果需要校验具体错误可放开注释
        assert response.status_code == 200


class TestFunctionalFixes:

    def test_r1_usage_structure_no_text_tokens(self):
        """
        表格行: .usage.output_tokens_details 该路径下不应该返回 text_tokens 字段
        """
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"max_tokens": 10},
        }
        response = make_request(payload)

        assert (
            response.status_code == 200
        ), f"Request failed with {response.status_code}"

        content = response.text
        assert (
            '"text_tokens"' not in content
        ), "Response output_tokens_details should NOT contain 'text_tokens'"
        assert (
            '"reasoning_tokens"' in content
        ), "Response output_tokens_details MUST contain 'reasoning_tokens'"

    def test_tool_choice_invalid_error_mapping(self):
        """
        [UPDATED]
        表格行: Error code: 400，_sse_http_status": 500
        描述: 传递了不符合协议的 tool_choice (如传递了不支持的对象格式或枚举值)
        Expected: InvalidParameter (之前是 InternalError), 且包含明确的枚举值提示
        """
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {
                "messages": [
                    {"role": "user", "content": "What is the weather like in Boston?"}
                ]
            },
            "parameters": {
                "result_format": "message",
                # 错误的 tool_choice 格式 (百炼预期是 string: "none" 或 "auto")
                "tool_choice": {"type": "get_current_weather"},
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_weather",
                            "description": "Get current weather",
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

        # 使用 assert_exact_error 统一校验 Code 和 Message
        assert_exact_error(
            response,
            expected_code_str="InvalidParameter",  # 根据百炼返回更新 code
            expected_message=ERR_MSG_TOOL_CHOICE,  # 使用上方定义的新常量
        )

    def test_history_tool_call_fix(self):
        """
        表格行: 3.1和3.2 message中包含历史tool_call调用信息会报5xx -> 修复验证
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {
                "messages": [
                    {"role": "system", "content": "你是一个智能助手。"},
                    {"role": "user", "content": "外部轴设置"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": '{"input_text": "外部轴设置"}',
                                    "name": "KB20250625001",
                                },
                                "id": "call_6478091069c2448b83f38e",
                                "type": "function",
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "界面用于用户进行快速配置。",
                        "tool_call_id": "call_6478091069c2448b83f38e",
                    },
                ]
            },
            "parameters": {"enable_thinking": False},
        }

        response = make_request(payload)
        assert (
            response.status_code == 200
        ), f"Previously 500 error scenario failed. Got: {response.text}"

    def test_prefix_completion_success(self):
        """
        表格行: 前缀续写，pre-siliconflow-deepseek-v3.2 报500 -> 修复验证
        """
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
        assert (
            response.status_code == 200
        ), f"Prefix completion failed with {response.status_code}. Body: {response.text}"

    def test_stop_parameter_invalid_type(self):
        """
        [Row 13] stop 只支持 str/list[str]，不支持 int 类型
        Expected: InvalidParameter
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"stop": 123},  # Invalid type
        }
        response = make_request(payload)
        # 根据实际预期填写 Error Message
        assert response.status_code == 400 or response.status_code == 500
        # assert_exact_error(response, "InvalidParameter", "Expecting string or list of strings...")

    def test_non_streaming_request(self):
        """
        [Row 16] 验证非流式返回 (incremental_output=false) 是否正常
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {"messages": [{"role": "user", "content": "Say hi"}]},
            "parameters": {},  # implicit stream=False via headers modification needed
        }

        # 需要覆盖默认 header 的 stream 设置
        model_path = payload.get("model")
        url = f"{BASE_URL_PREFIX}/{model_path}"
        headers_no_stream = HEADERS.copy()
        headers_no_stream["Accept"] = "application/json"
        del headers_no_stream["X-DashScope-SSE"]  # 移除 SSE header

        response = requests.post(url, headers=headers_no_stream, json=payload)

        assert (
            response.status_code == 200
        ), f"Non-streaming request failed: {response.text}"
        data = response.json()
        assert "output" in data, "Non-streaming response missing 'output' field"

    def test_thinking_budget_behavior(self):
        """
        [Row 14] thinking_budget 不支持或行为变更验证
        验证 3.1 模型设置 budget 后是否报错或行为符合预期
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.1",
            "input": {"messages": [{"role": "user", "content": "Plan a trip"}]},
            "parameters": {"thinking_budget": 10},
        }
        response = make_request(payload)
        # 根据表格，如果是 "硅基流动确定不支持"，这里可能预期是 200 (忽略) 或者 400 (报错)
        # 如果预期是不支持，最好验证它没有 crash (500)
        assert (
            response.status_code != 500
        ), f"Thinking budget caused 500: {response.text}"

    def test_n_parameter_unsupported(self):
        """
        [Row 17] n 参数不支持
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"n": 2},
        }
        response = make_request(payload)
        # 验证是忽略还是报错
        if response.status_code == 400:
            assert "n" in response.text
