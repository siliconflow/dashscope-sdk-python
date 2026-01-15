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
# R1 不支持 enable_thinking 的报错 (注意：表格中该报错包含 Python 字典的字符串表示，需严格匹配引号)
ERR_MSG_R1_THINKING = "Error code: 400 - {'code': 20015, 'message': 'Value error, current model does not support parameter `enable_thinking`.', 'data': None}"

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
    1. HTTP 状态码通常为 4xx 或 500 (根据表格，部分 4xx 业务错误可能返回 200 或 400，此处以解析 body 为主)
    2. JSON body 中的 code 字段
    3. JSON body 中的 message 字段 (逐字匹配)
    """
    try:
        data = response.json()
    except Exception:
        pytest.fail(f"Response is not valid JSON: {response.text}")

    # 1. Check Error Code (e.g., 'InvalidParameter' or 'InternalError')
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
        Expected: InvalidParameter, <400> ... unable to parse string as a number
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"top_p": "a"},
        }
        response = make_request(payload)

        # 根据表格预期，HTTP Code 可能是 400 或 500，但我们主要校验 Body 内容
        # 表格预期返回: code="InvalidParameter"
        assert_exact_error(
            response,
            expected_code_str="InvalidParameter",
            expected_message=ERR_MSG_TOP_P_TYPE,
        )

    def test_invalid_parameter_range_top_p(self):
        """
        表格行: pre-siliconflow-deepseek-v3.1 top_p取值范围（0,1.0]
        Input: top_p = 0
        Expected: InvalidParameter, Range of top_p should be (0.0, 1.0]
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
        Input: temperature = 2.1
        Expected: InvalidParameter, Temperature should be in [0.0, 2.0]
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
        Input: partial=True AND enable_thinking=True
        Expected: InvalidParameter, Partial mode is not supported when enable_thinking is true
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
        Input: model=deepseek-r1, enable_thinking=True
        Expected: InternalError, Error code: 400 - {'code': 20015...}
        """
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"enable_thinking": True},
        }
        response = make_request(payload)

        # 表格显示此处返回的是 InternalError，且 message 是上游透传回来的原始错误
        assert_exact_error(
            response,
            expected_code_str="InternalError",
            expected_message=ERR_MSG_R1_THINKING,
        )


class TestFunctionalFixes:
    """
    测试表格中提到的功能修复验证 (Verify Fixes) 和特定的字段格式检查
    """

    def test_r1_usage_structure_no_text_tokens(self):
        """
        表格行: .usage.output_tokens_details 该路径下不应该返回 text_tokens 字段
        Model: pre-siliconflow/deepseek-r1
        Check: usage.output_tokens_details 应该只包含 reasoning_tokens
        """
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"max_tokens": 10} # 限制输出以加快测试
        }
        response = make_request(payload)

        assert response.status_code == 200, f"Request failed with {response.status_code}"

        # 解析 SSE 流或直接读取 JSON (假设是非流式或读取最后一块)
        # 这里简化处理：如果是流式，我们需要解析最后一个包含 usage 的块
        # 为了测试稳定性，建议在此处强制是非流式请求，或者手动解析SSE
        # 这里演示解析 JSON Body (如果接口支持非流式返回) 或者解析 SSE

        # 简单起见，读取整个流并查找 usage
        content = response.text
        assert '"text_tokens"' not in content, "Response output_tokens_details should NOT contain 'text_tokens'"
        assert '"reasoning_tokens"' in content, "Response output_tokens_details MUST contain 'reasoning_tokens'"

    def test_tool_choice_invalid_error_mapping(self):
        """
        表格行: Error code: 400，_sse_http_status": 500
        描述: 传递了不符合协议的 tool_choice (传递了type但没有function定义，或格式错误)
        Expected: 返回明确的 InternalError 且包含 upstream 的 400 信息
        """
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {
                "messages": [{"role": "user", "content": "What is the weather like in Boston?"}]
            },
            "parameters": {
                "result_format": "message",
                # 错误的 tool_choice 格式，导致上游报错
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
                                "required": ["location"]
                            }
                        }
                    }
                ]
            }
        }
        response = make_request(payload)
        data = response.json()

        expected_msg_snippet = "Input should be 'none', 'auto' or 'required'"

        assert data.get("code") == "InternalError"
        # 验证错误信息是否透传了上游的详细校验失败信息
        assert expected_msg_snippet in data.get("message"), \
            f"Expected error message to contain '{expected_msg_snippet}', got: {data.get('message')}"

    def test_history_tool_call_fix(self):
        """
        表格行: 3.1和3.2 message中包含历史tool_call调用信息会报5xx -> 修复验证
        Model: pre-siliconflow/deepseek-v3.2
        Input: 包含 system, user, assistant(tool_calls), tool(result) 的完整历史
        Expected: 200 OK (之前报 500)
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个智能助手。"
                    },
                    {
                        "role": "user",
                        "content": "外部轴设置"
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": "{\"input_text\": \"外部轴设置\"}",
                                    "name": "KB20250625001"
                                },
                                "id": "call_6478091069c2448b83f38e",
                                "type": "function"
                            }
                        ]
                    },
                    {
                        "role": "tool",
                        "content": "界面用于用户进行快速配置。",
                        "tool_call_id": "call_6478091069c2448b83f38e"
                    }
                ]
            },
            # 确保不开启思考，避免干扰测试 tool history 功能
            "parameters": {"enable_thinking": False}
        }

        response = make_request(payload)
        assert response.status_code == 200, f"Previously 500 error scenario failed. Got: {response.text}"

    def test_prefix_completion_success(self):
        """
        表格行: 前缀续写，pre-siliconflow-deepseek-v3.2 报500 -> 修复验证
        Model: pre-siliconflow/deepseek-v3.2
        Scenario: Assistant 消息带 partial=True
        Expected: 200 OK (只要不开启 enable_thinking)
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {
                "messages": [
                    {"role": "user", "content": "你好"},
                    {
                        "role": "assistant",
                        "partial": True,
                        "content": "你好，我是"
                    }
                ]
            },
            # 明确关闭 thinking 以测试单纯的前缀续写功能
            "parameters": {"enable_thinking": False}
        }

        response = make_request(payload)
        assert response.status_code == 200, f"Prefix completion failed with {response.status_code}. Body: {response.text}"

        # 可选：验证返回内容确实是以前缀开始的续写
        # context = response.text...
