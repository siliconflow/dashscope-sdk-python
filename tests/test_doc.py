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
ERR_MSG_TOP_P_RANGE = "<400> InternalError.Algo.InvalidParameter: Range of top_p should be (0.0, 1.0]"
ERR_MSG_TEMP_RANGE = "<400> InternalError.Algo.InvalidParameter: Temperature should be in [0.0, 2.0]"
ERR_MSG_PARTIAL_THINKING_CONFLICT = "<400> InternalError.Algo.InvalidParameter: Partial mode is not supported when enable_thinking is true"
# R1 不支持 enable_thinking 的报错 (注意：表格中该报错包含 Python 字典的字符串表示，需严格匹配引号)
ERR_MSG_R1_THINKING = "Error code: 400 - {'code': 20015, 'message': 'Value error, current model does not support parameter `enable_thinking`.', 'data': None}"

# --- HELPERS ---

def make_request(payload: Dict[str, Any]) -> requests.Response:
    """Helper to send POST request using the Dynamic Path URL structure."""
    model_path = payload.get("model")
    url = f"{BASE_URL_PREFIX}/{model_path}"
    return requests.post(url, headers=HEADERS, json=payload, stream=True)

def assert_exact_error(response: requests.Response, expected_code_str: str, expected_message: str):
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
    assert actual_code == expected_code_str, f"Error Code mismatch.\nExpected: {expected_code_str}\nActual:   {actual_code}"

    # 2. Check Error Message (Exact String Match)
    actual_message = data.get("message")
    assert actual_message == expected_message, f"Error Message mismatch.\nExpected: {expected_message}\nActual:   {actual_message}"

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
            "parameters": {"top_p": "a"}
        }
        response = make_request(payload)

        # 根据表格预期，HTTP Code 可能是 400 或 500，但我们主要校验 Body 内容
        # 表格预期返回: code="InvalidParameter"
        assert_exact_error(
            response,
            expected_code_str="InvalidParameter",
            expected_message=ERR_MSG_TOP_P_TYPE
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
            "parameters": {"top_p": 0}
        }
        response = make_request(payload)

        assert_exact_error(
            response,
            expected_code_str="InvalidParameter",
            expected_message=ERR_MSG_TOP_P_RANGE
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
            "parameters": {"temperature": 2.1}
        }
        response = make_request(payload)

        assert_exact_error(
            response,
            expected_code_str="InvalidParameter",
            expected_message=ERR_MSG_TEMP_RANGE
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
                    {"role": "assistant", "partial": True, "content": "你好，我是"}
                ]
            },
            "parameters": {"enable_thinking": True}
        }
        response = make_request(payload)

        assert_exact_error(
            response,
            expected_code_str="InvalidParameter",
            expected_message=ERR_MSG_PARTIAL_THINKING_CONFLICT
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
            "parameters": {"enable_thinking": True}
        }
        response = make_request(payload)

        # 表格显示此处返回的是 InternalError，且 message 是上游透传回来的原始错误
        assert_exact_error(
            response,
            expected_code_str="InternalError",
            expected_message=ERR_MSG_R1_THINKING
        )

if __name__ == "__main__":
    pytest.main(["-v", __file__])
