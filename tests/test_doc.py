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

    # 处理 Headers
    current_headers = HEADERS.copy()
    if not stream:
        current_headers["Accept"] = "application/json"
        # 移除 SSE 标记
        if "X-DashScope-SSE" in current_headers:
            del current_headers["X-DashScope-SSE"]

    # 注意：requests.post 的 stream 参数控制是否立即下载响应体，
    # 但业务层面的流式由 headers 和 json body 决定，这里保持 requests 的 stream=True
    # 只是为了方便拿到原始响应，或者统一设为 stream 变量也可以。
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
        [Row 16] 验证非流式返回 (incremental_output=false)
        使用优化后的 make_request，代码更干净
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {"messages": [{"role": "user", "content": "Say hi"}]},
            "parameters": {"incremental_output": False}, # 显式设置参数
        }

        # 调用优化后的 helper，传入 stream=False
        response = make_request(payload, stream=False)

        assert response.status_code == 200, f"Non-streaming request failed: {response.text}"
        data = response.json()
        # 非流式应该直接返回 output 字段，而不是 SSE 的 delta
        assert "output" in data, "Non-streaming response missing 'output' field"

    def test_stop_behavior_in_reasoning(self):
        """
        [Updated] 验证 Stop 参数在 Reasoning 模型中的行为：
        1. 确认 reasoning_content 能够正常返回 (不被误截断)。
        2. 确认 stop 参数作用于正文 (content)，并正确截断。
        """
        # 设置一个具体的 Stop 词，诱导模型在正文中输出它
        target_stop_word = "Banana"
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {
                "messages": [{
                    "role": "user",
                    "content": f"Please think about the color of the sun, and then simply output the phrase: 'The sun is like a {target_stop_word}'"
                }]
            },
            "parameters": {
                "stop": [target_stop_word],
                # R1 自动开启思考，但也可能需要显式参数，视具体后端实现而定，这里保持默认或按需添加
            }
        }

        response = make_request(payload)
        assert response.status_code == 200

        # --- SSE 解析逻辑 ---
        collected_reasoning = ""
        collected_content = ""
        final_finish_reason = None

        for line in response.iter_lines():
            if not line:
                continue

            # 解码 line
            line_text = line.decode('utf-8')

            if line_text.startswith("data:"):
                json_str = line_text[5:].strip()
                if json_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(json_str)
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})

                    # 1. 收集 reasoning_content
                    # 注意：有些实现可能叫 reasoning_text，视具体 API 定义，这里假设是 reasoning_content
                    if "reasoning_content" in delta:
                        collected_reasoning += delta["reasoning_content"]

                    # 2. 收集正文 content
                    if "content" in delta:
                        collected_content += delta["content"]

                    # 3. 捕捉 finish_reason
                    if choices[0].get("finish_reason"):
                        final_finish_reason = choices[0].get("finish_reason")

                except json.JSONDecodeError:
                    continue

        # --- 断言验证 ---
        print(f"\n[DEBUG] Collected Reasoning Length: {len(collected_reasoning)}")
        print(f"[DEBUG] Collected Content: '{collected_content}'")
        print(f"[DEBUG] Finish Reason: {final_finish_reason}")

        # 1. 验证 reasoning 内容回来了 (Think 过程不应为空)
        assert len(collected_reasoning) > 10, "Expected significant reasoning content from R1 model."

        # 2. 验证 Stop 生效 (Content 应该包含 'The sun is like a ' 但不包含 'Banana')
        assert "The sun" in collected_content
        assert target_stop_word not in collected_content, f"Content should stop before '{target_stop_word}'"

        # 3. 验证 finish_reason 是 stop
        assert final_finish_reason == "stop", f"Expected finish_reason to be 'stop', but got '{final_finish_reason}'"

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
