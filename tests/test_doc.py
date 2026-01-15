import pytest
import json
import requests
from typing import Generator, List, Dict, Any
from dataclasses import dataclass
import os

# --- CONSTANTS & CONFIGURATION ---
GATEWAY_URL = "http://localhost:8000/api/v1/services/aigc/text-generation/generation"
# 如果需要测试硅基流动的真实环境，请切换 URL 并设置 API KEY
# GATEWAY_URL = "https://api-bailian.siliconflow.cn/api/v1/services/aigc/text-generation/generation"

API_KEY = os.getenv("SILICONFLOW_API_KEY", "test_api_key")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
    "X-DashScope-SSE": "enable",  # 模拟 DashScope 协议头
}

# --- TOOL DEFINITIONS ---
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

# --- HELPERS ---


@dataclass
class SSEFrame:
    """Formal representation of a Server-Sent Event frame for validation."""

    id: str
    output: Dict[str, Any]
    usage: Dict[str, Any]
    request_id: str


def parse_sse_stream(response: requests.Response) -> Generator[SSEFrame, None, None]:
    """Parses the raw SSE stream."""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data:"):
                json_str = decoded_line[5:].strip()
                try:
                    data = json.loads(json_str)
                    yield SSEFrame(
                        id=data.get("output", {})
                        .get("choices", [{}])[0]
                        .get("id", "unknown"),
                        output=data.get("output", {}),
                        usage=data.get("usage", {}),
                        request_id=data.get("request_id", ""),
                    )
                except json.JSONDecodeError:
                    continue


def make_request(payload):
    """Helper to send POST request."""
    return requests.post(GATEWAY_URL, headers=HEADERS, json=payload, stream=True)


# --- TEST SUITE ---


class TestParameterValidation:
    """
    对应表格中参数校验相关的错误用例 (4xx Error Codes)
    """

    def test_invalid_parameter_type_top_p(self):
        """
        Case: parameters.top_p 输入字符串 'a'，预期返回 400 InvalidParameter。
        Bug描述: 曾返回 InternalError 500。
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"top_p": "a"},  # Invalid type
        }
        response = make_request(payload)

        # 验证状态码不应为 500
        assert (
            response.status_code != 500
        ), "Should not return 500 for invalid parameter type"
        assert response.status_code == 400

        data = response.json()
        assert "InvalidParameter" in data.get(
            "code", ""
        ) or "InvalidParameter" in data.get("message", "")

    @pytest.mark.parametrize("top_p_value", [0, 0.0])
    def test_invalid_parameter_range_top_p(self, top_p_value):
        """
        Case: pre-siliconflow-deepseek-v3.1 top_p取值范围 (0, 1.0]。
        测试边界值 0，预期报错。
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"top_p": top_p_value},
        }
        response = make_request(payload)

        assert response.status_code == 400, f"top_p={top_p_value} should be invalid"
        data = response.json()
        assert "Range of top_p should be" in data.get("message", "")

    def test_invalid_parameter_range_temperature(self):
        """
        Case: pre-siliconflow-deepseek-v3.1 temperature 取值范围 [0, 2]。
        测试值 2.1，预期报错。
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"temperature": 2.1},
        }
        response = make_request(payload)

        assert response.status_code == 400
        data = response.json()
        assert "Temperature should be in" in data.get("message", "")


class TestDeepSeekR1Specifics:
    """
    针对 R1 模型的特定测试用例
    """

    def test_r1_usage_structure(self):
        """
        Case: .usage.output_tokens_details 该路径下不应该返回 text_tokens 字段。
        R1 模型推理侧可能没有 text_tokens。
        """
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {},
        }
        response = make_request(payload)
        assert response.status_code == 200

        # 检查流式返回的最后一帧 Usage
        frames = list(parse_sse_stream(response))
        assert len(frames) > 0
        final_usage = frames[-1].usage

        output_details = final_usage.get("output_tokens_details", {})
        # 验证 output_tokens_details 存在
        assert output_details, "output_tokens_details missing"
        # 验证不包含 text_tokens (根据表格描述这是预期行为)
        assert (
            "text_tokens" not in output_details
        ), "R1 usage should not contain text_tokens"
        # 验证包含 reasoning_tokens
        assert "reasoning_tokens" in output_details

    def test_r1_enable_thinking_parameter_error(self):
        """
        Case: r1传了 enable_thinking 报错。
        预期: 400 Value error, current model does not support parameter `enable_thinking`.
        """
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"enable_thinking": True},
        }
        response = make_request(payload)

        assert response.status_code == 400
        data = response.json()
        assert "does not support parameter" in data.get("message", "")


class TestAdvancedFeatures:
    """
    复杂场景：前缀续写、历史消息包含 ToolCall 等
    """

    def test_prefix_completion_thinking_conflict(self):
        """
        Case: 思考模式下(enable_thinking=true)，不支持前缀续写(partial=true)。
        预期返回: 400 InvalidParameter.
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

        assert response.status_code == 400
        data = response.json()
        assert "Partial mode is not supported when enable_thinking is true" in data.get(
            "message", ""
        )

    def test_history_with_tool_calls(self):
        """
        Case: 3.1和3.2 message中包含历史 tool_call 调用信息曾报 5xx。
        预期: 200 OK。
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.2",
            "input": {
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个为智能助手。请使用简洁、自然、适合朗读的中文回答",
                    },
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
            "parameters": {"enable_thinking": True},
        }
        response = make_request(payload)

        # 核心验证：不能崩 (500)
        assert (
            response.status_code != 500
        ), "Server returned 500 for history with tool calls"
        assert response.status_code == 200

    def test_r1_tool_call_format_wrapping(self):
        """
        Case: Error code: 400, _sse_http_status: 500 (Wrappping error).
        Input should be 'none', 'auto' or 'required'.
        验证 R1 对 result_format='message' 和 tool_choice 的组合处理。
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
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_current_weather"},
                },  # 修正后的 tool_choice 格式
                "tools": TOOL_VECTOR_WEATHER,
            },
        }

        # 注意：CSV中提到的错误是 `tool_choice` 格式问题导致的 400 被包了一层 500
        # 这里我们发送请求并检查状态码
        response = make_request(payload)

        # 即使失败，也应该返回标准的 400 而不是 InternalError
        if response.status_code != 200:
            error_data = response.json()
            # 确保不是 500 或者 InternalError
            assert response.status_code != 500
            assert error_data.get("code") != "InternalError"
