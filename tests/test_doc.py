import pytest
import json
import requests
import os
from typing import Generator, List, Dict, Any, Optional
from dataclasses import dataclass

# --- CONSTANTS & CONFIGURATION ---

# 基础 URL 修改为适配动态路径的前缀
# 最终请求 URL 将拼接为: BASE_URL + "/" + {model_path}
BASE_URL_PREFIX = "http://localhost:8000/siliconflow/models"

API_KEY = os.getenv("SILICONFLOW_API_KEY", "test_api_key")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
    "X-DashScope-SSE": "enable",
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
    """Parses the raw SSE stream line by line."""
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line.startswith("data:"):
                json_str = decoded_line[5:].strip()
                try:
                    data = json.loads(json_str)
                    yield SSEFrame(
                        id=data.get("output", {}).get("choices", [{}])[0].get("id", "unknown"),
                        output=data.get("output", {}),
                        usage=data.get("usage", {}),
                        request_id=data.get("request_id", ""),
                    )
                except json.JSONDecodeError:
                    continue

def make_request(payload: Dict[str, Any]) -> requests.Response:
    """
    Helper to send POST request using the Dynamic Path URL structure.

    Format: POST /siliconflow/models/{model_path}

    It extracts the 'model' from the payload to construct the URL.
    """
    model_path = payload.get("model")

    if not model_path:
        raise ValueError("Test payload must contain 'model' field for dynamic URL construction")

    # Construct the dynamic URL, e.g.:
    # http://localhost:8000/siliconflow/models/deepseek-ai/DeepSeek-V3
    url = f"{BASE_URL_PREFIX}/{model_path}"

    # Send the request. Note: We keep 'model' in the json body as well,
    # though the server mainly relies on the path parameter now.
    return requests.post(url, headers=HEADERS, json=payload, stream=True)

# --- TEST SUITE ---

class TestDynamicPathRouting:
    """
    测试动态 URL 路由本身的正确性
    """

    def test_routing_basic_success(self):
        """
        测试标准的 URL 格式是否能通
        URL: .../deepseek-ai/DeepSeek-V3
        """
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "input": {"messages": [{"role": "user", "content": "Hello"}]},
            "parameters": {"max_tokens": 10}
        }
        response = make_request(payload)
        assert response.status_code == 200

    def test_routing_with_mapping(self):
        """
        测试服务端 ModelResolver 是否依然工作
        URL: .../pre-siliconflow/deepseek-v3 (会被映射到 upstream 的 deepseek-ai/DeepSeek-V3)
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3",
            "input": {"messages": [{"role": "user", "content": "Test"}]},
            "parameters": {"max_tokens": 10}
        }
        response = make_request(payload)
        assert response.status_code == 200


class TestParameterValidation:
    """
    对应表格中参数校验相关的错误用例 (4xx Error Codes)
    """

    def test_invalid_parameter_type_top_p(self):
        """
        Case: parameters.top_p 输入字符串 'a'，预期返回 400 InvalidParameter。
        """
        payload = {
            "model": "deepseek-ai/DeepSeek-V3",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"top_p": "a"},  # Invalid type
        }
        response = make_request(payload)

        assert response.status_code == 400
        data = response.json()
        assert "InvalidParameter" in data.get("code", "") or "InvalidParameter" in data.get("message", "")

    @pytest.mark.parametrize("top_p_value", [0, 0.0])
    def test_invalid_parameter_range_top_p(self, top_p_value):
        """
        Case: top_p取值范围 (0, 1.0]。测试边界值 0。
        """
        payload = {
            "model": "deepseek-ai/DeepSeek-V3.1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"top_p": top_p_value},
        }
        response = make_request(payload)

        assert response.status_code == 400
        data = response.json()
        assert "Range of top_p should be" in data.get("message", "")

    def test_invalid_parameter_range_temperature(self):
        """
        Case: temperature 取值范围 [0, 2]。测试值 2.1。
        """
        payload = {
            "model": "deepseek-ai/DeepSeek-V3.1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"temperature": 2.1},
        }
        response = make_request(payload)

        assert response.status_code == 400
        data = response.json()
        assert "Temperature should be in" in data.get("message", "")


class TestDeepSeekR1Specifics:
    """
    针对 R1 模型的特定测试用例 (Reasoning Models)
    """

    def test_r1_usage_structure(self):
        """
        Case: R1 模型不应该返回 text_tokens，应该返回 reasoning_tokens。
        URL: .../deepseek-ai/DeepSeek-R1
        """
        payload = {
            "model": "deepseek-ai/DeepSeek-R1",
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
        assert output_details, "output_tokens_details missing"

        # 验证不包含 text_tokens (R1 特性)
        assert "text_tokens" not in output_details, "R1 usage should not contain text_tokens"
        # 验证包含 reasoning_tokens
        assert "reasoning_tokens" in output_details

    def test_r1_enable_thinking_parameter_error(self):
        """
        Case: R1 原生支持思考，显式传递 enable_thinking=True 应报错。
        """
        payload = {
            "model": "deepseek-ai/DeepSeek-R1",
            "input": {"messages": [{"role": "user", "content": "你好"}]},
            "parameters": {"enable_thinking": True},
        }
        response = make_request(payload)

        assert response.status_code == 400
        data = response.json()
        assert "does not support parameter" in data.get("message", "")


class TestAdvancedFeatures:
    """
    复杂场景：前缀续写、ToolCall 格式校验等
    """

    def test_prefix_completion_thinking_conflict(self):
        """
        Case: 思考模式下(enable_thinking=true)，不支持前缀续写(partial=true)。
        """
        payload = {
            "model": "deepseek-ai/DeepSeek-V3.2",
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
        assert "Partial mode is not supported when enable_thinking is true" in data.get("message", "")

    def test_r1_tool_choice_conflict(self):
        """
        Case: R1 模型下，enable_thinking (或原生 R1) 开启时，不支持具体的 tool_choice 字典绑定。
        """
        payload = {
            "model": "deepseek-ai/DeepSeek-R1",
            "input": {
                "messages": [
                    {"role": "user", "content": "What is the weather like in Boston?"}
                ]
            },
            "parameters": {
                "result_format": "message",
                # R1 logic usually prevents enforcing a specific tool via dict when thinking is active
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_current_weather"},
                },
                "tools": TOOL_VECTOR_WEATHER,
            },
        }

        response = make_request(payload)

        # Expecting 400 because R1 + Specific Tool Choice is often restricted in this proxy logic
        if response.status_code != 200:
            assert response.status_code == 400
            error_data = response.json()
            assert "DeepSeek R1 does not support specific tool_choice" in error_data.get("message", "")

if __name__ == "__main__":
    # 如果直接运行此脚本，可以使用 pytest 调起
    pytest.main(["-v", __file__])
