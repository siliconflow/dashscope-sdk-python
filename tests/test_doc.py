import pytest
import requests
import os
import json
from typing import Dict, Any

# --- CONFIGURATION ---
BASE_URL_PREFIX = "http://localhost:8000/siliconflow/models"
# BASE_URL_PREFIX = "https://api-bailian.siliconflow.cn/siliconflow/models"
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
ERR_MSG_TOOL_CHOICE = "<400> InternalError.Algo.InvalidParameter: Input should be a valid string: parameters.tool_choice.str & Field required: parameters.tool_choice.ToolChoice.function"

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

    def test_r1_force_tool_choice_with_complex_tools(self):
        """
        Test derived from curl command:
        Scenario: DeepSeek-R1 with multiple tools and explicit tool_choice object.
        Expected: Should fail validation because tool_choice must be 'none' or 'auto' (string),
        not a dictionary object, consistent with ERR_MSG_TOOL_CHOICE.
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
                "tool_choice": {"type": "get_current_weather"},
                "tools": [
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
                                    "unit": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                    },
                                },
                                "required": ["location"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_time",
                            "description": "Get the current time in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "unit": {"type": "string", "enum": []},
                                },
                                "required": ["location"],
                            },
                        },
                    },
                ],
            },
            "resources": [],
        }

        response = make_request(payload)

        # 验证是否返回了预期的参数校验错误
        assert_exact_error(response, "InvalidParameter", ERR_MSG_TOOL_CHOICE)

    def test_tool_call_index_field_presence(self):
        """
        Verify that tool_calls objects in the response contain the 'index' field.
        This addresses the regression where 'index': 0 was missing.
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.1",
            "input": {
                "messages": [
                    {"role": "system", "content": "You are a translation helper."},
                    {
                        "role": "user",
                        "content": "Please translate 'Hello' to Spanish using the tool.",
                    },
                ]
            },
            "parameters": {
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "translationTool",
                            "description": "Translate text",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "translation": {
                                        "description": "The translated text",
                                        "type": "string",
                                    }
                                },
                                "required": ["translation"],
                            },
                        },
                    }
                ],
                "tool_choice": "required",
                # Important: Use non-streaming to check the final full object structure easily,
                # or stream and reconstruct. The curl example used non-streaming (default).
                # We will check the non-streaming response body.
                "incremental_output": False,
            },
        }

        # Force non-streaming request to inspect the final message object directly
        response = make_request(payload, stream=False)
        assert response.status_code == 200, f"Request failed: {response.text}"

        data = response.json()

        # Navigate to the tool_calls
        # Note: The response structure depends on the gateway.
        # Based on the user's curl output: root -> choices -> [0] -> message -> tool_calls -> [0]
        # Based on SiliconFlow/DashScope typically: output -> choices -> ...

        # We will handle the standard DashScope structure which puts result in 'output'
        # or the OpenAI compatible structure. The user's example shows OpenAI format ("choices" at root).
        # Our helper 'make_request' calls the SiliconFlow endpoint.

        # If the API returns standard OpenAI format:
        if "choices" in data:
            choices = data["choices"]
        elif "output" in data and "choices" in data["output"]:
            choices = data["output"]["choices"]
        else:
            pytest.fail(f"Unexpected response structure: {data.keys()}")

        assert len(choices) > 0, "No choices returned"
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        assert len(tool_calls) > 0, "Expected tool calls but got none"

        first_tool_call = tool_calls[0]

        # THE ASSERTION
        assert (
            "index" in first_tool_call
        ), f"Missing 'index' field in tool_call object: {first_tool_call}"
        assert (
            first_tool_call["index"] == 0
        ), f"Expected index 0, got {first_tool_call.get('index')}"

    def test_dashscope_stream_tool_calls_type_presence(self):
        """
        验证 DashScope 流式协议（incremental_output=True）下，
        tool_calls 返回的每一包数据（不仅是第一包）都包含 'type' 字段。
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.1",
            "input": {
                "messages": [
                    {"role": "user", "content": "What is the weather like in Boston?"}
                ]
            },
            "parameters": {
                "incremental_output": True,
                "result_format": "message",
                "tool_choice": "auto",
                "tools": [
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
                                    "unit": {
                                        "type": "string",
                                        "enum": ["celsius", "fahrenheit"],
                                    },
                                },
                                "required": ["location"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "get_current_time",
                            "description": "Get the current time in a given location",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state, e.g. San Francisco, CA",
                                    },
                                    "unit": {"type": "string", "enum": []},
                                },
                                "required": ["location"],
                            },
                        },
                    },
                ],
            },
        }

        response = make_request(payload, stream=True)
        assert response.status_code == 200, f"Request failed: {response.text}"

        packet_count_with_tool_calls = 0

        for line in response.iter_lines():
            if not line:
                continue

            line_text = line.decode("utf-8")
            if not line_text.startswith("data:"):
                continue

            json_str = line_text[5:].strip()
            if json_str == "[DONE]":
                break

            try:
                chunk = json.loads(json_str)

                # 适配 DashScope 结构: output -> choices -> message -> tool_calls
                # 注意：流式返回中，DashScope 结构通常在 output 下
                output = chunk.get("output", {})
                choices = output.get("choices", [])

                if not choices:
                    continue

                message = choices[0].get("message", {})

                # 检查是否存在 tool_calls
                if "tool_calls" in message:
                    tool_calls = message["tool_calls"]

                    # 遍历该包中的所有 tool_call
                    for tc in tool_calls:
                        packet_count_with_tool_calls += 1

                        # --- 核心断言 ---
                        # 验证每一包 tool_call 都必须包含 'type' 字段
                        assert "type" in tc, (
                            f"Missing 'type' field in tool_call packet.\n"
                            f"Full Chunk: {json_str}"
                        )
                        assert tc["type"] == "function", (
                            f"Invalid 'type' value: {tc.get('type')}\n"
                            f"Full Chunk: {json_str}"
                        )

                        # 顺便验证 index 字段 (通常也需要)
                        assert (
                            "index" in tc
                        ), f"Missing 'index' field in tool_call packet: {json_str}"

            except json.JSONDecodeError:
                continue

        # 确保确实触发了工具调用，否则测试没有实际意义
        assert (
            packet_count_with_tool_calls > 0
        ), "No tool calls were returned in the stream, test inconclusive."

    def test_dash_sync_sql_tool_call_index_presence(self):
        """
        Scenario based on the provided curl command:
        - Model: pre-siliconflow/deepseek-v3.1
        - Mode: DashScope Synchronous (Non-streaming)
        - Context: SQL Tool scenario
        - Requirement: output.choices[0].message.tool_calls[0] must contain 'index' field.
        """
        payload = {
            "model": "pre-siliconflow/deepseek-v3.1",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": "Which sales agent made the most in sales in 2009?",
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "function": {
                                    "arguments": "{}",
                                    "name": "sql_db_list_tables",
                                },
                                "id": "tool_abcd123",
                                "type": "function",
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "Album, Artist, Customer, Employee, Genre, Invoice, InvoiceLine, MediaType, Playlist, PlaylistTrack, Track",
                        "tool_call_id": "tool_abcd123",
                    },
                ]
            },
            "parameters": {
                "incremental_output": False,  # Key: Synchronous request
                "n": 1,
                "result_format": "message",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "sql_db_schema",
                            "description": "Input to this tool is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first! Example Input: table1, table2, table3",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "table_names": {
                                        "description": "A comma-separated list of the table names for which to return the schema. Example input: 'table1, table2, table3'",
                                        "type": "string",
                                    }
                                },
                                "required": ["table_names"],
                            },
                        },
                    }
                ],
            },
        }

        # 1. 发送非流式请求
        response = make_request(payload, stream=False)
        assert response.status_code == 200, f"Request failed: {response.text}"

        # 2. 解析响应
        data = response.json()

        # 兼容不同的 Response 根节点结构 (Direct 'choices' or 'output.choices')
        if "output" in data and "choices" in data["output"]:
            choices = data["output"]["choices"]
        elif "choices" in data:
            choices = data["choices"]
        else:
            pytest.fail(f"Unexpected response structure: {data.keys()}")

        assert len(choices) > 0, "No choices returned"

        # 3. 定位 tool_calls
        message = choices[0].get("message", {})
        tool_calls = message.get("tool_calls", [])

        assert (
            len(tool_calls) > 0
        ), "Expected model to generate a tool_call (sql_db_schema) but got none."

        # 4. 核心断言：检查 index 字段是否存在
        first_tool_call = tool_calls[0]
        print(f"\n[DEBUG] Tool Call Received: {json.dumps(first_tool_call, indent=2)}")

        assert "index" in first_tool_call, (
            f"❌ Critical Failure: 'index' field missing in tool_calls[0] for Dash Synchronous request.\n"
            f"Received: {first_tool_call}"
        )

        # 验证 index 值通常为 0 (对于单工具调用)
        assert (
            first_tool_call["index"] == 0
        ), f"Expected index 0, got {first_tool_call.get('index')}"

    def test_r1_conflict_partial_and_thinking_strict(self):
        """
        Scenario: R1 Model provided with both partial (prefix) and enable_thinking.
        Expectation: Strict 400 Error.
        """
        payload = {
            "model": "pre-siliconflow/deepseek-r1",
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

    def test_v3_base_ignore_conflict_success(self):
        """
        Scenario: V3 (Base) Model provided with both partial and enable_thinking.
        Expectation: Return 200 OK (Compatibility mode: ignores thinking).
        """
        # 注意：这里特指 deepseek-v3，不包含 .1 或 .2
        payload = {
            "model": "pre-siliconflow/deepseek-v3",
            "input": {
                "messages": [
                    {"role": "user", "content": "你好"},
                    {"role": "assistant", "partial": True, "content": "你好，我是"},
                ]
            },
            "parameters": {"enable_thinking": True},
        }

        response = make_request(payload)
        assert (
            response.status_code == 200
        ), f"V3 Base should handle partial+thinking gracefully. Got: {response.text}"

    def test_v3_2_conflict_strict_error(self):
        """
        Scenario: V3.2 Model provided with both partial and enable_thinking.
        Expectation: Strict 400 Error (Unlike V3 Base).
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
            response, "InvalidParameter", ERR_MSG_PARTIAL_THINKING_CONFLICT
        )

    # --- 在 TestFunctionalFixes 类中添加 ---

    def test_v3_ignore_conflict_partial_and_thinking_success(self):
        """
        Scenario: V3.2 Model provided with both partial (prefix) and enable_thinking.
        Expectation: Return 200 OK.
        The system should ignore the conflict (or ignore the thinking param) and
        process the request normally instead of throwing a 400 error.
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

        # 验证 V3 必须成功返回 200，而不是 400
        assert (
            response.status_code == 200
        ), f"V3 should handle partial+thinking gracefully. Got: {response.text}"

        # 额外验证：确保返回的内容没有包含报错信息的 JSON 文本
        try:
            # 如果是流式，检查第一行是否不是报错
            if response.encoding is None:
                response.encoding = "utf-8"

            first_line = next(response.iter_lines()).decode("utf-8")
            # 简单的非报错检查 (正常流式通常以 data: 开头，或者是 {"output":...} 如果是非流式)
            assert "InvalidParameter" not in first_line
        except StopIteration:
            pass  # 空响应在 status_code 检查时已捕获
