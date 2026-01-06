curl --location "http://localhost:8000/api/v1/services/aigc/text-generation/generation" \
--header "x-api-key: $DASHSCOPE_API_KEY" \
--header "Content-Type: application/json" \
--header "X-DashScope-SSE: enable" \
--data '{
    "model": "deepseek-r1",
    "input":{
        "messages":[
            {
                "role": "user",
                "content": "你是谁？"
            }
        ]
    },
    "parameters": {
        "result_format": "message",
        "incremental_output":true,
        "max_tokens": 1024
    }
}'
