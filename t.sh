curl --location "http://localhost:8000/siliconflow/models/deepseek-ai/DeepSeek-V3" \
--header "x-api-key: $SILICONFLOW_API_KEY" \
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
        "max_tokens": "100.1"
    }
}'
