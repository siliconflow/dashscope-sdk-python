curl -X POST "https://api.siliconflow.cn/v1/chat/completions" \
  -H "Authorization: Bearer $SILICONFLOW_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-V3",
    "stream": true,
    "messages": [
      {
        "role": "user",
        "content": "今天天气好吗？"
      },
      {
        "role": "assistant",
        "content": "今天天气不错，要出去玩玩嘛？"
      },
      {
        "role": "user",
        "content": "那你有什么地方推荐？"
      },
      {
        "role": "assistant",
        "content": "我建议你去公园，春天来了，花朵开了，很美丽。"
      },
      {
        "role": "user",
        "content": "给我推荐3个杭州的景点"
      }
    ],
    "top_p": 0.8,
    "top_k": -1,
    "repetition_penalty": 0.8,
    "max_tokens": 512
  }'
