curl -X POST http://localhost:11434/api/chat \
  -d '{
    "model": "Qwen3.5",
    "messages": [
      {"role": "user", "content": "How many letter r are in strawberry?"}
    ],
    "think": false
  }'
