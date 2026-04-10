curl http://192.168.0.187:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ",
    "messages": [{"role": "user", "content": "Write a Python function to calculate Fibonacci and also list the files in the current directory using tools if available."}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "list_directory",
          "description": "List files in a directory",
          "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
          }
        }
      }
    ],
    "tool_choice": "auto"
  }'
