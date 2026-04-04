# Introduction

The following are scripts used to run opencode on Linux.  The command line

# VLLM Start Script

``` bash
export VLLM_DISABLE_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=16

vllm serve qwen3-coder-30b-fp8 --enable-expert-parallel --enforce-eager --dtype auto --tensor-parallel-size 2 --gpu-memory-utilization 0.90 --cpu-offload-gb 8 --port 8000 --enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768 --max-num-batched-tokens 8192 --disable-custom-all-reduce --enforce-eager
```

# Open Code Config

Place in `~/.config/opencode/opencode.json`

``` bash
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vLLM (local)",
      "options": {
        "baseURL": "http://192.168.0.187:8000/v1"
      },
      "models": {
        "qwen2.5-7b-instruct": {
          "name": "Qwen2.5"
        },
        "TinyLlama-1.1B-Chat-v1.0": {
          "name": "TinyLlama",
          "options": {
            "max_tokens": 2048,
            "max_completion_tokens": 2048
          }
        },
        "qwen3-coder-30b-fp8": {
          "name": "qwen3-coder-30b-fp8",
          "options": {
            "max_tokens": 8192,
            "max_completion_tokens": 8192,
            "temperature": 0.7,
            "top_p": 0.95
          }
        }
      }
    }
  },
  "model": "qwen3-coder-30b-fp8",
  "small_model": "qwen3-coder-30b-fp8",
  "compaction": {
    "auto": true,
    "prune": true,
    "reserved": 8192,
    "threshold": 0.80
  }
}
```
