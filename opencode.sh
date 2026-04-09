#!/bin/bash
export VLLM_DISABLE_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=16

# New additions to reduce timeouts and improve stability on dual-GPU
export VLLM_RPC_TIMEOUT=300          # Increase from default 60s
export NCCL_P2P_DISABLE=1            # Often helps with shm_broadcast hangs on consumer GPUs
# export VLLM_USE_V1=0               # Optional: fall back to older engine if V1 is causing issues (test later)

vllm serve qwen3-coder-30b-fp8 --enable-expert-parallel --enforce-eager --dtype auto --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --cpu-offload-gb 8 --port 8000 --max-model-len 65536 --max-num-batched-tokens 32768 --disable-custom-all-reduce --enable-auto-tool-choice --tool-call-parser hermes --chat-template-content-format string  
