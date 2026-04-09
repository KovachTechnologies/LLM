#!/bin/bash
export VLLM_DISABLE_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=16

# Stability tweaks tuned for your dual 5060 Ti rig
export VLLM_RPC_TIMEOUT=600
export NCCL_P2P_DISABLE=1
export VLLM_HOST_IP=192.168.0.187

vllm serve Qwen2.5-Coder-14B-Instruct --enforce-eager --dtype auto --tensor-parallel-size 2 --gpu-memory-utilization 0.85 --cpu-offload-gb 8 --port 8000 --max-model-len 32768 --max-num-batched-tokens 16384 --disable-custom-all-reduce --enable-auto-tool-choice --tool-call-parser hermes --chat-template-content-format string
