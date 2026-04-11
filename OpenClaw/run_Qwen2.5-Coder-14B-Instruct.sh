#!/bin/bash
export VLLM_DISABLE_USAGE_STATS=1 export DO_NOT_TRACK=1
export OMP_NUM_THREADS=16

# Stability tweaks for dual 5060 Ti (Blackwell sm_120)
export VLLM_RPC_TIMEOUT=600
export NCCL_P2P_DISABLE=1
export VLLM_HOST_IP=192.168.0.187

# Critical Blackwell / FlashInfer fixes
export VLLM_ATTENTION_BACKEND=FLASH_ATTN          # Force FlashAttention instead of FlashInfer
export FLASHINFER_DISABLE=1                       # Completely skip FlashInfer JIT
export VLLM_DISABLE_FLASHINFER_PREFILL=1          # Extra safety

# Optimized for Qwen2.5-Coder-14B AWQ on 2x16GB
vllm serve Qwen/Qwen2.5-Coder-14B-Instruct-AWQ --quantization awq --enforce-eager --dtype float16 --tensor-parallel-size 2 --gpu-memory-utilization 0.92 --cpu-offload-gb 4 --max-model-len 32768 --max-num-batched-tokens 16384 --disable-custom-all-reduce --enable-auto-tool-choice --tool-call-parser hermes --port 8000
