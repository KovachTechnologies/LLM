# OPENCODE.md - Running Qwen3-Coder-30B-FP8 Locally on Dual RTX 5060 Ti

## Introduction

This document captures the journey of getting **OpenCode** running smoothly with a powerful local model (`qwen3-coder-30b-fp8`, a Qwen3 MoE variant) on a Linux rig. The setup uses **vLLM** as the inference server for OpenAI-compatible API endpoints and **OpenCode** as the frontend/agent interface.

Key highlights:
- Full tensor-parallel inference across **two RTX 5060 Ti 16GB GPUs**
- Nearly maxed out VRAM (~15.3 GB used per card out of 16.3 GB) — pretty badass for a 30B-class MoE model in FP8
- Long context support (up to 65k tokens)
- Hermes toolchain for reliable tool calling (this was the fix for repeated query failures)
- Auto-compaction and pruning to keep sessions manageable

## System Details

**Hardware Rig (as of April 2026):**
- **GPUs**: 2× NVIDIA GeForce RTX 5060 Ti 16GB (Ada Lovelace architecture, excellent FP8 performance)
- **Driver**: NVIDIA 580.126.09
- **CUDA**: 13.0
- **CPU**: (from previous context — assuming high-core count with OMP_NUM_THREADS=16 tuned for your system)
- **RAM**: Sufficient for CPU offload (we allocate 8 GB)
- **Network**: Local access via `http://192.168.0.187:8000/v1`

**NVIDIA-SMI Snapshot (model fully loaded and running):**

``` bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.126.09 Driver Version: 580.126.09 CUDA Version: 13.0                    |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name           Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap    | Memory-Usage           | GPU-Util  Compute M. |
|                                   |                        | MIG M.               |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060 Ti   Off | 00000000:01:00.0  On |                  N/A |
|   0%   46C    P8     6W / 180W    | 15312MiB / 16311MiB    |      0%  Default     |
|                                   |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA GeForce RTX 5060 Ti   Off | 00000000:08:00.0 Off |                  N/A |
|   0%   44C    P8     9W / 180W    | 15304MiB / 16311MiB    |      0%  Default     |
|                                   |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

Processes:
GPU   GI  CI  PID   Type   Process name          GPU Memory
      ID  ID             Usage
0     N/A N/A 121815  C    VLLM::Worker_TP0_EP0   15272MiB
1     N/A N/A 121816  C    VLLM::Worker_TP1_EP1   15272MiB
```

We are pushing these dual 16 GB cards right to the edge with --gpu-memory-utilization 0.92 and expert/tensor parallelism. The FP8 quantization + MoE architecture makes this feasible without excessive swapping.

## vLLM Start Script

Save this as a convenient script (e.g., start-vllm-qwen.sh) and make it executable.

``` bash
#!/bin/bash
export VLLM_DISABLE_USAGE_STATS=1
export DO_NOT_TRACK=1
export OMP_NUM_THREADS=16

vllm serve qwen3-coder-30b-fp8 \
  --enable-expert-parallel \
  --enforce-eager \
  --dtype auto \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --cpu-offload-gb 8 \
  --port 8000 \
  --max-model-len 65536 \
  --max-num-batched-tokens 32768 \
  --disable-custom-all-reduce \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

### Explanation of Key Command-Line Parameters

- `--enable-expert-parallel` — Critical for this MoE model (Qwen3-30B-A3B). It uses expert parallelism for the MoE layers instead of pure tensor parallelism, improving efficiency on multi-GPU setups.
- `--enforce-eager` — Disables torch.compile / CUDA graphs. Faster startup and more stable on newer/edge hardware; slightly slower inference but worth it for reliability.
- `--dtype` auto — Lets vLLM automatically select the best precision (FP8 weights are used here).
- `--tensor-parallel-size` 2 — Splits the model across both GPUs (TP=2). Combined with expert-parallel, this is how we fit and run the full model.
- `--gpu-memory-utilization` 0.92 — Aggressively uses 92% of each GPU's VRAM. This is what lets us max out the dual 5060 Tis (~15.3 GB used per card).
- `--cpu-offload-gb` 8 — Offloads 8 GB of model weights/KV cache to system RAM when VRAM pressure spikes. Helps stability at high utilization.
- `--max-model-len` 65536 — Supports up to 64k context length (great for large codebases).
- `--max-num-batched-tokens` 32768 — Limits tokens per batch for memory control and throughput.
- `--disable-custom-all-reduce` — Disables vLLM's custom all-reduce kernel (can help stability or compatibility on certain driver/GPU combos).
- `--enable-auto-tool-choice` + `--tool-call-parser hermes` — Enables automatic tool calling and uses the Hermes parser format. This pairs with the Hermes toolchain for reliable function calling in OpenCode.

## OpenCode Configuration

Place this in `~/.config/opencode/opencode.json` (or a project-specific opencode.json).

``` json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "vLLM (local)",
      "options": {
        "baseURL": "http://192.168.0.187:8000/v1",
        "timeout": 900000,
        "chunkTimeout": 300000
      },
      "models": {
        "qwen3-coder-30b-fp8": {
          "name": "qwen3-coder-30b-fp8",
          "options": {
            "max_tokens": 32768,
            "max_completion_tokens": 32768
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
    "reserved": 12288,
    "threshold": 0.88
  }
}
```

### Explanation of Key Parameters in opencode.json

- `provider.vllm` — Points OpenCode to your local vLLM server using the OpenAI-compatible endpoint.
- `baseURL` — Your machine's IP and vLLM port.
- `model / small_model` — Both set to the same model (no need for a tiny fallback on this rig).
- `compaction`
 - `auto: true` — Automatically compacts the session when context fills up.
 - `prune: true` — Removes old tool outputs to save tokens.
 - `reserved: 12288` — Keeps a buffer of ~12k tokens during compaction to avoid overflow.
 - `threshold: 0.88` — Triggers compaction at 88% of context window (custom-tuned; default is often ~0.75–0.9).

This setup keeps long coding sessions manageable without losing too much history.

## Hermes Toolchain

The Hermes toolchain (from Nous Research) provides robust tool-calling support and was the missing piece that resolved repeated query failures.

- Installation:
``` bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

After installation, the `--tool-call-parser hermes` flag in the vLLM script + `--enable-auto-tool-choice` makes tool use reliable in OpenCode.
