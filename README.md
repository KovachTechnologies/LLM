# LLM
Details related to LLM research, including setup, deployment, webapps, etc. associated with serving models.

# 1. Prerequisites

## 1.1 Install NVIDIA DriversUpdate your system:

Update

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install ubuntu-drivers-common
```

Add Nvidia repo
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
```

autoinstall
```bash
sudo ubuntu-drivers autoinstall
```

(didn't work).  Try 

```bash
udo apt purge nvidia*
sudo apt install nvidia-driver-575-open
```

Verify
```bash
nvidia-smi
```

## 1.2 install cuda toolkit

```bash
sudo apt install nvidia-cuda-toolkit
```

Verify

```bash
nvcc --version
nvidia-smi
```

## 1.3 Python
If python is not installed

```bash
sudo apt install python3.12 python3.122venv python3-pip
```

Install python modules.  Create requirements.txt

```
torch==2.9.0
transformers==4.57.1
huggingface_hub
accelerate>=0.26.0
numpy
streamlit
gradio
kernels
vllm
vllm[openai]
pdfplumber
docx
python-docx 
fastapi 
uvicorn 
httpx
```

run pip install

```bash
pip install -r requirements.txt
```

Verify

```bash
python3 -c "import torch; print(torch.cuda.is_available())" # should print True
```

Verify `kernels` module (needed for mxfp4)

```bash
python3 -c "
from transformers import GptOssConfig
cfg = GptOssConfig.from_pretrained('openai/gpt-oss-120b')
print('Quant config:', cfg.quantization_config)  # Should show 'mxfp4'
"
```

## 1.4 Install Hugging Face

Install astral-uv

```bash
sudo snap install astral-uv --classic
```

Install hugging face (`hf`) using uv

```bash
uv tool install "huggingface_hub" --force
```

# 2.0 Downloading Models

## 2.1 Download Models

Login with token

```bash
hf auth login
```

The prompts will ask you for the token

### 2.1.1 Tinyllama
```
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 --local-dir ~/models/TinyLlama-1.1B-Chat-v1.0
```

### 2.1.2 Llama 3.1b
```bash
hf download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir ~/models/llama-3.1-8b-instruct
```

### 2.1.3 Qwen2 72b
```bash
hf download Qwen/Qwen2-72B-Instruct --local-dir ~/models/qwen2-72b-instruct
```

### 2.1.4 Qwen3 Coder 30b
```bash
hf download Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 --local-dir ~/models/qwen3-coder-30b-instruct
```

### 2.1.5 Qwen3 30b
```bash
hf download Qwen/Qwen3-30B-A3B-Instruct-2507 --local-dir ~/models/qwen3-30b-instruct
```

### 2.1.6 GPT-OSS 20b
```bash
hf download openai/gpt-oss-20b --local-dir ~/models/openai-gptoss-20b
```

# 3 Serving Content

## 3.1 Running Models with vLLM

### 3.1.1 TinyLlama

Create a script to serve the model with vllm:

```bash
touch models/run_tinyllama.sh
chmod 777 models/run_tinyllama.sh
```

Paste the following content into `models/run_tinyllama.sh`.  
- Note that if we are running models concurrently, the port numbers must be distinct.
- Note that model name must match that of huggingface (in the `hf download` step)

```bash
export OMP_NUM_THREADS=10
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
python3 -m vllm.entrypoints.openai.api_server \
    --model /home/daniel/models/TinyLlama-1.1B-Chat-v1.0 \
    --trust-remote-code \
    --enforce-eager \
    --dtype auto \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --port 8000
```

Run the code

```bash
cd ~/models
./run_tinyllama.sh
```

Query with curl.  Note that the port number must match the above.

```bash
curl -X GET http://0.0.0.0:8000/v1/models
```

### 3.1.2 Llama 3.1b

Create a script to serve the model with vllm:

```bash
touch models/run_llama-3.1-8b.sh 
chmod 777 models/run_llama-3.1-8b.sh 
```

Paste the following content into `models/run_llama-3.1-8b.sh`.
- Note that if we are running models concurrently, the port numbers must be distinct.
- Note that model name must match that of huggingface (in the `hf download` step)

```bash
export OMP_NUM_THREADS=10
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
python3 -m vllm.entrypoints.openai.api_server \
    --model /home/daniel/models/llama-3.1-8b-instruct \
    --trust-remote-code \
    --enforce-eager \
    --dtype auto \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --port 8001
```

Run the code

```bash
cd ~/models
./run_llama-3.1-8b.sh
```

Query with curl.  Note that the port number must match the above.

```bash
curl -X GET http://0.0.0.0:8001/v1/models
```

### 3.1.3 Qwen2 72b
Create a script to serve the model with vllm:

```bash
touch models/run_qwen2-72b.sh 
chmod 777 models/run_qwen2-72b.sh 
```

Paste the following content into `models/run_qwen2-72b.sh`.
- Note that if we are running models concurrently, the port numbers must be distinct.
- Note that model name must match that of huggingface (in the `hf download` step)

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1      # averts 'max size' error
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2-72B-Instruct \
    --dtype auto \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --port 8002
```

Run the code

```bash
cd ~/models
./run_qwen2-72b.sh
```

Query with curl.  Note that the port number must match the above.

```bash
curl -X GET http://0.0.0.0:8002/v1/models
```

### 3.1.4 Qwen3 Coder 30b
Create a script to serve the model with vllm:

```bash
touch models/run_qwen3-coder-30b-instruct.sh 
chmod 777 models/run_qwen3-coder-30b-instruct.sh 
```

Paste the following content into `models/run_qwen3-30b-instruct.sh`.
- Note that if we are running models concurrently, the port numbers must be distinct.
- Note that model name must match that of huggingface (in the `hf download` step)

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1      # averts 'max size' error
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --dtype auto \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --port 8003
```

Run the code

```bash
cd ~/models
./run_qwen3-coder-30b-instruct.sh
```

Query with curl.  Note that the port number must match the above.

```bash
curl -X GET http://0.0.0.0:8003/v1/models
```

### 3.1.5 Qwen3 30b
Create a script to serve the model with vllm:

```bash
touch models/run_qwen3-30b-instruct.sh 
chmod 777 models/run_qwen3-30b-instruct.sh 
```

Paste the following content into `models/run_qwen3-30b-instruct.sh`.
- Note that if we are running models concurrently, the port numbers must be distinct.
- Note that model name must match that of huggingface (in the `hf download` step)

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1      # averts 'max size' error
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --dtype auto \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --port 8004
```

Run the code

```bash
cd ~/models
./run_qwen3-30b-instruct.sh
```

Query with curl.  Note that the port number must match the above.

```bash
curl -X GET http://0.0.0.0:8004/v1/models
```

### 2.1.6 GPT-OSS 20b
```bash
hf download openai/gpt-oss-20b ~/models/openai-gptoss-20b
```

Create a script to serve the model with vllm:

```bash
touch models/run_openai-gptoss-20b.sh 
chmod 777 models/run_openai-gptoss-20b.sh 
```

Paste the following content into `models/run_openai-gptoss-20b.sh`.
- Note that if we are running models concurrently, the port numbers must be distinct.
- Note that model name must match that of huggingface (in the `hf download` step)

```bash
export OMP_NUM_THREADS=10
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1      # averts 'max size' error
python3 -m vllm.entrypoints.openai.api_server \
    --model /home/daniel/models/openai-gptoss-20b \
    --trust-remote-code \
    --enforce-eager \
    --dtype auto \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --port 8005
```

Run the code

```bash
cd ~/models
./run_openai-gptoss-20b.sh
```

Query with curl.  Note that the port number must match the above.

```bash
curl -X GET http://0.0.0.0:8005/v1/models
```

# Web Application

## Write Web Applications

See web applications in the `webapps` directory.
