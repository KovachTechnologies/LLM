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

### 2.1.1 Tinyllama
```
mkdir ~/models/tinyllama-1.1b
cd ~/models/tinyllama-1.1b
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/config.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.json
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/tokenizer.model
wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/model.safetensors
```

### 2.1.2 Llama 3.1b
```bash
hf download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir ~/models/llama-3.1-8b-instruct
```

### 2.1.3 Qwen2
(todo)

### 2.1.4 Qwen 20b
(todo)

### 2.1.5 GPT-OSS
(todo)

# 3 Serving Content

## 3.1 Running Models with vLLM

### 3.1.1 TinyLlama

Create a script to serve the model with vllm:

```bash
touch models/run_tinyllama.sh
chmod 777 models/run_tinyllama.sh
```

Paste the following content into `models/run_tinyllama.sh`.  Note that if we are running models concurrently, the port numbers must be distinct.

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1      # averts 'max size' error
python3 -m vllm.entrypoints.openai.api_server \
    --model tinyllama-1.1b \
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

Paste the following content into `models/run_llama-3.1-8b.sh`.  Note that if we are running models concurrently, the port numbers must be distinct.

```bash
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1      # averts 'max size' error
python3 -m vllm.entrypoints.openai.api_server \
    --model llama-3.1-8b-instruct  \
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



### 3.1.3 Qwen2
(todo)

### 3.1.4 Qwen 20b
(todo)

### 3.1.5 GPT-OSS
(todo)

# Web Application

## Write Web Applications

```python
```

