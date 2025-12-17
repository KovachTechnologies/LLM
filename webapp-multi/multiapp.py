import subprocess
import time
import os
from io import BytesIO
from typing import Dict, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI  # For client-side API calls to vLLM
import pycurl

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # Create a 'templates' dir
app.mount("/static", StaticFiles(directory="static"), name="static")  # For CSS/JS if needed

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Model configs: Adjust paths, ports, and vLLM args
# Model configs: Adjust paths, ports, and vLLM args
tinyllama_path = "/home/daniel/models/TinyLlama-1.1B-Chat-v1.0"
llama31b_path = "/home/daniel/models/llama-3.1-8b-instruct"
gptoss20b_path = "/home/daniel/models/openai-gptoss-20b"

TINYLLAMA = "tinyllama"
LLAMA31B = "llama3-1b"
GPTOSS20B = "gptoss-20b"

MODELS = {
    GPTOSS20B: {
        "path": gptoss20b_path,
        "port": 8005,
        "script" : "/home/daniel/models/run_openai-gptoss-20b.sh",
        "vllm_args": ["--model", gptoss20b_path, "--port", "8005", "--quantization", "awq", "--tensor-parallel-size", "2"]  # Large, both GPUs, quantized
    },
    LLAMA31B: {
        "path": llama31b_path,
        "port": 8001,
        "script" : "/home/daniel/models/run_llama-3.1-8b.sh",
        "vllm_args": ["--model", llama31b_path, "--port", "8001", "--dtype", "auto", "--tensor-parallel-size", "1", "--max-model-len", "32768"]
    },
    TINYLLAMA: {
        "path": tinyllama_path,
        "port": 8000,
        "script" : "/home/daniel/models/run_tinyllama.sh",
        "vllm_args": ["--model", tinyllama_path, "--port", "8000", "--dtype", "auto", "--tensor-parallel-size", "1", "--max-model-len", "32768"]  # Small, one GPU
    }
}


current_model = None
vllm_process: subprocess.Popen = None

class ChatRequest(BaseModel):
    model: str
    messages: List[Dict[str, str]]

def do_curl( port ) :
    print( f'curl -X GET http://0.0.0.0:{port}/v1/models' )
    buffer = BytesIO()
    c = pycurl.Curl()

    c.setopt(c.URL, f'http://0.0.0.0:{port}/v1/models')
    c.setopt(c.WRITEDATA, buffer)
    #c.setopt(c.CAINFO, certifi.where())
    c.perform()
    c.close()
    body = buffer.getvalue()
    print(body.decode('iso-8859-1'))
    return True

def start_vllm(model_name: str):
    global vllm_process, current_model, vllm_pid
    if current_model == model_name and vllm_process and vllm_process.poll() is None:
        return  # Already running

    # Kill existing if any
    if vllm_process:
        print( f"--> Killing process {vllm_pid}" )
        vllm_process.terminate()
        vllm_process.wait()
        time.sleep(1)
        kill_string = f"nvidia-smi --query-compute-apps=pid | while read line; do if [ $line != 'pid' ]; then kill -9 $line; fi; done"
        print( kill_string )
        os.system( kill_string )
        time.sleep(1)


    config = MODELS.get(model_name)
    port = config[ "port" ]
    if not config:
        raise ValueError("Invalid model")

    # Start vLLM server
    #cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server"] + config["vllm_args"]
    cmd = ["bash", config["script"]]
    print( "--> Starting process" )
    print( " ".join( cmd ) )
    vllm_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    vllm_pid = vllm_process.pid
    print( f"--> pid: {vllm_pid}" )
    current_model = model_name

    # Wait for server to start (poll health)
    start_time = time.time()
    while time.time() - start_time < 120:  # Timeout 2min
        try:
            models_loaded = do_curl( port )
            return
        except:
            time.sleep(5)
    raise RuntimeError("vLLM failed to start")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "models": list(MODELS.keys())})

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        start_vllm(req.model)
        config = MODELS[req.model]
        client = OpenAI(base_url=f"http://localhost:{config['port']}/v1", api_key="fake")

        def generate():
            stream = client.chat.completions.create(
                model=config["path"],  # vLLM uses model path as ID
                messages=req.messages,
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
# uvicorn multiapp:app --host 0.0.0.0 --port 5001
