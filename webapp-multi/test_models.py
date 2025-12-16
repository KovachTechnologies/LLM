import json
import os
import pycurl
import subprocess
import sys
import time

from io import BytesIO

os.environ["OMP_NUM_THREADS"] = "10"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Model configs: Adjust paths, ports, and vLLM args
tinyllama_path = "/home/daniel/models/TinyLlama-1.1B-Chat-v1.0"
llama31b_path = "/home/daniel/models/llama-3.1-8b-instruct"
gptoss20b_path = "/home/daniel/models/openai-gptoss-20b"

TINYLLAMA = "tinyllama"
LLAMA31B = "llama3-1b"
GPTOSS20B = "gptoss-20b"

MODELS = {
    TINYLLAMA: {
        "path": tinyllama_path,
        "port": 8000,
        "script" : "/home/daniel/models/run_tinyllama.sh",
        "vllm_args": ["--model", tinyllama_path, "--port", "8000", "--dtype", "auto", "--tensor-parallel-size", "1", "--max-model-len", "32768"]  # Small, one GPU
    },
    LLAMA31B: {
        "path": llama31b_path,
        "port": 8001,
        "script" : "/home/daniel/models/run_llama-3.1-8b.sh",
        "vllm_args": ["--model", llama31b_path, "--port", "8001", "--dtype", "auto", "--tensor-parallel-size", "1", "--max-model-len", "32768"]
    },
    GPTOSS20B: {
        "path": gptoss20b_path,
        "port": 8002,
        "script" : "/home/daniel/models/run_openai-gptoss-20b.sh",
        "vllm_args": ["--model", gptoss20b_path, "--port", "8002", "--quantization", "awq", "--tensor-parallel-size", "2"]  # Large, both GPUs, quantized
    }
}

def finish( p ) :
    print( "--> Model loaded successfully" )
    p.kill()
    sys.exit()

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



if __name__ == "__main__" :

    import argparse
    parser = argparse.ArgumentParser(description="Test loading models via python")
    parser.add_argument('-1', '--tinyllama', action='store_true', help="Launch tinyllama")
    parser.add_argument('-2', '--llama31b', action='store_true', help="Launch llama 3.1b")
    parser.add_argument('-3', '--gptoss20b', action='store_true', help="Launch gptoss 20b")
    args = parser.parse_args()

    selection = TINYLLAMA
    if args.tinyllama :
        selection = TINYLLAMA
    if args.llama31b :
        selection = LLAMA31B
    if args.gptoss20b :
        selection = GPTOSS20B

    config = MODELS.get(selection)
    port = config[ "port" ]
    #cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server"] + config["vllm_args"]
    cmd = ["bash", config["script"]]
    print( " ".join( cmd ) )
    vllm_process = subprocess.Popen(cmd)
    print( "--> vllm process" )
    start_time = time.time()
    print( "--> time" )
    time_index = 0
    time_delta = 5
    print( "--> Starting process" )
    models_loaded = False
    while time.time() - start_time < 120:  # Timeout 2min
        time_index += time_delta
        try:
            models_loaded = do_curl( port )
            break
        except:
            time.sleep(time_delta)
            print( f"--> Still not loaded: {time_index} seconds" )
    
    if not models_loaded :
        raise RuntimeError("vLLM failed to start")
