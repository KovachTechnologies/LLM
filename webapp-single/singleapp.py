import json
import os
import pycurl

from io import BytesIO
from typing import List, Dict
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from openai import OpenAI

# Fixed configuration for the already-running GPTOSS 20B
MODEL_NAME = "GPTOSS 20B"
MODEL_PATH = "openai-gptoss-20b"  # Used only as the model ID for vLLM
VLLM_PORT = 8000
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1"

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
    data = json.loads( body.decode('iso-8859-1') )
    return data[ "data" ][ 0 ][ "id" ]

MODEL_NAME = do_curl( VLLM_PORT )
MODEL_PATH = do_curl( VLLM_PORT )


app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Create a persistent client (reused across requests)
client = OpenAI(base_url=VLLM_BASE_URL, api_key="fake")

class ChatRequest(BaseModel):
    messages: List[Dict[str, str]]

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Pass a simple title or model name to the template
    return templates.TemplateResponse(
        "index_single.html",
        {"request": request, "model_name": MODEL_NAME}
    )

@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        def generate():
            stream = client.chat.completions.create(
                model=MODEL_PATH,  # vLLM expects the model path as the model identifier
                messages=req.messages,
                stream=True,
                temperature=0.7,
                max_tokens=2048,
                # Add any default params you like here
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return StreamingResponse(generate(), media_type="text/event-stream")
    except Exception as e:
        return {"error": str(e)}

# Run with: uvicorn singleapp:app --host 0.0.0.0 --port 5001
