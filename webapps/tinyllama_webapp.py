# tinyllama_webapp.py
import os
import json
import asyncio
from typing import List

import pdfplumber
import docx
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
VLLM_URL   = "http://localhost:8000/v1"
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"   # <-- from curl /v1/models
PORT       = 5001

# ----------------------------------------------------------------------
# FastAPI
# ----------------------------------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static", exist_ok=True)

# ----------------------------------------------------------------------
# Helper: extract text
# ----------------------------------------------------------------------
def extract_text(file: UploadFile) -> str:
    filename = file.filename.lower()
    content = asyncio.get_event_loop().run_in_executor(None, file.file.read)
    content = asyncio.get_event_loop().run_until_complete(content)

    if filename.endswith(".pdf"):
        with pdfplumber.open(file.file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif filename.endswith((".docx", ".doc")):
        doc = docx.Document(file.file)
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

# ----------------------------------------------------------------------
# Chat request model
# ----------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 512
    temperature: float = 0.7

# ----------------------------------------------------------------------
# Manual SSE streamer (uses httpx)
# ----------------------------------------------------------------------
async def stream_chat(req: ChatRequest):
    payload = {
        "model": MODEL_NAME,
        "messages": [m.dict() for m in req.messages],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{VLLM_URL}/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break
                data = json.loads(line[6:])
                delta = data.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(stream_chat(req), media_type="text/event-stream")

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    history: str = Form("[]"),
):
    prev = json.loads(history) if history else []

    text = await asyncio.get_event_loop().run_in_executor(None, extract_text, file)
    if len(text) > 30_000:
        text = text[:30_000] + "\n[ … truncated … ]"

    system_msg = {
        "role": "system",
        "content": "You are an assistant that has just received the following document. "
                   "Answer any user questions based on it.\n\n---\n" + text + "\n---"
    }

    # Insert system message at the very beginning
    new_history = [system_msg] + [m for m in prev if m["role"] != "system"]
    return {"history": new_history}

# ----------------------------------------------------------------------
# HTML UI (unchanged)
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>GPT-OSS-120B Chat + Docs</title>
<style>
  body{font-family:Arial,Helvetica,sans-serif;background:#f4f4f9;margin:0;padding:0}
  #app{max-width:960px;margin:2rem auto;background:#fff;padding:1.5rem;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,.1)}
  #chat{height:60vh;overflow-y:auto;border:1px solid #ddd;border-radius:4px;padding:0.5rem;margin-bottom:1rem}
  .msg{margin:0.5rem 0;padding:0.75rem 1rem;border-radius:8px;max-width:85%;word-wrap:break-word}
  .user{background:#007bff;color:#fff;margin-left:auto}
  .assistant{background:#e9ecef}
  .system{background:#fff3cd;color:#856404;font-style:italic}
  #input-area{display:flex;gap:0.5rem;align-items:center}
  #user-input{flex:1;padding:0.75rem;border:1px solid #ced4da;border-radius:4px;font-size:1rem}
  button{padding:0.75rem 1.2rem;background:#28a745;color:#fff;border:none;border-radius:4px;cursor:pointer}
  button:disabled{background:#6c757d;cursor:not-allowed}
  #file-input{display:none}
  label[for=file-input]{cursor:pointer;background:#0d6efd;color:#fff;padding:0.5rem 1rem;border-radius:4px;font-size:0.9rem}
</style>
</head>
<body>
<div id="app">
  <h2>GPT-OSS-120B (vLLM) Chat + Document Upload</h2>
  <div id="chat"></div>

  <div id="input-area">
    <input id="user-input" placeholder="Type a message…" autocomplete="off">
    <button id="send-btn">Send</button>
    <label for="file-input">Upload PDF/DOCX</label>
    <input type="file" id="file-input" accept=".pdf,.docx,.doc">
  </div>
</div>

<script>
const chatDiv = document.getElementById('chat');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const fileInput = document.getElementById('file-input');

let conversation = [];

function addMessage(role, text) {
  const div = document.createElement('div');
  div.className = `msg ${role}`;
  div.textContent = text;
  chatDiv.appendChild(div);
  div.scrollIntoView({behavior: 'smooth'});
  return div;
}

// ---------- SEND ----------
async function sendMessage() {
  const txt = input.value.trim();
  if (!txt) return;
  addMessage('user', txt);
  conversation.push({role:'user', content:txt});
  input.value = '';
  sendBtn.disabled = true;

  const assistantDiv = addMessage('assistant', '');

  const resp = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({messages: conversation, max_tokens: 512, temperature: 0.7})
  });

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let out = '';
  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    out += decoder.decode(value);
    assistantDiv.textContent = out;
  }
  conversation.push({role:'assistant', content: out});
  sendBtn.disabled = false;
}

// ---------- UPLOAD ----------
fileInput.addEventListener('change', async () => {
  const file = fileInput.files[0];
  if (!file) return;
  const form = new FormData();
  form.append('file', file);
  form.append('history', JSON.stringify(conversation));

  const up = await fetch('/api/upload', {method:'POST', body:form});
  const data = await up.json();
  conversation = data.history;

  const sys = conversation.find(m => m.role === 'system');
  if (sys) addMessage('system', `Document "${file.name}" uploaded (${sys.content.length} chars)`);
  fileInput.value = '';
});

sendBtn.onclick = sendMessage;
input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } });
</script>
</body>
</html>
"""

# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
# uvicorn tinyllama_webapp:app --host 0.0.0.0 --port 5001
