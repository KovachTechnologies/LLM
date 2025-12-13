# llama31_webapp.py
import os
import json
import asyncio
from typing import List

import pdfplumber
from docx import Document  # Correct import from python-docx
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx

# ----------------------------------------------------------------------
# CONFIG - Tailored for your dual RTX 5060 Ti rig
# ----------------------------------------------------------------------
VLLM_BASE_URL = "http://localhost:8001/v1"  # Your Llama 3.1 8B on port 8001
PORT = 5001

# ----------------------------------------------------------------------
# FastAPI
# ----------------------------------------------------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
os.makedirs("static", exist_ok=True)

# ----------------------------------------------------------------------
# Helper: extract text from uploaded file
# ----------------------------------------------------------------------
async def extract_text(file: UploadFile) -> str:
    content = await file.read()
    filename = file.filename.lower()

    if filename.endswith(".pdf"):
        with pdfplumber.open(content) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif filename.endswith((".docx", ".doc")):
        doc = Document(content)
        return "\n".join(p.text for p in doc.paragraphs)
    elif filename.endswith(".txt"):
        return content.decode("utf-8")
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type: only PDF, DOCX, DOC, TXT")

# ----------------------------------------------------------------------
# Chat models
# ----------------------------------------------------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: int = 1024
    temperature: float = 0.7

# ----------------------------------------------------------------------
# Stream responses from vLLM
# ----------------------------------------------------------------------
async def stream_chat(req: ChatRequest):
    payload = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # Optional, but helps if multiple models
        "messages": [m.dict() for m in req.messages],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "stream": True,
    }

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{VLLM_BASE_URL}/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    delta = data["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta

# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(stream_chat(req), media_type="text/event-stream")

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), history: str = Form("[]")):
    prev_history = json.loads(history) if history else []

    text = await extract_text(file)
    if len(text) > 40_000:  # Safe limit for 32k context (leave room for chat)
        text = text[:40_000] + "\n\n[Document truncated for length]"

    system_msg = {
        "role": "system",
        "content": "You are a helpful assistant. Use the following document content to answer questions accurately.\n\n---\n" + text + "\n---"
    }

    # Replace or add system message at the start
    new_history = [system_msg] + [m for m in prev_history if m["role"] != "system"]
    return {"history": new_history}

# ----------------------------------------------------------------------
# Simple HTML UI
# ----------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Llama 3.1 8B Chat (Dual 5060 Ti Rig)</title>
    <style>
        body {font-family: Arial, sans-serif; background: #f0f2f6; margin: 0; padding: 0;}
        #app {max-width: 960px; margin: 2rem auto; background: #fff; padding: 1.5rem; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);}
        #chat {height: 70vh; overflow-y: auto; border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background: #fafafa;}
        .msg {margin: 0.75rem 0; padding: 1rem; border-radius: 12px; max-width: 80%; word-wrap: break-word;}
        .user {background: #007bff; color: white; margin-left: auto;}
        .assistant {background: #e9ecef;}
        .system {background: #fff3cd; color: #856404; font-style: italic; font-size: 0.9em;}
        #input-area {display: flex; gap: 0.5rem;}
        #user-input {flex: 1; padding: 1rem; border: 1px solid #ced4da; border-radius: 8px; font-size: 1rem;}
        button {padding: 1rem 1.5rem; background: #28a745; color: white; border: none; border-radius: 8px; cursor: pointer;}
        button:disabled {background: #6c757d;}
        label[for=file-input] {cursor: pointer; background: #0d6efd; color: white; padding: 1rem; border-radius: 8px;}
        #file-input {display: none;}
    </style>
</head>
<body>
<div id="app">
    <h2>Llama 3.1 8B Instruct Chat (vLLM on Dual RTX 5060 Ti)</h2>
    <div id="chat"></div>
    <div id="input-area">
        <input id="user-input" placeholder="Type your message..." autocomplete="off">
        <button id="send-btn">Send</button>
        <label for="file-input">Upload PDF/DOCX/TXT</label>
        <input type="file" id="file-input" accept=".pdf,.docx,.doc,.txt">
    </div>
</div>

<script>
let conversation = [];

const chatDiv = document.getElementById('chat');
const input = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const fileInput = document.getElementById('file-input');

function addMessage(role, text) {
    const div = document.createElement('div');
    div.className = `msg ${role}`;
    div.textContent = text;
    chatDiv.appendChild(div);
    div.scrollIntoView({behavior: 'smooth'});
    return div;
}

async function sendMessage() {
    const txt = input.value.trim();
    if (!txt) return;
    addMessage('user', txt);
    conversation.push({role: 'user', content: txt});
    input.value = '';
    sendBtn.disabled = true;

    const assistantDiv = addMessage('assistant', '');
    let fullResponse = '';

    const resp = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({messages: conversation, max_tokens: 1024, temperature: 0.7})
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
        const {value, done} = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        fullResponse += chunk;
        assistantDiv.textContent = fullResponse;
    }
    conversation.push({role: 'assistant', content: fullResponse});
    sendBtn.disabled = false;
}

fileInput.addEventListener('change', async () => {
    const file = fileInput.files[0];
    if (!file) return;

    const form = new FormData();
    form.append('file', file);
    form.append('history', JSON.stringify(conversation));

    const up = await fetch('/api/upload', {method: 'POST', body: form});
    const data = await up.json();
    conversation = data.history;

    const sysMsg = conversation.find(m => m.role === 'system');
    if (sysMsg) {
        addMessage('system', `Uploaded "${file.name}" (${sysMsg.content.length.toLocaleString()} chars extracted)`);
    }
    fileInput.value = '';
});

sendBtn.onclick = sendMessage;
input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});
</script>
</body>
</html>
"""

# ----------------------------------------------------------------------
# Run command (on your rig)
# ----------------------------------------------------------------------
# uvicorn llama31_webapp:app --host 0.0.0.0 --port 5001 --workers 1
