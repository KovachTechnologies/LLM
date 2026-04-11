import io
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import ollama
import json
from PyPDF2 import PdfReader
from docx import Document
import os

app = FastAPI(title="Squawk AI")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


def extract_text_from_file(file: UploadFile) -> str:
    content = file.file.read()
    filename = file.filename.lower()
    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif filename.endswith((".doc", ".docx")):
        doc = Document(io.BytesIO(content))
        return "\n".join(para.text for para in doc.paragraphs)
    elif filename.endswith(".txt"):
        return content.decode("utf-8", errors="ignore")
    return "[Unsupported file type]"


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        request, "index.html", context={"request": request}
    )


@app.get("/current_model")
async def get_current_model():
    try:
        models = ollama.list()
        if models and models.get("models"):
            model_name = models["models"][0]["model"]
            return {"model": model_name}
        return {"model": "No model loaded — run 'ollama run <model>' first"}
    except Exception as e:
        return {"model": f"Ollama error: {str(e)}"}


@app.post("/chat")
async def chat(messages: str = Form(...), file: UploadFile = File(None)):
    messages_list = json.loads(messages)

    # Handle file upload and inject content
    if file and file.size > 0:
        text = extract_text_from_file(file)
        if text.strip():
            messages_list.append(
                {
                    "role": "user",
                    "content": f"File content from '{file.filename}':\n\n{text}\n\nPlease use this information to answer or continue the conversation.",
                }
            )

    async def generate():
        try:
            # Automatically get the currently running model
            models_info = ollama.list()
            if not models_info or not models_info.get("models"):
                yield "\n\n[Error: No Ollama model is currently running. Please run 'ollama run yourmodel' in terminal first.]"
                return

            current_model = models_info["models"][0]["model"]
            print(f"DEBUG: Using model {current_model}")  # Debug line

            stream = ollama.chat(
                model=current_model,  # This fixes the 'name' error
                messages=messages_list,
                stream=True,
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as e:
            print(f"DEBUG: Ollama error occurred: {str(e)}")  # Debug line
            yield f"\n\n[Ollama error: {str(e)}]"

    return StreamingResponse(generate(), media_type="text/plain")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
