# Squawk AI

A web interface for interacting with local LLMs powered by Ollama. This application allows you to chat with your local models through a clean, responsive web interface and upload documents for context-aware responses.

## Features

- **Local AI Chat**: Communicate with your locally running Ollama models
- **File Upload Support**: Upload TXT, PDF, and Word documents for context-aware conversations
- **Automatic Model Detection**: Automatically detects and uses your currently running Ollama model
- **Responsive UI**: Clean, modern interface with Tailwind CSS styling
- **Real-time Streaming**: Responses are streamed in real-time for a smooth chat experience
- **Copy Functionality**: Easily copy assistant responses with one click

## Requirements

- Python 3.7+
- Ollama (https://ollama.com/)
- Local LLM model(s) running via Ollama

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd webapp-ollama
```

2. Create a virtual environment and install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install fastapi uvicorn ollama python-multipart jinja2 pypdf2 python-docx
```

## Running the Application

1. Ensure Ollama is running with a model:
```bash
ollama run <your-model-name>
```

2. Start the web application:
```bash
python app.py
```

3. Open your browser to `http://localhost:5000`

## Usage

1. The interface automatically detects your running Ollama model
2. Type messages in the input field at the bottom
3. Upload files using the paperclip icon for context-aware responses
4. Responses are streamed in real-time
5. Click the copy button to copy assistant responses

## Project Structure

```
.
├── app.py              # Main FastAPI application
├── templates/          # HTML templates
├── static/             # Static assets (favicon, etc.)
├── install.sh          # Installation script
├── README.md           # This file
└── LICENSE             # MIT License
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.