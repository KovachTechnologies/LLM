# Installation

``` bash
curl -fsSL https://ollama.com/install.sh | sh
```

# Running Models

``` bash
ollama pull qwen3-coder:30b-a3b-q4_K_M
```

``` bash
ollama run qwen3-coder:30b-a3b-q4_K_M
```

``` bash
ollama serve
```

# Model Details

``` bash
ollama show qwen3-coder:30b-a3b-q4_K_M
```

# Server

By default, the ollama server runs on `http://localhost:11434`.  To verify it is running:

``` bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:11434     # should return 200
```
