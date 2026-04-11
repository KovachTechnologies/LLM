## Install

``` bash
curl -fsSL https://openclaw.ai/install.sh | bash
npm install -g openclaw@latest
```

## Run qwen3 coder via ollama

``` bash
ollama pull qwen3-coder:30b-a3b-q4_K_M
```

``` bash
ollama run qwen3-coder:30b-a3b-q4_K_M
```


``` bash
OLLAMA_HOST=0.0.0.0 ollama serve qwen3-coder:30b-a3b-q4_K_M
```

## Run Onboarding to Set Up

``` bash
openclaw onboard --install-daemon
```

Valdiate Gateways, and get addresses for browser.  Should look like

``` bash
│  Web UI: http://127.0.0.1:18789/                             │
│  Web UI (with token): http://127.0.0.1:18789/#token=default  │
│  Gateway WS: ws://127.0.0.1:18789                            │
│  Gateway: not detected (timeout)                             │
│  Docs: https://docs.openclaw.ai/web/control-ui  
``` 

## Manually Setting the Config

Location: `~/.openclaw/openclaw.json`

Set parameters, use open claw doctor to fix errors:

``` bash
openclaw doctor --fix
```

## OpenClaw Gateway

``` bash
openclaw gateway run
openclaw gateway stop
openclaw gateway restart
```

## Error Logs

``` bash
openclaw logs --follow
```


