# OpenCode Integration Guide

This guide explains how to integrate libqllm with the OpenCode AI coding assistant, allowing you to use local LLM models for code generation and assistance.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Supported Models](#supported-models)
- [Testing the Setup](#testing-the-setup)
- [Performance Considerations](#performance-considerations)
- [Architecture Details](#architecture-details)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Testing Results](#testing-results)
- [Contributing](#contributing)
- [References](#references)
- [Current Status](#current-status)
  - [What's Working](#whats-working)
  - [Recommended Models](#recommended-models)
  - [Context Size Issue](#context-size-issue)
  - [Code Fix Applied](#code-fix-applied)
- [Build Notes](#build-notes)

## Overview

`qllmd` provides an OpenAI-compatible HTTP API directly, enabling OpenCode to use local GGUF models for inference. This setup gives you complete control over your AI coding assistant without relying on external API services or running a separate proxy.

## Architecture

```
OpenCode CLI
    ↓ (HTTP/OpenAI API)
qllmd daemon (port 4242)
    ↓ (llama.cpp)
Local GGUF Model
```

## Prerequisites

1. **Built libqllm** with all binaries:
   ```bash
   make
   ```

2. **Downloaded GGUF model** (recommended for OpenCode: Qwen2.5 Coder 3B or another instruct/coder GGUF):
   ```bash
   bin/qllm-list
   bin/qllm-path '*qwen2.5-coder*.gguf'
   ```

3. **OpenCode CLI** installed:
   ```bash
   npm install -g opencode
   # or follow instructions at https://opencode.ai
   ```

## Quick Start

### 1. Start qllmd

```bash
cd /path/to/libqllm
MODEL="$(bin/qllm-path '*qwen2.5-coder*.gguf')"
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH bin/qllmd -d -p 4242 "$MODEL"
```

This starts the daemon on port 4242. The same port serves the line-based qllmd TCP protocol and the OpenAI-compatible HTTP API (`/v1/...`). The `-d` flag runs it in daemon mode (background).

**Note**: The daemon will take 10-30 seconds to load the model and initialize the shared context.

### 2. Configure OpenCode

Create or edit `opencode.json` in your project directory (or globally in `~/.config/opencode/`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "libqllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "libqllm (local)",
      "options": {
        "baseURL": "http://127.0.0.1:4242/v1",
        "apiKey": "dummy"
      },
      "models": {
        "qwen2.5-coder": {
          "name": "qwen2.5-coder-3b-instruct-q4_k_m",
          "limit": {
            "context": 4096,
            "output": 1024
          }
        }
      }
    }
  }
}
```

**Configuration notes**:
- `baseURL`: Must point to the qllmd HTTP API (default: `http://127.0.0.1:4242/v1`)
- `apiKey`: Can be any value (e.g., "dummy") since local authentication is not enforced
- `limit.context`: Should not exceed the context passed to qllmd with `-c`
- `limit.output`: Keep this comfortably below the context size so OpenCode has room for prompts and tool results

### 3. Use OpenCode with libqllm

```bash
opencode run -m libqllm/qwen2.5-coder "Write a hello world function in Python"
```

Or interactively:
```bash
opencode chat -m libqllm/qwen2.5-coder
```

## Supported Models

You can use any GGUF model with qllmd. The daemon automatically detects chat templates for:

- **Phi-3** models (e.g., Phi-3-mini-4k-instruct)
- **Mistral** models (e.g., Mistral-7B-Instruct)
- **Gemma** models (e.g., gemma-2-9b-it)
- **Generic** models (fallback template)

To use a different model, update the `opencode.json` configuration:

```json
{
  "models": {
    "mistral-7b": {
      "name": "Mistral-7B-Instruct (local)",
      "limit": {
        "context": 8192,
        "output": 4096
      }
    }
  }
}
```

Then restart qllmd with the new model:
```bash
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH bin/qllmd -d -p 4242 Mistral-7B-Instruct-v0.3.Q8_0.gguf
```

## Testing the Setup

### Test the daemon directly

```bash
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH bin/qllm-chat -p 4242
```

Type messages and verify you get responses.

### Test the OpenAI-compatible HTTP API

```bash
curl http://127.0.0.1:4242/health
curl http://127.0.0.1:4242/v1/models
curl -X POST http://127.0.0.1:4242/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder",
    "messages": [{"role": "user", "content": "Say hello"}],
    "temperature": 0.7,
    "max_tokens": 50
  }'
```

You should receive an OpenAI-formatted JSON response.

### Test OpenCode integration

```bash
cd /tmp/test_project
cp /path/to/libqllm/opencode.json .
opencode run -m libqllm/qwen2.5-coder "Write a simple test"
```

## Performance Considerations

### CPU Inference

- **Expected behavior**: Generation is slow on CPU (1-3 tokens/sec for 3.8B models)
- **Recommended**: Use quantized models (Q4_K_M, Q5_K_M, Q8_0) for better performance
- **Model size vs speed**: Smaller models (3B-7B) are more practical for CPU inference

### GPU Acceleration

libqllm uses llama.cpp's Vulkan backend for GPU acceleration:

- **Automatic**: Vulkan is used automatically for compute buffers if available
- **Full GPU offload**: Not yet supported in qllmd (future enhancement)
- **Current limitation**: Context still lives on CPU, only compute uses GPU

### Memory Usage

- **Model memory**: Approximately model file size + 20%
- **Context memory**: (n_ctx × n_layers × 2 × sizeof(fp16)) ≈ 768 MiB for 2048 ctx
- **Total example**: Phi-3-mini Q8_0 uses ~4.5 GiB RAM

## Architecture Details

### Shared Context Design

qllmd uses a **shared context architecture**:

- **Single context** created at daemon startup
- **Reused** across all client connections
- **Thread-safe**: Mutex-protected access
- **Trade-off**: Conversations are not isolated between connections

This design was necessary to work around Vulkan backend limitations in llama.cpp.

### Protocol Details

**qllmd TCP protocol** (port 4242):
```
Client → Server: chat\n
Client → Server: ask <message>\n
Server → Client: <response>\x04
```

**New: messages command** (recommended for multi-turn):
```
Client → Server: chat\n
Client → Server: messages [{"role":"user","content":"Hello"}]\n
Server → Client: <response>\x04
```

**Model info command**:
```
Client → Server: info\n
Server → Client: {"model":"qwen2.5-coder-3b-instruct-q4_k_m.gguf","template":"chatml"}
```

**qllmd OpenAI-compatible HTTP API** (port 4242):
- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `POST /v1/completions` - OpenAI-compatible text completions
- `GET /v1/models` - List available models (dynamically detects loaded model)
- `GET /health` - Health check endpoint

## Known Limitations

### 1. Vulkan Backend Limitation

The Vulkan backend in llama.cpp cannot handle multiple context instances reliably. Workarounds implemented:

- **Skip batch_free**: `llama_batch_free()` is not called (small memory leak)
- **Shared context**: Single context reused across connections
- **No conversation isolation**: Different clients share the same context state

### 2. Performance

- **CPU inference is slow**: 1-3 tokens/sec for 3.8B models
- **No GPU offload**: Context and KV cache are CPU-only
- **Single-threaded generation**: One request processed at a time

### 3. Configuration

- **Fixed context size**: Must be set at daemon startup (default: 2048)
- **No dynamic model loading**: Requires daemon restart to change models
- **No model caching**: Each daemon instance loads one model only

## Troubleshooting

### Daemon fails to start

**Symptom**: `qllmd` exits immediately or crashes

**Solutions**:
1. Ensure `crb.txt` file exists in working directory:
   ```bash
   touch crb.txt
   ```

2. Check if model file path is correct:
   ```bash
   bin/qllm-list
   bin/qllm-path '*qwen2.5-coder*.gguf'
   ```

3. Verify LD_LIBRARY_PATH is set:
   ```bash
   export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
   ```

### API returns 500 errors

**Symptom**: qllmd returns HTTP 500 from `/v1/chat/completions` or `/v1/completions`

**Solutions**:
1. Check if qllmd is running:
   ```bash
   ps aux | grep qllmd
   ```

2. Test daemon connectivity:
   ```bash
   LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH bin/qllm-chat -p 4242
   ```

3. Check daemon logs (if running in foreground)

### OpenCode can't connect

**Symptom**: OpenCode shows connection errors

**Solutions**:
1. Verify qllmd is running and serving HTTP:
   ```bash
   curl http://127.0.0.1:4242/health
   curl http://127.0.0.1:4242/v1/models
   ```

2. Check `opencode.json` uses `http://127.0.0.1:4242/v1`

3. Ensure `@ai-sdk/openai-compatible` npm package will be installed by OpenCode

### Slow generation

**Expected behavior**: CPU inference is inherently slow

**Improvements**:
- Use smaller models (Phi-3-mini instead of Llama-3-70B)
- Use lower quantization (Q4_K_M instead of Q8_0)
- Reduce context size: `bin/qllmd -c 1024 ...`
- Wait for GPU offload support (future)

### Model hangs or crashes

**Symptom**: Daemon stops responding during generation

**Solutions**:
1. Check system memory availability:
   ```bash
   free -h
   ```

2. Reduce context size to fit in RAM:
   ```bash
   bin/qllmd -c 512 -d -p 4242 model.gguf
   ```

3. Use more aggressive quantization (Q4_K_M)

## Advanced Usage

### Custom System Prompts

Start qllmd with a system prompt:

```bash
bin/qllmd -d -p 4242 -S "You are a helpful coding assistant." model.gguf
```

### Adjusting Sampling Parameters

Edit `src/libqllm.c` to modify defaults:
- `temperature` (default: 0.7)
- `top_p` (default: 0.9)
- `top_k` (default: 40)
- `repeat_penalty` (default: 1.1)

Rebuild after changes:
```bash
make
```

### Running Multiple Models

Run multiple daemon instances on different ports:

```bash
# Terminal 1: Qwen Coder on port 4242
bin/qllmd -d -p 4242 qwen2.5-coder-3b-instruct-q4_k_m.gguf

# Terminal 2: Mistral on port 4243
bin/qllmd -d -p 4243 Mistral-7B-Instruct-v0.3.Q8_0.gguf
```

Update `opencode.json` with multiple providers:
```json
{
  "provider": {
    "libqllm-qwen": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://127.0.0.1:4242/v1", "apiKey": "dummy"},
      "models": {"qwen2.5-coder": {"name": "Qwen2.5 Coder", "limit": {"context": 4096, "output": 1024}}}
    },
    "libqllm-mistral": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://127.0.0.1:4243/v1", "apiKey": "dummy"},
      "models": {"mistral": {"name": "Mistral-7B", "limit": {"context": 8192, "output": 4096}}}
    }
  }
}
```

### Startup Script

Create a script to launch qllmd:

```bash
#!/bin/bash
# start-libqllm.sh

cd /path/to/libqllm

# Start qllmd
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
MODEL="$(bin/qllm-path '*qwen2.5-coder*.gguf')"
bin/qllmd -d -p 4242 "$MODEL"

# Wait for daemon to initialize
sleep 15

echo "libqllm services started"
echo "qllmd: localhost:4242"
echo "OpenAI API: http://localhost:4242/v1"
```

## Testing Results

### Verified Working

1. **qllmd daemon** - Loads model and handles requests correctly
2. **info command** - Returns model info:
   ```
   $ echo "info" | nc localhost 4242
   {"model":"qwen2.5-coder-3b-instruct-q4_k_m.gguf","template":"chatml"}
   ```

3. **messages command** - Multi-turn conversation:
   ```
   $ python3 -c "
   import socket,json
   s=socket.socket()
   s.connect(('127.0.0.1',4242))
   s.sendall(b'chat\n')
   s.sendall(b'messages [{\"role\":\"user\",\"content\":\"My name is Alice\"},{\"role\":\"assistant\",\"content\":\"Hello Alice!\"},{\"role\":\"user\",\"content\":\"What is my name?\"}]\n')
   print(s.recv(4096).decode())
   "
   Your name is Alice. It was nice to meet you!
   ```

4. **qllmd OpenAI-compatible HTTP API**:
   - `GET /v1/models` - Returns detected model
   - `GET /health` - Returns health status
   - `POST /v1/chat/completions` - Works with full conversation history

### Example API Response

```bash
$ curl -X POST http://127.0.0.1:4242/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-coder",
    "messages": [
      {"role": "user", "content": "Write hello world in Python"}
    ]
  }'

{
  "choices": [{
    "message": {
      "content": "def hello_world():\n    print(\"Hello, World!\")\n\nhello_world()"
    }
  }]
}
```

### Performance

- Simple query: ~15-30 seconds
- Multi-turn: ~20-40 seconds
- Code generation: ~30-60 seconds

(Note: Exact timing depends on model size, quantization, and backend.)

## Contributing

Found issues or have improvements? Please contribute:

1. **Bug reports**: Open an issue with reproduction steps
2. **Performance improvements**: Submit PRs with benchmarks
3. **Model support**: Test with different GGUF models and report compatibility
4. **Documentation**: Improve this guide based on your experience

## References

- **libqllm GitHub**: [Your repo URL]
- **OpenCode Documentation**: https://opencode.ai/docs
- **llama.cpp**: https://github.com/ggerganov/llama.cpp
- **GGUF models**: https://huggingface.co/models?library=gguf

## Current Status

**TL;DR: Streaming and non-streaming requests work directly through qllmd. Use a coder/instruct model with enough context for OpenCode prompts and tool results.**

### What's Working

- ✅ Streaming chat completions (SSE format)
- ✅ Non-streaming chat completions
- ✅ Text completions for fill-in-the-middle style autocomplete
- ✅ OpenAI-compatible API (/v1/models, /health, /v1/chat/completions, /v1/completions)
- ✅ curl and standard HTTP clients
- ✅ OpenCode connects and sends requests

### Recommended Models

| Model | Context | Status |
|-------|---------|--------|
| Qwen2.5-Coder-3B-Instruct | 32K train / use 4K+ | ✅ Recommended |
| Mistral-7B-Instruct | 8K | ✅ Recommended |
| Gemma-2-9b-it | 8K | ✅ Recommended |
| Phi-3.5-mini-instruct | 6K | ⚠️ May work |
| Phi-3-mini-4k-instruct | 4K | ❌ Too small |

### Context Size Issue

OpenCode sends ~10KB system prompt + 11 tool definitions. Even with aggressive truncation:
- 4K models: qllm_prime fails → no response
- 6K models: May work with truncated tools
- 8K models: Works out of the box

### Code Fix Applied

1. **Direct HTTP handlers** in `src/qllmd.c` for `/health`, `/v1/models`, `/v1/chat/completions`, and `/v1/completions`
2. **JSON escaping** in `src/qllmd.c`
3. **Streaming usage ordering** compatible with OpenAI-style SSE clients

See [Build Notes](#build-notes) for rebuilding.

## Build Notes

After code changes, rebuild and restart:
```bash
make

# Restart qllmd with 8K context model (recommended)
pkill -f qllmd
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH bin/qllmd -d -p 4242 Mistral-7B-Instruct-v0.3.Q8_0.gguf
```

### Verify Streaming Works

```bash
curl -N http://127.0.0.1:4242/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "mistral", "messages": [{"role": "user", "content": "hi"}], "stream": true}'
```

### Test with OpenCode

```bash
opencode run -m libqllm/qwen2.5-coder "hello"
```
