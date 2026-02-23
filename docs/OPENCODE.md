# OpenCode Integration Guide

This guide explains how to integrate libqllm with the OpenCode AI coding assistant, allowing you to use local LLM models for code generation and assistance.

## Overview

libqllm provides an OpenAI-compatible HTTP API server (`qllm-serve`) that wraps the qllmd daemon, enabling OpenCode to use local GGUF models for inference. This setup gives you complete control over your AI coding assistant without relying on external API services.

## Architecture

```
OpenCode CLI
    ↓ (HTTP/OpenAI API)
qllm-serve (port 8001)
    ↓ (TCP protocol)
qllmd daemon (port 4242)
    ↓ (llama.cpp)
Local GGUF Model
```

## Prerequisites

1. **Built libqllm** with all binaries:
   ```bash
   make
   ```

2. **Downloaded GGUF model** (recommended: Phi-3-mini-4k-instruct.Q8_0.gguf):
   ```bash
   bin/qllm-list
   # Download your chosen model to ~/.cache/huggingface/hub/
   ```

3. **Python 3** with required packages:
   ```bash
   pip3 install flask requests
   ```

4. **OpenCode CLI** installed:
   ```bash
   npm install -g opencode
   # or follow instructions at https://opencode.ai
   ```

## Quick Start

### 1. Start the qllmd daemon

```bash
cd /path/to/libqllm
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH bin/qllmd -d -p 4242 Phi-3-mini-4k-instruct.Q8_0.gguf
```

This starts the daemon on port 4242. The `-d` flag runs it in daemon mode (background).

**Note**: The daemon will take 10-30 seconds to load the model and initialize the shared context.

### 2. Start the API server

```bash
cd /path/to/libqllm
bin/qllm-serve --port 8001
```

This starts the OpenAI-compatible HTTP server on port 8001.

### 3. Configure OpenCode

Create or edit `opencode.json` in your project directory (or globally in `~/.config/opencode/`):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "libqllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "libqllm (local)",
      "options": {
        "baseURL": "http://localhost:8001/v1",
        "apiKey": "dummy"
      },
      "models": {
        "phi-3": {
          "name": "Phi-3-mini-4k-instruct (local)",
          "limit": {
            "context": 4096,
            "output": 2048
          }
        }
      }
    }
  }
}
```

**Configuration notes**:
- `baseURL`: Must point to your qllm-serve instance (default: `http://localhost:8001/v1`)
- `apiKey`: Can be any value (e.g., "dummy") since local authentication is not enforced
- `limit.context`: Should match your model's context window
- `limit.output`: Maximum tokens for completion (should be ≤ context / 2)

### 4. Use OpenCode with libqllm

```bash
opencode run -m libqllm/phi-3 "Write a hello world function in Python"
```

Or interactively:
```bash
opencode chat -m libqllm/phi-3
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

### Test the API server

```bash
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-3",
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
opencode run -m libqllm/phi-3 "Write a simple test"
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
Server → Client: {"model":"Phi-3-mini-4k-instruct.Q8_0.gguf","template":"phi3"}
```

**qllm-serve HTTP API** (port 8001):
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
   bin/qllm-path Phi-3-mini-4k-instruct.Q8_0.gguf
   ```

3. Verify LD_LIBRARY_PATH is set:
   ```bash
   export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
   ```

### API returns 500 errors

**Symptom**: qllm-serve returns HTTP 500

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
1. Verify qllm-serve is running:
   ```bash
   curl http://localhost:8001/v1/models
   ```

2. Check `opencode.json` baseURL matches server port

3. Ensure `@ai-sdk/openai-compatible` npm package will be installed by OpenCode

### Slow generation

**Expected behavior**: CPU inference is inherently slow

**Improvements**:
- Use smaller models (Phi-3-mini instead of Llama-3-70B)
- Use lower quantization (Q4_K_M instead of Q8_0)
- Reduce context size: `bin/qllmd -n 1024 ...`
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
   bin/qllmd -n 512 -d -p 4242 model.gguf
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
# Terminal 1: Phi-3 on port 4242
bin/qllmd -d -p 4242 Phi-3-mini-4k-instruct.Q8_0.gguf

# Terminal 2: Mistral on port 4243
bin/qllmd -d -p 4243 Mistral-7B-Instruct-v0.3.Q8_0.gguf

# Terminal 3: API server for Phi-3
bin/qllm-serve --port 8001 --qllmd-port 4242

# Terminal 4: API server for Mistral
bin/qllm-serve --port 8002 --qllmd-port 4243
```

Update `opencode.json` with multiple providers:
```json
{
  "provider": {
    "libqllm-phi3": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://localhost:8001/v1", "apiKey": "dummy"},
      "models": {"phi-3": {"name": "Phi-3", "limit": {"context": 4096, "output": 2048}}}
    },
    "libqllm-mistral": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://localhost:8002/v1", "apiKey": "dummy"},
      "models": {"mistral": {"name": "Mistral-7B", "limit": {"context": 8192, "output": 4096}}}
    }
  }
}
```

### Startup Script

Create a script to launch both daemons:

```bash
#!/bin/bash
# start-libqllm.sh

cd /path/to/libqllm

# Start qllmd
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
bin/qllmd -d -p 4242 Phi-3-mini-4k-instruct.Q8_0.gguf

# Wait for daemon to initialize
sleep 15

# Start API server
bin/qllm-serve --port 8001 &

echo "libqllm services started"
echo "qllmd: localhost:4242"
echo "qllm-serve: http://localhost:8001"
```

## Testing Results

### Verified Working

1. **qllmd daemon** - Loads model and handles requests correctly
2. **info command** - Returns model info:
   ```
   $ echo "info" | nc localhost 4242
   {"model":"Phi-3-mini-4k-instruct.Q8_0.gguf","template":"phi3"}
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

4. **qllm-serve API**:
   - `GET /v1/models` - Returns detected model
   - `GET /health` - Returns health status
   - `POST /v1/chat/completions` - Works with full conversation history

### Example API Response

```bash
$ curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "phi-3",
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

(Note: Using Phi-3-mini Q8_0 on CPU)

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

## Current Status (February 2026)

### Working Features

- ✅ Server runs and accepts HTTP connections
- ✅ OpenAI-compatible API endpoints (/v1/models, /health, /v1/chat/completions)
- ✅ Non-streaming chat completions
- ✅ Streaming chat completions (proper SSE format)
- ✅ System prompt truncation for large prompts
- ✅ curl and other standard HTTP clients work perfectly
- ✅ OpenCode connects to server and sends requests
- ✅ OpenCode receives streaming data (verified via debug logs)

### Known Issues

- The OpenCode TUI may hang after receiving data ("build · phi-3" state). This appears to be a display/compatibility issue with how OpenCode's Bun runtime processes the streaming response, even though data is being received correctly.

### Debugging

If you encounter issues, run opencode with debug logging:
```bash
opencode --print-logs --log-level DEBUG --model libqllm/phi-3 run "test"
```

Look for these indicators in the logs:
- `providerID=libqllm` - Correct provider is selected
- `service=llm ... stream` - Streaming is enabled
- `service=message.part.delta publishing` - Data is being received
- `service=session.processor process` - Request is being processed
