# Testing OpenCode with libqllm

This document explains how to test OpenCode with a local LLM using libqllm.

## What This Does

OpenCode sends a large system prompt (~10KB) plus tool definitions (~11KB) to the LLM. For small context models (like Phi-3 with 4K), this exceeds the limit. This implementation adds dynamic context management that:

1. Detects the model's context size from qllmd
2. Calculates if prompt + tools fit
3. Truncates the system prompt or skips tools if needed

## Prerequisites

- libqllm already built (`make` in the project root)
- opencode CLI installed (`npm install -g opencode-ai`)
- A GGUF model file (tested with Phi-3-mini-4k)

## Quick Start

### Terminal 1: Start qllmd

```bash
cd /home/quirinpa/libqllm

# Find your model path (or use your own path)
./bin/qllm-path "*Phi*"

# Start qllmd with the model (use full path, not glob)
# Example path - replace with your actual model path
export MODEL_PATH="/path/to/Phi-3-mini-4k-instruct.Q8_0.gguf"
export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
./bin/qllmd -p 4242 -c 4096 "$MODEL_PATH"
```

Wait for it to finish loading. You'll see:
```
qllmd: Detected model template: Phi-3
qllmd: Shared context created successfully
```

### Terminal 2: Start qllm-serve

```bash
cd /home/quirinpa/libqllm
./bin/qllm-serve --port 8001 --qllmd-port 4242
```

You'll see:
```
qllm-serve: OpenAI-compatible API server
Listening on 127.0.0.1:8001
```

### Terminal 3: Run OpenCode

```bash
opencode run -m libqllm/phi-3
```

Then type a simple message like:
```
What is 1+1?
```

### Expected Results

**In qllm-serve terminal, you should see:**
```
Context strategy: tools=False, max_system=3840, reason=tools_oversized
System prompt truncated from 10022 to 3840 chars
Skipping tools: tools_oversized
```

This means:
- The system detected 4096 context
- Tools (~11KB) + prompt exceeds 4K limit
- System prompt was truncated to 3840 chars
- Tools were skipped (they don't fit)

**In opencode, you should get a response** (the model will respond despite truncated context).

### Verifying the Response

1. After typing your message in opencode, watch for the model's response
2. Watch the qllm-serve terminal - you should see HTTP 200
3. Watch opencode - the model's response should appear in the TUI

## How to Verify It's Working

### 1. Verify qllmd is running

```bash
echo "info" | nc -w 2 127.0.0.1 4242
```

Should return:
```json
{"model":"Phi-3-mini-4k-instruct.Q8_0.gguf","template":"phi3","n_ctx":4096}
```

### 2. Verify qllm-serve is running

```bash
curl -s http://127.0.0.1:8001/health
```

Should return:
```json
{"status":"ok","model":"Phi-3-mini-4k-instruct.Q8_0.gguf"}
```

### 3. Test the API directly

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"phi-3","messages":[{"role":"user","content":"Hi"}],"stream":false}'
```

Should return JSON with the model's response.

### "Address already in use"

Something is using the port. Kill existing processes:

```bash
pkill -f qllmd
pkill -f qllm-serve
sleep 2
# Then retry starting them
```

### qllmd fails to load model

The model path must be a full path, not a glob. Use:

```bash
export MODEL_PATH="/full/path/to/model.gguf"
./bin/qllmd -p 4242 -c 4096 "$MODEL_PATH"
```

### qllmd info shows wrong n_ctx

Make sure you pass `-c` with the desired context size. Without it, defaults to 2048.

```bash
./bin/qllmd -p 4242 -c 4096 "$MODEL_PATH"  # 4K context
./bin/qllmd -p 4242 -c 8192 "$MODEL_PATH"  # 8K context
```

### opencode not using libqllm

Check what models opencode sees:

```bash
opencode models libqllm
```

Should show: `libqllm/phi-3`

## Understanding the Logs

| Log Message | Meaning |
|-------------|---------|
| `Context strategy: tools=False, max_system=3840, reason=tools_oversized` | Tools don't fit, will skip them |
| `Context strategy: tools=True, max_system=..., reason=fits` | Everything fits |
| `Context strategy: tools=True, max_system=..., reason=truncate_system` | Tools fit, but system needs truncation |
| `System prompt truncated from X to Y chars` | System prompt was shortened |
| `Skipping tools: tools_oversized` | Tools were excluded |

## Testing Different Models

### Phi-3-mini (4K context)

```bash
export MODEL_PATH="/path/to/Phi-3-mini-4k-instruct.Q8_0.gguf"
./bin/qllmd -p 4242 -c 4096 "$MODEL_PATH"
```

Expected: Tools skipped, system truncated

### Mistral-7B (8K context)

```bash
export MODEL_PATH="/path/to/mistral-7b-instruct.Q4_K_M.gguf"
./bin/qllmd -p 4242 -c 8192 "$MODEL_PATH"
```

Expected: Tools included, minimal truncation

## Cleanup

```bash
pkill -f qllmd
pkill -f qllm-serve
```

Or press Ctrl+C in each terminal.
