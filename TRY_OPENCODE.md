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
- A GGUF model file (tested with Phi-3-mini-4k and Qwen 2.5 4K)

## Quick Start

### Terminal 1: Start qllmd

```bash
cd /home/quirinpa/libqllm

# Find your model path (or use your own path)
./bin/qllm-path "*Phi*"

# Start qllmd with the model (use full path, not glob)
# IMPORTANT: You MUST pass -c 4096 (or your model's context size)
# Without this, it defaults to 2048 and will cause errors
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

**In opencode, you MUST see a response from the model** in the chat area. This is the ultimate test - if you don't see the model's response in opencode's TUI, something is wrong.

**IMPORTANT:** With 4K context models, the system prompt must be truncated heavily to fit. This may cause the model to produce unusual or garbled responses. For best results, use an 8K+ context model (see "Testing Different Models" below).

### What Success Looks Like

After typing your message in opencode:

1. You should see the model's response appear in the chat area of the TUI
2. It should be a complete sentence or answer
3. The qllm-serve terminal should show HTTP 200

**Note:** On CPU-only machines, the model may take 2+ minutes to respond. Be patient! If you see "qllmd request timeout" in the logs, the model is still running but took too long.

**If you only see the build message but no response from the model, the integration is NOT working.**

Example of SUCCESS (with 8K+ model):
```
> build · phi-3
What is 1+1?

2
```

Example of SUCCESS (with 4K model - may be garbled):
```
> build · phi-3
What is 1+1?

Hello there! What's on your mind? [System prompt truncated due to length...]
```

Example of FAILURE (something is wrong):
```
> build · phi-3
What is 1+1?

(No response appears - this means something is broken)
```

## How to Verify It's Working (Optional)

These are additional ways to debug if opencode doesn't show a response:

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

### Model produces strange/garbled output

With 4K context (Phi-3, Qwen, etc.), truncating the system prompt breaks the model's understanding. The response may contain repeated text like "[System prompt truncated due to length - visit https://opencode.ai for help]" or be generally incoherent.

This affects ALL 4K context models. The solution is the same: use an 8K+ context model.

### Request times out ("qllmd request timeout")

If you see `ERROR - Streaming error: qllmd request timeout` or `ERROR - Streaming error: qllmd streaming error: [Errno 111] Connection refused`, it means:

1. **You forgot to pass `-c 4096`** to qllmd - this is **REQUIRED** for 4K models
2. The model is running slowly (CPU mode) and took more than 120 seconds to respond

**Solution:**
1. Make sure to pass `-c 4096` (or your model's context size) when starting qllmd
2. For CPU inference, responses may take 2+ minutes - be patient
3. Or use a faster machine with GPU acceleration

## Understanding the Logs

| Log Message | Meaning |
|-------------|---------|
| `Context strategy: tools=False, max_system=3840, reason=tools_oversized` | Tools don't fit, will skip them |
| `Context strategy: tools=True, max_system=..., reason=fits` | Everything fits |
| `Context strategy: tools=True, max_system=..., reason=truncate_system` | Tools fit, but system needs truncation |
| `System prompt truncated from X to Y chars` | System prompt was shortened |
| `Skipping tools: tools_oversized` | Tools were excluded |

## Testing Different Models (Recommended)

**For best results, use an 8K+ context model.** The 4K Phi-3 model requires heavy system prompt truncation, which causes garbled responses. An 8K model (like Mistral) has enough context to include both the full system prompt and tools.

### Phi-3-mini (4K context)

```bash
export MODEL_PATH="/path/to/Phi-3-mini-4k-instruct.Q8_0.gguf"
./bin/qllmd -p 4242 -c 4096 "$MODEL_PATH"
```

Expected: Tools skipped, system truncated (may produce garbled output)

### Qwen 2.5 (4K context)

```bash
export MODEL_PATH="/path/to/Qwen2.5-4B-Instruct-Q4_K_M.gguf"
./bin/qllmd -p 4242 -c 4096 "$MODEL_PATH"
```

Expected: Works but produces garbled output (same issue as Phi-3 4K)

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
