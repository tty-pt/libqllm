# qllmd Protocol Specification

Version: 2.0 (with JSON message support)

## Overview

qllmd uses a line-based TCP protocol on port 4242 (default). Commands are sent as newline-terminated strings, responses are streamed with a delimiter.

## Connection Flow

```
Client connects → TCP handshake
Client: chat\n
Client: <command>\n
Server: <response><delimiter>
```

## Delimiter

ASCII character 4 (EOT - End of Transmission): `\x04`

## Commands

### 1. chat
**Purpose**: Initialize a chat session

**Format**: `chat\n`

**Response**: None (implicit ready state)

**Example**:
```
Client: chat\n
```

---

### 2. ask (Legacy)
**Purpose**: Single-turn text query

**Format**: `ask <message>\n`

**Response**: Streaming text followed by delimiter

**Example**:
```
Client: ask Hello, how are you?\n
Server: I'm doing well, thank you for asking!\x04\n
```

**Note**: Maintained for backward compatibility with qllm-chat. New applications should use `messages` command.

---

### 3. messages (Recommended)
**Purpose**: Multi-turn conversation with full history

**Format**: `messages <JSON_ARRAY>\n`

**JSON Schema**:
```json
[
  {
    "role": "system" | "user" | "assistant",
    "content": "string"
  },
  ...
]
```

**Response**: Streaming text followed by delimiter

**Examples**:

Simple query:
```
Client: messages [{"role":"user","content":"Hello"}]\n
Server: Hello! How can I help you today?\x04\n
```

With system message:
```
Client: messages [{"role":"system","content":"You are a pirate"},{"role":"user","content":"Introduce yourself"}]\n
Server: Ahoy there, matey! I be a swashbucklin' pirate...\x04\n
```

Multi-turn conversation:
```
Client: messages [{"role":"user","content":"My name is Alice"},{"role":"assistant","content":"Hello Alice!"},{"role":"user","content":"What is my name?"}]\n
Server: Your name is Alice.\x04\n
```

**Error responses** (JSON format):
```json
{"error": "Invalid JSON", "detail": "Unexpected token..."}
{"error": "Messages must be an array"}
{"error": "Failed to format conversation"}
```

---

### 4. info
**Purpose**: Query loaded model information

**Format**: `info\n`

**Response**: JSON object with model metadata

**Example**:
```
Client: info\n
Server: {"model":"Phi-3-mini-4k-instruct.Q8_0.gguf","template":"phi3"}\n
```

**Fields**:
- `model`: Filename of loaded GGUF model
- `template`: Chat template type (phi3, mistral, gemma, generic)

---

## Chat Templates

qllmd automatically detects and applies chat templates based on model filename.

### Supported Templates

1. **Phi-3** (detected: "phi-3", "phi3")
2. **Mistral** (detected: "mistral")
3. **Gemma** (detected: "gemma")
4. **Generic** (fallback)

### Template Application

Templates are applied by `format_conversation()` when processing `messages` commands.

**Example** (Phi-3 multi-turn):
```
Input: [
  {"role":"system","content":"You are helpful"},
  {"role":"user","content":"Hello"},
  {"role":"assistant","content":"Hi!"},
  {"role":"user","content":"How are you?"}
]

Formatted prompt:
<|system|>
You are helpful<|end|>
<|user|>
Hello<|end|>
<|assistant|>
Hi!<|end|>
<|user|>
How are you?<|end|>
<|assistant|>
```

---

## System Prompts

Two ways to set system prompts:

### 1. Global (via -S flag)
```bash
qllmd -S "You are a helpful assistant" model.gguf
```
- Applied to every conversation
- Prepended before JSON system messages (if both present)

### 2. Per-message (via JSON)
```json
[{"role":"system","content":"You are a pirate"}, ...]
```
- Applied to that conversation only
- Takes precedence over global if `skip_global_system` is set

---

## Error Handling

### Connection Errors
- Daemon not running → Connection refused
- Timeout → No response after N seconds

### Command Errors
- Invalid JSON → `{"error":"Invalid JSON",...}`
- Wrong type → `{"error":"Messages must be an array"}`
- Malformed message → `{"error":"Failed to format conversation"}`

### Generation Errors
- Context overflow → Response truncated
- Model error → Partial response + early delimiter

---

## Performance Characteristics

- **Single-threaded**: One request processed at a time
- **Shared context**: All connections share one llama.cpp context
- **Stateless**: No server-side conversation tracking
- **Streaming**: Tokens sent as generated (not buffered)

---

## Client Examples

### netcat
```bash
echo -e 'chat\nmessages [{"role":"user","content":"Hello"}]' | nc localhost 4242
```

### Python
```python
import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('127.0.0.1', 4242))

# Initialize
sock.sendall(b'chat\n')

# Query
messages = [{"role": "user", "content": "Hello"}]
cmd = f'messages {json.dumps(messages)}\n'
sock.sendall(cmd.encode())

# Read response
response = b''
while True:
    chunk = sock.recv(4096)
    response += chunk
    if b'\x04' in response:
        break

print(response.split(b'\x04')[0].decode())
sock.close()
```

---

## Version History

**v2.0** (Current):
- Added `messages` command for multi-turn conversations
- Added `info` command for model detection
- JSON error responses
- Dynamic buffer allocation

**v1.0**:
- Initial release with `chat` and `ask` commands
- Single-turn only
- Fixed buffer sizes
