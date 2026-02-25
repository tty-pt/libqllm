# libqllm OpenCode Integration Plan

## Overview

This document outlines the implementation plan for adding true OpenCode support to libqllm through:
1. **Tool calling support** - Enable function calling for AI coding agents
2. **True streaming** - Token-by-token streaming for real-time responses

## Architecture

```
OpenCode
    ↓ HTTP /v1/chat/completions (with tools)
qllm-serve (Python)
    ↓ TCP messages command (with tools JSON)
qllmd (C)
    ↓
format_conversation() - include tools in prompt
    ↓
qllm_generate_stream() - true streaming
    ↓
parse_tool_call() - detect <tool_call>... pattern
    ↓
qllm-serve → OpenAI SSE format
    ↓
OpenCode
```

## Implementation Plan

### Phase 1: Extend qllmd messages Command

**File**: `src/qllmd.c`

#### 1.1 Parse tools and stream in do_MESSAGES()

Modify `do_MESSAGES()` to accept extended JSON:
```json
{
  "messages": [...],
  "tools": [...],    // NEW: tools array
  "stream": false    // NEW: streaming flag
}
```

- Extract `tools` from JSON (optional, default null)
- Extract `stream` boolean (optional, default false)

#### 1.2 Add format_tools() function

Convert OpenAI `tools` array to text description for prompt injection.

```c
static char *format_tools(cJSON *tools, chat_template_t template);
```

**Output format**:
```
You have access to functions. To call a function, use the format:
<tool_call>function_name|arguments</tool_call>

Available tools:
1. read_file
   Read contents of a file
   Parameters: {"path": "string"}

2. write_file
   Write content to a file
   Parameters: {"path": "string", "content": "string"}
```

#### 1.3 Modify format_conversation()

Add tools parameter and inject after system prompt:
```c
static char *format_conversation(cJSON *messages, cJSON *tools, 
                                  chat_template_t template, int skip_global_system);
```

### Phase 2: Tool Call Detection

**File**: `src/qllmd.c`

#### 2.1 Add parse_tool_call()

Detect `<tool_call>func_name|args</tool_call>` pattern in generated output.

```c
typedef struct {
    char name[256];
    char arguments[4096];
} tool_call_t;

static int parse_tool_call(const char *output, tool_call_t *tc);
```

**Pattern**: `<tool_call>function_name|{"arg1": "value1"}</tool_call>`

#### 2.2 Response Format

New JSON response format with finish_reason:
```json
// Regular response
{"content": "The weather is sunny.", "finish_reason": "stop"}

// Tool call response
{
  "content": "Let me check that file for you.",
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\": \"main.c\"}"
      }
    }
  ],
  "finish_reason": "tool_calls"
}
```

### Phase 3: True Streaming Support

**File**: `src/qllmd.c`

#### 3.1 Streaming Response Protocol (SSE format)

```
data: {"type":"chunk","delta":"Hello"}
data: {"type":"chunk","delta":" world"}
data: {"type":"tool_call","id":"call_1","function":{"name":"read_file","arguments":"{\"path\":\"main.c\"}"}}
data: {"type":"stop","finish_reason":"tool_calls"}
```

#### 3.2 Implement Streaming in generate()

Add new streaming generation function:
```c
void generate_stream(int fd, const char *prompt, 
                    int (*on_token)(void *ctx, const char *token, size_t len),
                    void *ctx);
```

Use existing `qllm_generate_stream()` from qllm.h with callback to send SSE chunks.

#### 3.3 Handle stream Parameter

In `do_MESSAGES()`:
- If `stream: true`: use streaming mode
- If `stream: false` or omitted: use existing blocking mode

### Phase 4: Update qllm-serve (Python)

**File**: `bin/qllm-serve`

#### 4.1 Forward tools to qllmd

Extract `tools` from OpenAI request and pass via messages command:
```python
request_tools = data.get('tools', [])
stream = data.get('stream', False)

# Forward to qllmd
cmd = f'messages {{"messages": {messages_json}, "tools": {json.dumps(request_tools)}, "stream": {str(stream).lower()}}}\n'
```

#### 4.2 Handle Streaming Response

Convert qllmd SSE to OpenAI SSE format:
```python
# qllmd format
data: {"type":"chunk","delta":"Hello"}

# OpenAI format  
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"Hello"}}]}
```

#### 4.3 Format tool_calls

Parse qllmd tool_calls and convert to OpenAI format:
```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "read_file",
        "arguments": "{\"path\": \"main.c\"}"
      }
    }
  ]
}
```

#### 4.4 Handle Tool Result Messages

Accept `role: "tool"` messages from OpenCode:
```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "File contents..."
}
```

Forward to qllmd for continued generation.

### Phase 5: Error Handling

**Files**: `src/qllmd.c`, `bin/qllm-serve`

#### 5.1 OpenAI-style Errors

```json
{
  "error": {
    "message": "Invalid JSON in request",
    "type": "invalid_request_error",
    "param": "messages",
    "code": 400
  }
}
```

#### 5.2 HTTP Status Codes

| Code | Error Type |
|------|------------|
| 400  | Invalid request |
| 401  | Authentication failed |
| 429  | Rate limit exceeded |
| 500  | Internal server error |
| 503  | Service unavailable |

## Protocol Specification

### Request: messages Command

```
Client → qllmd: messages {"messages":[...],"tools":[...],"stream":true}
```

### Response: Streaming Mode

```
qllmd → Client: data: {"type":"start","id":"chatcmpl-xxx"}
qllmd → Client: data: {"type":"chunk","delta":"Hello"}
qllmd → Client: data: {"type":"chunk","delta":" world"}
qllmd → Client: data: {"type":"tool_call","id":"call_1","function":{"name":"read_file","arguments":"{}"}}
qllmd → Client: data: {"type":"stop","finish_reason":"tool_calls"}
```

### Response: Non-streaming Mode

```
qllmd → Client: {"content":"...","finish_reason":"stop"}
qllmd → Client: \x04
```

## Backward Compatibility

- If no `tools` in request → behave as before (no tool calling)
- If `stream` is false or omitted → use existing blocking mode
- Existing `ask` and `messages` commands unchanged

## Testing Checklist

- [ ] Simple chat without tools works
- [ ] Chat with tools generates tool call
- [ ] Streaming sends tokens in real-time
- [ ] Tool result message continues conversation
- [ ] OpenCode can use tools
- [ ] Error responses are OpenAI-compatible

## File Changes Summary

| File | Changes |
|------|---------|
| `src/qllmd.c` | Parse tools, format_tools, streaming, tool call detection |
| `src/qllmd.h` | (none needed) |
| `include/ttypt/qllm.h` | (none needed - already has streaming) |
| `bin/qllm-serve` | Forward tools, handle streaming, format tool_calls |

## References

- OpenAI Chat Completions API: https://platform.openai.com/docs/api-reference/chat/create
- OpenAI Tool Calls: https://platform.openai.com/docs/guides/function-calling
- llama.cpp minja library: `submodules/llama.cpp/vendor/minja/chat-template.hpp`
- SSE Format: https://html.spec.whatwg.org/multipage/server-sent-events.html
