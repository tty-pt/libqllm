# libqllm Project Overview

`libqllm` is a C-based library and daemon that provides a high-performance interface to Large Language Models (LLMs) via `llama.cpp`. It is designed to be lightweight, efficient, and easily integrable with other tools, notably providing an OpenAI-compatible API for AI coding assistants like **OpenCode**.

## Architecture

The project is composed of several key layers:

*   **`libqllm` (Core Library):** A C wrapper around `llama.cpp`. It manages model loading, inference, and context compression.
*   **`qllmd` (Daemon):** A server process that hosts the model and handles requests via a custom line-based TCP protocol and a native HTTP server. It uses a **shared context architecture** to support multiple connections efficiently while working around Vulkan backend limitations.
*   **`qllm-serve` (API Proxy):** A Python-based server that wraps `qllmd` to provide a fully OpenAI-compatible HTTP API (`/v1/chat/completions`).
*   **Networking & Utilities:** Leverages `ndc` for networking, `qmap` for persistent model metadata caching, and `cJSON` for JSON processing.

## Building and Running

### Prerequisites

*   `llama.cpp` (managed as a submodule in `submodules/llama.cpp`)
*   `ndc`, `ndx`, `qmap` (expected in parent directory or system paths)
*   Python 3 (for `qllm-serve` and integration tests)
*   Vulkan SDK (for GPU acceleration on Linux)

### Build Commands

```bash
# Build all components (libqllm, qllmd, qllm-chat)
make

# Build submodules separately if needed
make -C submodules/llama.cpp/build
```

### Running the Services

1.  **Start the daemon (`qllmd`):**
    ```bash
    export LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH
    bin/qllmd -p 4242 -c 4096 /path/to/model.gguf
    ```
    *   `-p`: TCP port (default 4242)
    *   `-c`: Context size (default 2048)
    *   `-S`: Global system prompt

2.  **Start the API proxy (`qllm-serve`):**
    ```bash
    bin/qllm-serve --port 8001 --qllmd-port 4242
    ```

3.  **Interact via Chat CLI:**
    ```bash
    bin/qllm-chat -p 4242
    ```

## Development Conventions

### Coding Style
*   **C:** Follows a style similar to BSD/KNF. Use tabs for indentation.
*   **Naming:** Functions in the library are prefixed with `qllm_`.

### Testing
The project has a robust testing suite divided into unit and integration tests.

*   **Unit Tests:** Located in `tests/unit`. These use mocks for `llama.cpp` and `ndc`.
    ```bash
    make -C tests run
    ```
*   **Integration Tests:** Located in `tests/integration`.
    *   `test_live_protocol.py`: Validates the TCP and HTTP protocols against a real running model.
    ```bash
    make live-protocol-test MODEL=/path/to/model.gguf
    ```

### OpenCode Integration
The project is optimized for OpenCode. Configuration is managed via `opencode.json`. Key features for OpenCode include:
*   **Tool Calling:** Detects `<tool_call>...` patterns in model output.
*   **Dynamic Truncation:** `qllm-serve` automatically truncates OpenCode's large system prompts to fit the model's context window.

## Key Files
*   `src/libqllm.c`: Core logic for llama.cpp interaction.
*   `src/qllmd.c`: Daemon implementation and TCP/HTTP request handling.
*   `include/ttypt/qllm.h`: Main public API.
*   `bin/qllm-serve`: Python OpenAI-compatible API server.
*   `docs/OPENCODE.md`: Detailed integration guide for OpenCode.
*   `docs/PROTOCOL.md`: Specification of the `qllmd` TCP protocol.
