# libqllm
This is a library that is focused on making LLM usage easy and portable. The idea is you don't have to bother about CUDA or anything like that. You just install it via your favorite package manager, and then you can use it to do inference and generate embeddings. It is a wrap around llama-cpp, but with a simple interface and the build complexity hidden. It uses Vulkan on Linux and Metal on MacOS to allow for this portability.

This project comes with a few tools for ease-of-use, like a service program to allow for chat sessions, bash completion, and a client program. Also, qllm-list for listing your gguf models, and qllm-path for getting the real path to one.

Additionally, `qllmd` exposes an OpenAI-compatible HTTP API directly, so AI coding assistants like OpenCode can use a local GGUF model without a separate proxy. See docs/OPENCODE.md for the complete integration guide.

## Dependencies
- **qmap >= 0.6.0** - Hashtable library for efficient model caching with pointer stability
- **ndc >= 1.0.0** - Network daemon core for qllmd service
- **ndx >= 0.2.0** - Dynamic loading system (dependency of ndc)
- **qsys >= 0.0.1** - System utilities
- **llama.cpp** - LLM inference engine (included as submodule)

## Installation
Check out [these instructions](https://github.com/tty-pt/ci/blob/main/docs/install.md#install-ttypt-packages).
And use "libqllm" as the package name.

## Chat usage
Follow these instructions to install [huggingface-cli](https://huggingface.co/docs/huggingface_hub/guides/cli) so you can download models you can run.

Download a model, like:
```sh
huggingface-cli download reedmayhew/Grok-3-gemma3-4B-distilled gemma-3-finetune.Q8_0.gguf
```

Run:
```sh
qllmd -d -p 4242 gemma* # To start the service
qllm-chat # To talk to it
```

## OpenCode Integration

libqllm can be used as a local model provider for the OpenCode AI coding assistant. This gives you complete control over your coding assistant without relying on external API services.

Quick setup:
```sh
# 1. Find a downloaded GGUF model
bin/qllm-list
MODEL="$(bin/qllm-path '*qwen2.5-coder*.gguf')"

# 2. Start qllmd. It serves both the TCP protocol and OpenAI-compatible HTTP.
LD_LIBRARY_PATH=./lib:$LD_LIBRARY_PATH bin/qllmd -d -p 4242 "$MODEL"

# 3. Configure OpenCode
cp opencode.json /path/to/your/project/opencode.json
```

The included `opencode.json` points OpenCode at `http://127.0.0.1:4242/v1`:

```json
{
  "provider": {
    "libqllm": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {"baseURL": "http://127.0.0.1:4242/v1", "apiKey": "dummy"},
      "models": {"qwen2.5-coder": {"name": "qwen2.5-coder", "limit": {"context": 4096, "output": 1024}}}
    }
  }
}
```

```sh
# 4. Use with OpenCode
opencode run -m libqllm/qwen2.5-coder "Write a hello world function"
```

See **docs/OPENCODE.md** for the complete integration guide, including:
- Detailed setup instructions
- Supported models and chat templates
- Performance tuning and troubleshooting
- Advanced usage (multiple models, custom prompts, etc.)

## Features

- **Multi-turn conversations**: Full conversation history via JSON message arrays
- **Chat template support**: Automatic detection for Phi-3, Mistral, Gemma, and generic models
- **System prompts**: Configure assistant behavior with `-S` flag or per-message
- **Shared context architecture**: Efficient memory usage with context reuse
- **OpenAI-compatible API**: Works with OpenCode and other AI coding tools
- **Dynamic model detection**: API automatically reports loaded model info
- **Vulkan acceleration**: Automatic GPU compute buffer usage on Linux
- **Model caching**: Faster daemon restarts with persistent model metadata cache

## Environment Variables
- **QLLM_CACHE_FILE** - Optional. Path to persistent model cache file. When set, model metadata is cached across daemon restarts for faster initialization. Example: `export QLLM_CACHE_FILE=~/.cache/qllm/models.cache`

## Known Limitations

### Vulkan Backend
- **Multi-context issue**: The Vulkan backend cannot handle multiple llama.cpp contexts reliably
- **Workaround**: qllmd uses a shared context architecture (single context reused across connections)
- **Impact**: Conversations are not isolated between different client connections
- **Memory leak**: `llama_batch_free()` is skipped to prevent crashes with Vulkan backend

### Performance
- **CPU inference**: Generation is slow on CPU (1-3 tokens/sec for 3.8B models)
- **No GPU offload**: Model and KV cache remain on CPU, only compute uses Vulkan
- **Single-threaded**: One request processed at a time

### Configuration
- **Fixed context size**: Must be set at daemon startup (default: 2048 tokens)
- **No dynamic model switching**: Requires daemon restart to change models
- **Single model per daemon**: Each daemon instance loads exactly one model

For more details and workarounds, see docs/OPENCODE.md.

## Troubleshooting

### Persistent Cache Issues

**Q: Why am I getting "QLLM_CACHE_FILE path too long" warnings?**  
A: The cache file path must be less than 4096 characters. Use a shorter path or disable caching by unsetting the environment variable.

**Q: Why am I getting "QLLM_CACHE_FILE exists but may not be accessible" warnings?**  
A: The cache file exists but doesn't have read/write permissions. Fix with:
```sh
chmod 600 $QLLM_CACHE_FILE
# or remove it to start fresh:
rm $QLLM_CACHE_FILE
```

**Q: How do I verify the cache is working?**  
A: The first daemon startup with a new model will be slower (building cache). Subsequent startups should be faster. You can also check if the cache file exists and grows after loading models:
```sh
ls -lh $QLLM_CACHE_FILE
```

**Q: How do I clear the cache?**  
A: Simply remove the cache file:
```sh
rm $QLLM_CACHE_FILE
```

**Q: Can I use the same cache file for multiple daemons?**  
A: Yes, the cache uses model paths as keys, so multiple daemons can safely share the same cache file. qmap handles concurrent access internally.

### Dependency Version Issues

**Q: Getting "qmap version mismatch" errors?**  
A: Run `pkg-config --modversion qmap` to check your installed version. You need qmap >= 0.6.0. If the version is wrong:
```sh
cd /path/to/qmap
sudo make install
sudo ldconfig
```

**Q: How do I check all dependency versions?**  
A:
```sh
pkg-config --modversion qmap    # should be >= 0.6.0
pkg-config --modversion ndc     # should be >= 1.0.0
pkg-config --modversion ndx     # should be >= 0.2.0
pkg-config --modversion qsys    # should be >= 0.0.1
```
