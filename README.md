# libqllm
This is a library that is focused on making LLM usage easy and portable. The idea is you don't have to bother about CUDA or anything like that. You just install it via your favorite package manager, and then you can use it to do inference and generate embeddings. It is a wrap around llama-cpp, but with a simple interface and the build complexity hidden. It uses Vulkan on Linux and Metal on MacOS to allow for this portability.

This project comes with a few tools for ease-of-use: an axil module (`libaxil-qllm`) that serves chat sessions and OpenAI-compatible HTTP endpoints, a client program (`qllm-chat`), bash completion, `qllm-list` for listing your gguf models, and `qllm-path` for getting the real path to one.

## Installation
Check out [these instructions](https://github.com/tty-pt/ci/blob/main/docs/install.md#install-ttypt-packages).
And use "libqllm" as the package name.

The package installs the `libaxil-qllm` axil module, the `qllm-chat`, `qllm-list`
and `qllm-path` tools, and the `qllm.h` header.

## Building from source
```sh
git clone https://github.com/tty-pt/mk.git        # a sibling dir is expected
git clone --recursive https://github.com/tty-pt/libqllm.git
cd libqllm && make
sudo make install
```

Dependencies: the `axil`, `libxylem`, `libqmap` and `libqsys` packages (from the
tty.pt repo) provide the headers and libraries, or pass `SITE=/path/to/site` to
use a site checkout instead of installed packages. The build downloads the
LunarG Vulkan SDK and compiles `submodules/llama.cpp` from source, so `cmake`,
`clang` and `wget` are also required.

## Running the server
libqllm is served as the axil module `libaxil-qllm` (`lib/libaxil-qllm.so`). Pick the
model with the `QLLM_MODEL_PATH` env var (required); `QLLM_CRB_PATH` optionally points
at a system-prompt file (default `crb.txt` relative to the server's cwd, skipped if
absent).

```sh
QLLM_MODEL_PATH=/path/to/model.gguf axil -A -d -p 4242 -m libaxil-qllm
# run from ~/libqllm/lib so `-m libaxil-qllm` resolves, or pass an absolute
# path without the ".so" suffix, e.g. -m /home/you/libqllm/lib/libaxil-qllm
```

If you installed the `libqllm` package, `-m libaxil-qllm` resolves from
anywhere (the module lives in `/usr/lib`); the working-directory hint above only
matters when running from a source checkout.

- `-A` auto-authenticates every connection; without it, axil gates
  `on_axil_disconnect` and telnet session cleanup never fires.
- The module registers `ask`/`chat` (telnet) and the OpenAI-compatible HTTP
  endpoints `POST /v1/embeddings` and `POST /v1/chat/completions`.
- LLM inference runs on the module's worker thread, so it never blocks axil's
  single-threaded event loop.

## Chat usage
Follow these instructions to install [huggingface-cli](https://huggingface.co/docs/huggingface_hub/guides/cli) so you can download models you can run.

Download a model, like:
```sh
huggingface-cli download reedmayhew/Grok-3-gemma3-4B-distilled gemma-3-finetune.Q8_0.gguf
```

Run:
```sh
QLLM_MODEL_PATH=gemma-3-finetune.Q8_0.gguf axil -A -d -p 4242 -m libaxil-qllm # To start the service
qllm-chat # To talk to it
```

Telnet chat also works:
```sh
nc 127.0.0.1 4242        # then type:
#   ask tell me a joke   → tokens stream out, ends with "\r\n.\r\n"
#   chat                 → reset history + KV cache
#   ask $ pwd            → run a shell command as part of the chat
```

## OpenAI-compatible HTTP API

`POST /v1/chat/completions` — standard chat completion; send `"stream":true` for
SSE token streaming (ends with `data: [DONE]`):
```sh
curl -sS -N -X POST localhost:4242/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"hi"}],"stream":true}'

curl -sS -X POST localhost:4242/v1/chat/completions \
    -d '{"messages":[{"role":"user","content":"hi"}]}'
```

`POST /v1/embeddings` — OpenAI-compatible single-string embeddings. This lets any
OpenAI-compatible embeddings client — such as the `mm --embed` engine or the
`pi-mem` extension — use a locally-served embeddings provider:

```sh
# Direct query:
curl -sS -X POST localhost:4242/v1/embeddings \
    -d '{"input":"puddle reflection"}'

# Consume from the mm engine (in the site repo):
MM_EMBED_URL=http://localhost:4242/v1/embeddings \
  external/mm/bin/mm scan --embed "puddle" --topic mirror --level 2
```

The endpoint accepts a single-string `"input"` (batch input is reserved as a
follow-up) with an optional `"model"` field, and returns the standard OpenAI
embeddings JSON shape (`data[0].embedding` float array).

Note: mm's client caps vectors at 512 (VEC_MAX), so an 896-dim model like
`qwen2.5-0.5b-instruct` will not scan in mm — the endpoint itself is correct.

## Model selection

The model is chosen once, at server start, via the required `QLLM_MODEL_PATH`
env var (a `QLLM_CRB_PATH` env var optionally points at a system-prompt file,
default `crb.txt` in the server's cwd). A model *name* may be resolved to a
path with `qllm-path` in a later release.

Looking ahead, selection is meant to become **per-request**: both HTTP endpoints
already carry the OpenAI `model` field (today it only echoes `QLLM_MODEL_PATH`),
and it will feed a model registry so one server can serve several models.
Selecting models via extra axil command-line options was considered and
deliberately passed on: axil's option parsing is core-owned, and a per-request
`model` field is the right seam for the multi-model future.