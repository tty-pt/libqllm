# libqllm

[![C99](https://img.shields.io/badge/C-C99-555?logo=c)](#)
[![BSD-2-Clause](https://img.shields.io/badge/License-BSD--2--Clause-blue)](#)
[![Vulkan/Metal](https://img.shields.io/badge/Vulkan%2FMetal-backends-4B8BBE)](#)

> Local LLM inference and embeddings via llama.cpp, served as an axil module.

Making LLM usage easy and portable: no CUDA to worry about, no build
complexity to deal with. You install it through your favorite package manager
and use it for inference and embeddings. It wraps llama.cpp behind a single C
API, and uses Vulkan on Linux and Metal on macOS for that portability.

It ships as the axil module `libaxil-qllm`, which serves chat sessions and
OpenAI-compatible HTTP endpoints, together with a client program (`qllm-chat`),
bash completion, `qllm-list` for listing your GGUF models, and `qllm-path` for
getting the real path to one.

## Contents

- [Features](#features)
- [Install](#install)
- [Build from source](#build-from-source)
- [Quickstart](#quickstart)
- [Running the server](#running-the-server)
- [Chat usage](#chat-usage)
- [OpenAI-compatible HTTP API](#openai-compatible-http-api)
- [Model selection](#model-selection)
- [C API](#c-api)
- [Testing](#testing)
- [Documentation](#documentation)
- [License](#license)

## Features

- **Easy and portable** — no CUDA; install via your package manager and
  generate text or embeddings right away.
- **An axil module** — `libaxil-qllm` runs inside axil: telnet `ask`/`chat`
  commands and OpenAI-compatible HTTP endpoints (`/v1/chat/completions`,
  `/v1/embeddings`).
- **Worker-thread inference** — generation runs on the module's worker thread
  and never blocks axil's single-threaded event loop.
- **Portable backends** — Vulkan on Linux, Metal on macOS, all the CUDA pain
  hidden from you.
- **Sliding-window contexts** — long sessions survive limited context sizes via
  anchored `qllm_compress`, with multi-context accounting and a refcounted
  model cache.
- **Constrained decoding** — GBNF grammars via `qllm_set_grammar` and the
  sampler API.
- **Tools** — `qllm-chat` (chat client), `qllm-list` (list your GGUF models),
  `qllm-path` (resolve a model's real path).
- **One C API** — generation, streaming, embeddings and conversation history
  through `include/ttypt/qllm.h`.

## Install

Prebuilt packages are distributed from [tty.pt](https://tty.pt) for Linux (APT
/ Alpine / Arch / Fedora-RHEL), macOS (Homebrew), Windows (winget / MSYS2), and
OpenBSD. Follow the [installation instructions]
(https://github.com/tty-pt/ci/blob/main/docs/install.md) and use **libqllm**
as the package name.

The package installs the `libaxil-qllm` axil module, the `qllm-chat`,
`qllm-list` and `qllm-path` tools, and the `qllm.h` header.

## Build from source

```sh
git clone https://github.com/tty-pt/mk.git        # a sibling dir is expected
git clone --recursive https://github.com/tty-pt/libqllm.git
cd libqllm && make
make test             # run the in-tree mock-based test suite
sudo make install     # lib + headers + axil-qllm.pc -> $(PREFIX), default /usr
```

Link it from your own C code:

```sh
cc my_app.c $(pkg-config --cflags --libs axil-qllm)
```

**Dependencies:** the `axil`, `libxylem`, `libcorm` and `libqsys` packages
(from the tty.pt repo) provide the headers and libraries, or pass
`SITE=/path/to/site` to use a site checkout instead of installed packages. The
build downloads the LunarG Vulkan SDK and compiles `submodules/llama.cpp` from
source, so `cmake`, `clang` and `wget` are also required.

## Quickstart

```c
#include <stdio.h>
#include <ttypt/qllm.h>

int main(int argc, char **argv)
{
	struct qllm_config cfg = { .model_path = argv[1] };
	struct qllm_context *ctx = qllm_create(&cfg);
	char out[1024];
	long n = qllm_generate(ctx, "tell me a joke", out, sizeof out);

	if (n < 0)
		return 1;
	printf("%.*s\n", (int)n, out);
	qllm_free(ctx);
	return 0;
}
```

## Running the server

libqllm is served as the axil module `libaxil-qllm` (`lib/libaxil-qllm.so`).
Pick the model with the `QLLM_MODEL_PATH` env var (required); `QLLM_CRB_PATH`
optionally points at a system-prompt file (default `crb.txt` relative to the
server's cwd, skipped if absent).

```sh
QLLM_MODEL_PATH=/path/to/model.gguf axil -A -d -p 4242 -m libaxil-qllm
# run from ~/libqllm/lib so `-m libaxil-qllm` resolves, or pass an absolute
# path without the ".so" suffix, e.g. -m /home/you/libqllm/lib/libaxil-qllm
```

If you installed the `libqllm` package, `-m libaxil-qllm` resolves from
anywhere (the module lives in `/usr/lib`); the working-directory hint above
only matters when running from a source checkout.

- `-A` auto-authenticates every connection; without it, axil gates
  `on_axil_disconnect` and telnet session cleanup never fires.
- The module registers `ask`/`chat` (telnet) and the OpenAI-compatible HTTP
  endpoints `POST /v1/embeddings` and `POST /v1/chat/completions`.
- LLM inference runs on the module's worker thread, so it never blocks axil's
  single-threaded event loop.

## Chat usage

Download a small instruct model's GGUF directly with curl — no
huggingface-cli needed. A 0.5B model is ~490 MB and runs on a CPU-only machine:

```sh
curl -L -o qwen2.5-0.5b-instruct-q4_k_m.gguf \
    https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

Start the server and talk to it:
```sh
QLLM_MODEL_PATH=qwen2.5-0.5b-instruct-q4_k_m.gguf axil -A -d -p 4242 -m libaxil-qllm # server
qllm-chat                                                                             # chat client
```

Bigger models answer better but need more RAM/VRAM. To pick a different GGUF
from the Hugging Face Hub without installing the CLI, the pattern is
`curl -L .../resolve/main/<file>`. For gated models you can instead install
[huggingface-cli](https://huggingface.co/docs/huggingface_hub/guides/cli) and run
`huggingface-cli download <org>/<repo> <file>`.

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

## C API

Everything lives in `<ttypt/qllm.h>`; all functions are `extern "C"`-safe.

- **Configuration / lifecycle** — `struct qllm_config` (all fields optional
  except `model_path`), `qllm_create`, `qllm_free`.
- **Generation** — `qllm_generate` (into a user buffer), `qllm_generate_stream`
  (`qllm_token_cb` per chunk), `qllm_next` (one token, with an explicit
  sampler).
- **Embeddings** — `qllm_embed` (mean-pooled; needs
  `enable_embeddings` in the config).
- **Sampling & grammar** — `qllm_set_grammar`, plus the sampler chain
  `qllm_sampler_create`, `qllm_sampler_add_grammar`, `qllm_sampler_free`.
- **Conversation & context** — `qllm_prime`, `qllm_chat` (streaming chat
  completion over `struct qllm_message`), `qllm_render`, `qllm_reset`,
  `qllm_set_seq` (multi-sequence), `qllm_n_ctx`.
- **Sliding window** — `qllm_anchor_start` / `qllm_anchor_end` protect the
  current prompt+response, `qllm_compress` evicts older tokens, and
  `qllm_set_eos_bias` makes long generations terminate on their own.

## Testing

`make test` (delegates to `tests/Makefile`). The suite compiles
`src/libqllm.c` against **mocks** for llama/gguf/corm/vulkan
(`tests/mocks/`), so it runs without a model or GPU:

- `tests/unit/core/` — model cache refcounts, compression, context creation,
  generation, extended API behavior.
- `tests/edge/` — edge cases (`tests/edge/test_edge_cases.c`).
- `tests/stress/` — long-running scenarios (`tests/stress/test_stress.c`).

## Documentation

- [include/ttypt/qllm.h](./include/ttypt/qllm.h) — full API (Doxygen-annotated;
  `make docs` generates man pages).
- [CHANGELOG.md](./CHANGELOG.md) — version history.
- [PORT.md](./PORT.md) — porting notes.
- [IMPLEMENTATION_STATUS.md](./IMPLEMENTATION_STATUS.md) — implementation
  status.

## License

BSD 2-Clause License. Copyright 2026 Paulo André Azevedo Quirino. See
`LICENSE`.