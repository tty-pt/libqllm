# libqllm
This is a library that is focused on making LLM usage easy and portable. The idea is you don't have to bother about CUDA or anything like that. You just install it via your favorite package manager, and then you can use it to do inference and generate embeddings. It is a wrap around llama-cpp, but with a simple interface and the build complexity hidden. It uses Vulkan on Linux and Metal on MacOS to allow for this portability.

This project comes with a few tools for ease-of-use, like a service program to allow for chat sessions, bash completion, and a client program. Also, qllm-list for listing your gguf models, and qllm-path for getting the real path to one.

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

## Embeddings usage

`qllmd` also exposes an OpenAI-compatible `POST /v1/embeddings` endpoint, backed
by `qllm_embed()`. This lets any OpenAI-compatible embeddings client — such as
the `mm --embed` engine or the `pi-mem` extension — use a locally-served
embeddings provider:

```sh
qllmd -d -p 4242 -c 512 gemma* # Same daemon, model already loaded

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
