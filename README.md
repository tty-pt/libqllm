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
