## 1.1.0

- **axil module**: `libaxil-qllm` is now an axil module rather than a
  standalone daemon; `xy_install()` registers the `ask`/`chat` telnet commands
  and the OpenAI-compatible HTTP endpoints.
- **OpenAI-compatible embeddings**: new `POST /v1/embeddings` endpoint —
  single-string input, standard OpenAI JSON shape (`data[0].embedding`).
- **Worker-thread inference**: LLM inference runs on the module's worker
  thread, so it never blocks axil's single-threaded event loop.
- **Sliding-window context rework**: ported the sliding-window branch — model
  cache with refcounting, multi-context accounting (`n_contexts`),
  `qllm_compress`/`qllm_anchor_*`, `qllm_n_ctx`, history management
  (`qllm_prime`, `qllm_chat`, `qllm_render`, `qllm_reset`), and the
  sampler/grammar API (`qllm_set_grammar`, `qllm_sampler_*`).
- **Mock-based test suite**: unit (`tests/unit/core`), edge (`tests/edge`)
  and stress (`tests/stress`) tests compiled against llama/gguf/corm/vulkan
  mocks — `make test`.
- **Tools**: the `qllm-chat` client plus the `qllm-list` and `qllm-path`
  helpers.
- Dependencies updated; README shows how to download a GGUF model with `curl`
  (no `huggingface-cli` needed).

## [1.0.0] - 2026-05-02

- GPU offload improvements (`max_offload_bytes`) and better GGUF layer
  estimation.
- `qllm-chat`: tools support, first-char bug fix, larger system-prompt
  truncation.
- OpenAI-compatible streaming: final completion chunk with usage before
  `data: [DONE]`.
- `make test` target; llama/gguf sampler mocks aligned; integration
  orchestration scripts (`scripts/run-integration.sh`).
- Build / CI: winget target replacing pacman/mingw, brew fixes.

## [0.0.3] - 2026-02-01

- Add `pacman_mingw` build target; test target touched.

## [0.0.2] - 2025-12-22

- EOS bias (`qllm_set_eos_bias`).

## [0.0.1] - 2025-12-17

- Basic sliding-window context.