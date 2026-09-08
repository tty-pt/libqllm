#ifndef QLLM_H
#define QLLM_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for the model + context */
struct qllm_context;

/*
 * Configuration structure for creating a QLLM context.
 * All fields optional except model_path.
 */
struct qllm_config {
	const char   *model_path; /* Required */
	int32_t       n_ctx;      /* Context size (default 2048) */
	int32_t       n_threads;  /* Number of CPU threads (default: half of CPUs) */
	uint32_t      max_offload_bytes; /* Max byte offload */
	int32_t      n_contexts; /* How many contexts to account for */
};

/*
 * Create a new QLLM context.
 * Returns NULL on failure.
 */
struct qllm_context *
qllm_create(const struct qllm_config *cfg);

/*
 * Free a QLLM context.
 */
void
qllm_free(struct qllm_context *ctx);

/*
 * Non-streaming generation.
 * Writes into `out` (user allocated).
 * Returns number of bytes written, or -1 on error.
 */
long
qllm_generate(struct qllm_context *ctx,
	      const char *prompt,
	      char *out,
	      size_t out_size);

/*
 * Streaming callback type.
 * `chunk` is a piece of text from generation.
 * `len` is the chunk size.
 */
typedef void (*qllm_token_cb)(void *user,
			      const char *chunk,
			      size_t len);

/*
 * Streaming generation.
 * Calls cb() for each generated text chunk.
 * Returns 0 on success, < 0 on error.
 */
int
qllm_generate_stream(struct qllm_context *ctx,
		     const char *prompt,
		     qllm_token_cb cb,
		     void *user);

/*
 * Compute embeddings for the entire input text.
 * Writes a vector of size >= model embedding dimension.
 *
 * Returns:
 *   >0  = embedding dimension (success)
 *   <0  = error
 */
int
qllm_embed(struct qllm_context *ctx,
	   const char *text,
	   float *out,
	   size_t out_dim);

/*
 * Prime the context with a prompt.
 *
 * Returns:
 *   0  on success
 *  <0  on error
 */
int
qllm_prime(struct qllm_context *ctx,
	   const char *prompt);

/*
 * Generate the next token as text.
 *
 * Returns:
 *   >0  number of bytes written to 'out' (UTF-8, NUL-terminated)
 *    0  end of generation (EOS)
 *   <0  error
 */
int
qllm_next(struct qllm_context *ctx,
	  char *out,
	  size_t out_size);

/* Chat message — mirrors llama_chat_message. */
struct qllm_message {
	const char *role;    /* "system" | "user" | "assistant" | ... */
	const char *content;
};

/*
 * Streaming chat completion.
 * Renders the full conversation via the model's chat template, primes the
 * context with only the newly-added increment, then streams via cb().
 * prev_prompt is the previously rendered prompt (or NULL for the first turn).
 * Returns 0 on success, < 0 on error.
 */
int
qllm_chat(struct qllm_context *ctx,
	  const struct qllm_message *msgs, size_t n_msgs,
	  const char *prev_prompt,
	  qllm_token_cb cb, void *user);

/*
 * Render a conversation with the model's chat template.
 * add_ass=true appends the assistant prefix (for priming generation);
 * add_ass=false renders the exact conversation state.
 * On success *out is malloc'd (caller frees) and *out_len is set.
 * Returns 0 on success, < 0 on error.
 */
int
qllm_render(struct qllm_context *ctx,
	    const struct qllm_message *msgs, size_t n_msgs,
	    bool add_ass,
	    char **out, size_t *out_len);

/*
 * Reset the context's generation state: clears the KV cache and resets the
 * running position, keeping the loaded model and context alive.
 */
void
qllm_reset(struct qllm_context *ctx);

#ifdef __cplusplus
}
#endif

#endif /* QLLM_H */
