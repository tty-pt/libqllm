#ifndef QLLM_H
#define QLLM_H

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

void
qllm_compress(struct qllm_context *ctx, uint32_t limit);

void qllm_anchor_start(struct qllm_context *ctx);
void qllm_anchor_end(struct qllm_context *ctx);
void qllm_set_eos_bias(struct qllm_context *qctx,
		  int32_t start_tokens,
		  float max_bias);

#ifdef __cplusplus
}
#endif

#endif /* QLLM_H */
