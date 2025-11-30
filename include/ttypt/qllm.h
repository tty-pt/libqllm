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
ssize_t
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

#ifdef __cplusplus
}
#endif

#endif /* QLLM_H */
