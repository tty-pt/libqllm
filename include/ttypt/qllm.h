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
    uint32_t      max_offload_bytes; /* Max byte offload (legacy) */
    int32_t       n_gpu_layers; /* If >0, force number of GPU layers to keep (pass-through -g) */
    int32_t      n_contexts; /* How many contexts to account for */
	
	/* Sampling parameters (added for better generation control) */
	float         temperature;    /* Temperature (default 0.7, 0.0 = greedy) */
	int32_t       top_k;         /* Top-K sampling (default 40, 0 = disabled) */
	float         top_p;         /* Top-P/nucleus sampling (default 0.95, 1.0 = disabled) */
	float         repeat_penalty; /* Repeat penalty (default 1.1, 1.0 = disabled) */
	int32_t       repeat_last_n; /* Tokens to consider for repeat penalty (default 64) */
	
	/* Feature flags */
	int           enable_embeddings; /* Enable embeddings mode (needed for qllm_embed(), default 0) */
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
 * Get the actual context size used.
 */
int
qllm_n_ctx(struct qllm_context *ctx);

/*
 * Set the current sequence ID for generation.
 */
void
qllm_set_seq(struct qllm_context *ctx, uint32_t seq_id);

/*
 * Set a GBNF grammar for constrained decoding.
 * Pass NULL to disable grammar.
 */
int
qllm_set_grammar(struct qllm_context *ctx, const char *grammar_str);

/*
 * Create a new sampler chain based on config.
 */
struct llama_sampler *
qllm_sampler_create(struct qllm_context *ctx, const struct qllm_config *cfg);

int
qllm_sampler_add_grammar(struct qllm_context *ctx,
			 struct llama_sampler *sampler,
			 const char *grammar_str);

void
qllm_sampler_free(struct llama_sampler *smpl);

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
	  struct llama_sampler *sampler,
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
