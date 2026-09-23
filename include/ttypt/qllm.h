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

/* Opaque llama sampler (forward-declared; see llama.h for the real type). */
struct llama_sampler;

/*
 * Configuration structure for creating a QLLM context.
 * All fields optional except model_path.
 */
struct qllm_config {
	const char   *model_path; /* Required */
	int32_t       n_ctx;      /* Context size (0 = read from model GGUF, else 2048) */
	int32_t       n_threads;  /* Number of CPU threads (default: half of CPUs) */
	uint32_t      max_offload_bytes; /* Max byte offload */
	int32_t      n_contexts; /* How many contexts to account for */
	int32_t       n_gpu_layers; /* >0 force this many GPU layers (-g), 0 = auto */
	float         temperature;    /* Sampling temperature (default 0.7, 0.0 = greedy) */
	int32_t       top_k;         /* Top-k sampling (default 40, 0 = disabled) */
	float         top_p;         /* Nucleus sampling (default 0.95, 1.0 = disabled) */
	float         repeat_penalty; /* Repetition penalty (default 1.1, 1.0 = disabled) */
	int32_t       repeat_last_n;  /* Penalty lookback window (default 64) */
	int           enable_embeddings; /* Enable embeddings + mean pooling for qllm_embed() */
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
 * Get the actual context size used (returns 0 if ctx invalid).
 */
int
qllm_n_ctx(struct qllm_context *ctx);

/*
 * Set the current sequence ID for generation (multi-sequence support).
 */
void
qllm_set_seq(struct qllm_context *ctx, uint32_t seq_id);

/*
 * Set a GBNF grammar for constrained decoding.
 * Pass NULL to disable grammar. Returns 0 on success, -1 on error.
 */
int
qllm_set_grammar(struct qllm_context *ctx, const char *grammar_str);

/*
 * Create a new sampler chain (eos_bias + penalties + top_k/top_p + temp + dist)
 * based on config. Returns NULL on failure.
 */
struct llama_sampler *
qllm_sampler_create(struct qllm_context *ctx,
		    const struct qllm_config *cfg);

/*
 * Append a GBNF grammar sampler to an existing sampler chain.
 * Returns 0 on success, -1 on error.
 */
int
qllm_sampler_add_grammar(struct qllm_context *ctx,
			 struct llama_sampler *sampler,
			 const char *grammar_str);

/*
 * Free a sampler chain returned by qllm_sampler_create()/
 * qllm_sampler_add_grammar(). Safe to call with NULL.
 */
void
qllm_sampler_free(struct llama_sampler *smpl);

/*
 * Generate the next token as text.
 *
 * `sampler` may be NULL, in which case the context's internal sampler chain
 * is used.
 *
 * Returns:
 *   >0  number of bytes written to 'out' (UTF-8, NUL-terminated)
 *    0  end of generation (EOS/EOG)
 *   <0  error
 */
int
qllm_next(struct qllm_context *ctx,
	  struct llama_sampler *sampler,
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
 * Reset the context's generation state: clears the KV cache, keeps the
 * loaded model and context alive.
 */
void
qllm_reset(struct qllm_context *ctx);

/*
 * Mark the start of the protected region of the KV cache. Call before
 * priming a prompt so that qllm_compress() keeps this region (the current
 * prompt + response) while evicting older tokens.
 */
void
qllm_anchor_start(struct qllm_context *ctx);

/*
 * Mark the end of the protected region of the KV cache. Call after
 * generation completes.
 */
void
qllm_anchor_end(struct qllm_context *ctx);

/*
 * Sliding-window compression: shrink the KV sequence to `limit` tokens,
 * preferring to evict the oldest prefix and the tail beyond the anchored
 * region. limit == 0 clears the whole sequence.
 */
void
qllm_compress(struct qllm_context *ctx,
	      uint32_t limit);

/*
 * Configure the EOS-bias sampler. After `start_tokens` have been generated
 * the sampler ramps the end-of-turn token's logit up to `max_bias`, so long
 * generations terminate on their own.
 */
void
qllm_set_eos_bias(struct qllm_context *ctx,
		  int32_t start_tokens,
		  float max_bias);

#ifdef __cplusplus
}
#endif

#endif /* QLLM_H */
