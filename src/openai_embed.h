#ifndef OPENAI_EMBED_H
#define OPENAI_EMBED_H

/*
 * OpenAI-compatible embeddings endpoint for qllmd.
 *
 * Registers the HTTP handler "POST:/v1/embeddings" that parses an
 * OpenAI-style request body, defers the response, and submits the embed
 * job to the engine worker. The completion is delivered later via
 * qllmd.c's loop-thread `drain` callback using axil_respond_defer_finish.
 */

/*
 * Register the /v1/embeddings route. model_path is accepted for API
 * consistency only; the model context lives in the engine worker.
 * Returns 0 on success, -1 on failure.
 */
int openai_embed_init(const char *model_path);

/*
 * No-op in the async design; the engine owns model teardown via
 * qllm_engine_shutdown().
 */
void openai_embed_shutdown(void);

#endif /* OPENAI_EMBED_H */
