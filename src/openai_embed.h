#ifndef OPENAI_EMBED_H
#define OPENAI_EMBED_H

/*
 * OpenAI-compatible embeddings endpoint for qllmd.
 *
 * Registers the HTTP handler "POST:/v1/embeddings" that parses an
 * OpenAI-style request body and answers with the OpenAI embeddings
 * JSON shape, backed by qllm_embed().
 */

/*
 * Register the /v1/embeddings route and configure the shared embed
 * context. model_path is the GGUF model served by the daemon.
 * Returns 0 on success, -1 on failure.
 */
int openai_embed_init(const char *model_path);

/*
 * Tear down the shared embed context. Call at shutdown.
 */
void openai_embed_shutdown(void);

#endif /* OPENAI_EMBED_H */
