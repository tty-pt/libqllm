/* openai_embed.c — OpenAI-compatible embeddings endpoint for qllmd. */

#include "openai_embed.h"

#include <ttypt/axil.h>
#include <ttypt/qllm.h>

#include <json-c/json.h>

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* Dedicated, lazily-created embed context, separate from the per-fd
 * chat contexts. axil is single-threaded, so no locking is needed and
 * this cannot race the chat paths. qllm_embed() re-inits the llama
 * context internally (stateless single-shot), so a single persistent
 * context is safe to reuse across requests. */
static struct qllm_context *g_embed_ctx;
static char g_model_path[BUFSIZ];

static struct qllm_context *
embed_ctx_ensure(void)
{
	struct qllm_config cfg;

	if (g_embed_ctx)
		return g_embed_ctx;

	if (!g_model_path[0])
		return NULL;

	cfg.model_path = g_model_path;
	cfg.n_ctx = 0;
	cfg.n_threads = 0;
	cfg.max_offload_bytes = 0;
	cfg.n_contexts = 1;

	g_embed_ctx = qllm_create(&cfg);
	return g_embed_ctx;
}

static void
embed_respond_error(socket_t fd, int code, const char *msg)
{
	struct json_object *obj;
	const char *s;

	obj = json_object_new_object();
	json_object_object_add(obj, "error",
	    json_object_new_string(msg ? msg : "internal error"));
	s = json_object_to_json_string(obj);
	axil_header_set(fd, "Content-Type", "application/json");
	axil_respond(fd, code, s);
	json_object_put(obj);
}

static int
handler_embeddings(socket_t cfd, char *body)
{
	struct json_object *req = NULL, *input_obj;
	const char *input;
	struct json_object *resp = NULL, *data, *item, *embed, *usage;
	float *vec = NULL;
	int ret;
	int dim = 0;
	int i;
	const char *s;

	if (!body || !body[0]) {
		embed_respond_error(cfd, 400, "empty request body");
		return 0;
	}

	if (embed_ctx_ensure() == NULL) {
		embed_respond_error(cfd, 500, "model not loaded");
		return 0;
	}

	req = json_tokener_parse(body);
	if (!req || !json_object_is_type(req, json_type_object)) {
		embed_respond_error(cfd, 400, "invalid JSON body");
		goto out;
	}

	input_obj = json_object_object_get(req, "input");
	if (!input_obj || !json_object_is_type(input_obj, json_type_string)) {
		embed_respond_error(cfd, 400,
		    "missing or invalid \"input\" (expected a string)");
		goto out;
	}

	input = json_object_get_string(input_obj);
	if (!input || !input[0]) {
		embed_respond_error(cfd, 400, "\"input\" must not be empty");
		goto out;
	}

	/* Provisional max dimension: allocate generously and let
	 * qllm_embed validate/report the true n_embd. */
	vec = calloc(4096, sizeof(*vec));
	if (!vec) {
		embed_respond_error(cfd, 500, "out of memory");
		goto out;
	}

	ret = qllm_embed(g_embed_ctx, input, vec, 4096);
	if (ret <= 0) {
		embed_respond_error(cfd, 500, "embedding failed");
		goto out;
	}
	dim = ret;

	resp = json_object_new_object();
	data = json_object_new_array();
	item = json_object_new_object();

	embed = json_object_new_array();
	for (i = 0; i < dim; ++i)
		json_object_array_add(embed, json_object_new_double(vec[i]));
	json_object_object_add(item, "object",
	    json_object_new_string("embedding"));
	json_object_object_add(item, "embedding", embed);
	json_object_object_add(item, "index", json_object_new_int(0));
	json_object_array_add(data, item);

	json_object_object_add(resp, "data", data);
	json_object_object_add(resp, "model",
	    json_object_new_string(g_model_path));
	json_object_object_add(resp, "object",
	    json_object_new_string("list"));

	usage = json_object_new_object();
	json_object_object_add(usage, "prompt_tokens", json_object_new_int(0));
	json_object_object_add(usage, "total_tokens", json_object_new_int(0));
	json_object_object_add(resp, "usage", usage);

	s = json_object_to_json_string(resp);
	axil_header_set(cfd, "Content-Type", "application/json");
	axil_respond(cfd, 200, s);

out:
	if (vec)
		free(vec);
	if (req)
		json_object_put(req);
	if (resp)
		json_object_put(resp);
	return 0;
}

int
openai_embed_init(const char *model_path)
{
	if (!model_path || !model_path[0])
		return -1;

	if (strlen(model_path) >= sizeof(g_model_path))
		return -1;

	snprintf(g_model_path, sizeof(g_model_path), "%s", model_path);
	axil_register_handler("POST:/v1/embeddings", handler_embeddings);
	return 0;
}

void
openai_embed_shutdown(void)
{
	if (g_embed_ctx) {
		qllm_free(g_embed_ctx);
		g_embed_ctx = NULL;
	}
}
