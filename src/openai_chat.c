/* openai_chat.c — OpenAI-style /v1/chat/completions endpoint for qllmd.
 *
 * The body is handed to the engine verbatim (qllm_engine_chat_oneshot
 * validates "messages" and honors "stream"); the loop-thread drain turns the
 * resulting stream of results into either SSE (stream:true) or a single
 * chat.completion JSON response.
 */

#include "openai_chat.h"
#include "qllm-engine.h"
#include "qllm-pending.h"

#include <ttypt/axil.h>

#include <json-c/json.h>

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static void
chat_respond_error(socket_t fd, int code, const char *msg)
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
handler_chat_completions(socket_t cfd, char *body)
{
	struct json_object *req = NULL, *messages, *item, *role, *content, *sm;
	size_t n;
	int stream = 0;
	int i;
	void *handle;

	if (!body || !body[0]) {
		chat_respond_error(cfd, 400, "empty request body");
		return 0;
	}

	req = json_tokener_parse(body);
	if (!req || !json_object_is_type(req, json_type_object)) {
		chat_respond_error(cfd, 400, "invalid JSON body");
		goto out;
	}

	messages = json_object_object_get(req, "messages");
	if (!messages || !json_object_is_type(messages, json_type_array) ||
	    json_object_array_length(messages) == 0) {
		chat_respond_error(cfd, 400,
		    "missing or invalid \"messages\" (expected a non-empty array)");
		goto out;
	}

	n = json_object_array_length(messages);
	if (n > 64) {
		chat_respond_error(cfd, 400, "too many messages (max 64)");
		goto out;
	}

	for (i = 0; i < (int)n; ++i) {
		item = json_object_array_get_idx(messages, i);
		role = item ? json_object_object_get(item, "role") : NULL;
		content = item ? json_object_object_get(item, "content") : NULL;
		if (!item || !json_object_is_type(item, json_type_object) ||
		    !role || !json_object_is_type(role, json_type_string) ||
		    !content || !json_object_is_type(content, json_type_string)) {
			chat_respond_error(cfd, 400,
			    "each message needs string \"role\" and \"content\"");
			goto out;
		}
	}

	sm = json_object_object_get(req, "stream");
	if (sm)
		stream = json_object_get_boolean(sm) != 0;

	/* Send status + headers now, stream/complete the body later from the
	 * loop thread when the engine's worker finishes (`drain` in qllmd.c). */
	if (stream) {
		axil_header_set(cfd, "Content-Type", "text/event-stream");
		axil_header_set(cfd, "Cache-Control", "no-cache");
	} else {
		axil_header_set(cfd, "Content-Type", "application/json");
	}
	handle = axil_respond_defer(cfd, 200);
	if (!handle) {
		chat_respond_error(cfd, 503, "unavailable");
		goto out;
	}

	if (qllm_engine_chat_oneshot(body, make_pending(handle, stream, 0), NULL) != 0) {
		axil_respond_defer_abort(handle);
		chat_respond_error(cfd, 503, "engine busy");
		goto out;
	}

out:
	if (req)
		json_object_put(req);
	return 0;
}

int
openai_chat_init(const char *model_path)
{
	if (!model_path || !model_path[0])
		return -1;

	axil_register_handler("POST:/v1/chat/completions", handler_chat_completions);
	return 0;
}

void
openai_chat_shutdown(void)
{
}