/* engine-test.c — standalone driver for the qllm engine.
 *
 * No axil, no HTTP: verifies the worker-thread engine produces correct
 * embeddings (OpenAI-shaped JSON, dim 896, finite), scripted chat sessions
 * (streaming chunks + full reply, history carry-over, reset, close), and
 * stateless oneshot chat (plain JSON + SSE delta stream). Exercises the
 * self-pipe wake + completion drain path end to end.
 *
 * Usage: engine-test /path/to/model.gguf
 * Exit:  0 on PASS, 1 on FAIL.
 */

#include "qllm-engine.h"

#include <json-c/json.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DIM_EXPECTED 896
#define RECS_MAX     2048

struct result_rec {
	int   job_id;
	int   kind;
	int   err;
	int   more;
	int   seq;
	char *payload;
	int   dim;
	int   finite;
	int   ok_embed;
	double max_abs;
};

static struct result_rec g_recs[RECS_MAX];
static int g_nrecs;
static int g_finals;

static void
on_result(void *ud, const struct qllm_engine_result *r)
{
	struct result_rec *rec;

	(void)ud;
	if (g_nrecs >= RECS_MAX)
		return;
	rec = &g_recs[g_nrecs++];
	rec->job_id = r->job_id;
	rec->kind = r->kind;
	rec->err = r->err;
	rec->more = r->more;
	rec->seq = r->seq;
	rec->payload = r->payload ? strdup(r->payload) : NULL;
	rec->dim = 0;
	rec->finite = 0;
	rec->ok_embed = 0;
	rec->max_abs = 0.0;
	if (!r->more)
		++g_finals;
}

/* Keep draining until `target_finals` jobs have delivered their final result. */
static int
drain_until(int target_finals)
{
	int attempts;

	for (attempts = 0; attempts < 4000 && g_finals < target_finals; ++attempts) {
		(void)qllm_engine_wait(2000);
		qllm_engine_poll(on_result, NULL);
	}
	return g_finals >= target_finals ? 0 : -1;
}

/* The final (more=0) result seen for a job, per FIFO result order. */
static struct result_rec *
rec_last(int job_id)
{
	int i;

	for (i = g_nrecs - 1; i >= 0; --i)
		if (g_recs[i].job_id == job_id)
			return &g_recs[i];
	return NULL;
}

/* Parse an embeddings payload; fill dim/finite/max_abs (like the old test). */
static void
analyze_embed(struct result_rec *rec)
{
	struct json_object *o, *data, *item, *emb, *model;
	int n = 0, i;
	double mv = 0.0;

	if (!rec->payload || rec->err != 0)
		return;

	o = json_tokener_parse(rec->payload);
	if (!o)
		return;

	if (json_object_object_get_ex(o, "data", &data) &&
	    json_object_array_length(data) == 1) {
		item = json_object_array_get_idx(data, 0);
		emb = json_object_object_get(item, "embedding");
		n = (int)json_object_array_length(emb);
	}

	if (n > 0) {
		for (i = 0; i < n; ++i) {
			double v = json_object_get_double(
				json_object_array_get_idx(emb, i));
			if (!isfinite(v)) {
				n = -1;
				break;
			}
			if (fabs(v) > mv)
				mv = fabs(v);
		}
		if (n > 0)
			rec->finite = 1;
		rec->max_abs = mv;
	}
	rec->dim = n;
	rec->ok_embed = rec->finite && rec->dim == DIM_EXPECTED;
	json_object_put(o);
}

static int
fail(const char *msg)
{
	fprintf(stderr, "FAIL: %s\n", msg);
	return 1;
}

int
main(int argc, char **argv)
{
	const char *model;
	struct result_rec *r;
	int job_id, target = 0, i, chunks, seq;

	if (argc != 2) {
		fprintf(stderr, "usage: %s /path/to/model.gguf\n", argv[0]);
		return 1;
	}
	model = argv[1];

	if (qllm_engine_init(model) != 0)
		return fail("engine init");

	/* --- single embed --- */
	if (qllm_engine_submit(QE_JOB_EMBED, "puddle reflection", NULL, &job_id) != 0)
		return fail("submit puddle");
	target++;
	if (drain_until(target) != 0)
		return fail("wait puddle");
	r = rec_last(job_id);
	if (!r || r->kind != QE_JOB_EMBED || r->err != 0)
		return fail("puddle result fields");
	analyze_embed(r);
	if (!r->ok_embed)
		return fail("puddle dim/finite");
	printf("puddle: dim=%d max|v|=%.3g finite=%d ok\n",
		r->dim, r->max_abs, !!r->finite);

	/* --- two queued back to back --- */
	if (qllm_engine_submit(QE_JOB_EMBED, "dog", NULL, &job_id) != 0)
		return fail("submit dog");
	target++;
	if (qllm_engine_submit(QE_JOB_EMBED, "quantum", NULL, &job_id) != 0)
		return fail("submit quantum");
	target++;
	if (drain_until(target) != 0)
		return fail("wait batch");
	for (i = 0; i < g_nrecs; ++i) {
		if (g_recs[i].kind != QE_JOB_EMBED)
			continue;
		analyze_embed(&g_recs[i]);
		if (!g_recs[i].ok_embed) {
			fprintf(stderr, "FAIL: embed sample bad (err=%d dim=%d ok=%d)\n",
				g_recs[i].err, g_recs[i].dim, g_recs[i].ok_embed);
			return 1;
		}
	}

	qllm_engine_set_system("You are a terse assistant.");

	/* --- scripted session: open, two turns, reset, fresh turn, close --- */
	if (qllm_engine_session_open(7, NULL, NULL, &job_id) != 0)
		return fail("session open submit");
	target++;
	if (drain_until(target) != 0)
		return fail("session open wait");
	r = rec_last(job_id);
	if (!r || r->kind != QE_JOB_OPEN || r->err != 0)
		return fail("session open result");

	if (qllm_engine_chat(7, "Say the word cherry.", NULL, &job_id) != 0)
		return fail("chat turn1 submit");
	target++;
	if (drain_until(target) != 0)
		return fail("chat turn1 wait");

	chunks = 0;
	seq = 0;
	for (i = 0; i < g_nrecs; ++i) {
		if (g_recs[i].job_id != job_id)
			continue;
		if (g_recs[i].more) {
			if (g_recs[i].seq != seq || g_recs[i].err != 0 ||
			    g_recs[i].kind != QE_JOB_CHAT || !g_recs[i].payload ||
			    !g_recs[i].payload[0]) {
				fprintf(stderr, "FAIL: turn1 chunk (seq=%d recseq=%d)\n",
					g_recs[i].seq, seq);
				return 1;
			}
			++seq;
			++chunks;
		}
	}
	if (chunks < 1)
		return fail("turn1 no streamed chunks");
	r = rec_last(job_id);
	if (!r || r->more || r->err != 0 || !r->payload || !r->payload[0])
		return fail("turn1 final result");
	if (r->seq != chunks)
		return fail("turn1 final seq");

	/* full reply must equal the concatenation of the streamed chunks */
	{
		size_t catlen = 0, off = 0;
		char *cat;
		int k;

		for (k = 0; k < g_nrecs; ++k)
			if (g_recs[k].job_id == job_id && g_recs[k].more &&
			    g_recs[k].payload)
				catlen += strlen(g_recs[k].payload);
		cat = malloc(catlen + 1);
		if (!cat)
			return fail("turn1 cat oom");
		cat[0] = '\0';
		for (k = 0; k < g_nrecs; ++k) {
			if (g_recs[k].job_id != job_id || !g_recs[k].more ||
			    !g_recs[k].payload)
				continue;
			memcpy(cat + off, g_recs[k].payload,
				strlen(g_recs[k].payload));
			off += strlen(g_recs[k].payload);
			cat[off] = '\0';
		}
		if (strcmp(cat, r->payload) != 0) {
			free(cat);
			return fail("turn1 chunks != full reply");
		}
		printf("turn1: %d chunks, seq ok, full reply (%zu bytes)\n",
			chunks, strlen(r->payload));
		free(cat);
	}

	/* second turn: history carry-over (model should recall the word) */
	if (qllm_engine_chat(7, "What word did I ask you to say?", NULL, &job_id) != 0)
		return fail("chat turn2 submit");
	target++;
	if (drain_until(target) != 0)
		return fail("chat turn2 wait");
	r = rec_last(job_id);
	if (!r || r->more || r->err != 0 || !r->payload)
		return fail("turn2 final result");
	if (!strstr(r->payload, "cherry"))
		return fail("turn2 missed history");
	printf("turn2: recalled history ('%s')\n", "cherry");

	if (qllm_engine_session_reset(7, NULL, &job_id) != 0)
		return fail("session reset submit");
	target++;
	if (drain_until(target) != 0)
		return fail("session reset wait");
	r = rec_last(job_id);
	if (!r || r->kind != QE_JOB_RESET || r->err != 0)
		return fail("session reset result");

	if (qllm_engine_chat(7, "Say: pleasant", NULL, &job_id) != 0)
		return fail("chat after reset submit");
	target++;
	if (drain_until(target) != 0)
		return fail("chat after reset wait");
	r = rec_last(job_id);
	if (!r || r->more || r->err != 0 || !r->payload || !r->payload[0])
		return fail("chat after reset final");
	printf("after-reset: ok (%zu bytes)\n", strlen(r->payload));

	if (qllm_engine_session_close(7, NULL, &job_id) != 0)
		return fail("session close submit");
	target++;
	if (drain_until(target) != 0)
		return fail("session close wait");
	r = rec_last(job_id);
	if (!r || r->kind != QE_JOB_CLOSE || r->err != 0)
		return fail("session close result");

	/* --- oneshot chat, non-streaming --- */
	if (qllm_engine_chat_oneshot(
	    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}",
	    NULL, &job_id) != 0)
		return fail("oneshot submit");
	target++;
	if (drain_until(target) != 0)
		return fail("oneshot wait");
	r = rec_last(job_id);
	if (!r || r->more || r->err != 0 || !r->payload)
		return fail("oneshot final result");

	{
		struct json_object *o, *c0, *msg, *fin, *content;
		const char *finish, *text;
		int nerr = 0;

		o = json_tokener_parse(r->payload);
		if (!o) {
			fprintf(stderr, "FAIL: oneshot payload not JSON\n");
			return 1;
		}
		if (strcmp(json_object_get_string(json_object_object_get(o, "object")),
		    "chat.completion") != 0)
			nerr++;
		c0 = json_object_array_get_idx(json_object_object_get(o, "choices"), 0);
		msg = json_object_object_get(c0, "message");
		fin = json_object_object_get(c0, "finish_reason");
		content = json_object_object_get(msg, "content");
		text = content ? json_object_get_string(content) : NULL;
		finish = fin ? json_object_get_string(fin) : NULL;
		if (!text || !text[0] || !finish || strcmp(finish, "stop") != 0)
			nerr++;
		json_object_put(o);
		if (nerr)
			return fail("oneshot JSON shape/stop");
		printf("oneshot(non-stream): object=%s finish=%s (%zu bytes)\n",
			"chat.completion", "stop", strlen(r->payload));
	}

	/* --- oneshot chat, streaming --- */
	if (qllm_engine_chat_oneshot(
	    "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":true}",
	    NULL, &job_id) != 0)
		return fail("oneshot-stream submit");
	target++;
	if (drain_until(target) != 0)
		return fail("oneshot-stream wait");

	chunks = 0;
	for (i = 0; i < g_nrecs; ++i) {
		struct json_object *o;
		const char *finish = NULL;

		if (g_recs[i].job_id != job_id)
			continue;
		if (g_recs[i].more) {
			if (g_recs[i].kind != QE_JOB_CHAT_ONESHOT ||
			    g_recs[i].err != 0 || !g_recs[i].payload) {
				fprintf(stderr, "FAIL: stream chunk %d\n", i);
				return 1;
			}
			o = json_tokener_parse(g_recs[i].payload);
			if (!o) {
				fprintf(stderr, "FAIL: stream chunk not JSON\n");
				return 1;
			}
			json_object_put(o);
			++chunks;
		} else {
			o = json_tokener_parse(g_recs[i].payload);
			if (!o) {
				fprintf(stderr, "FAIL: stream final not JSON\n");
				return 1;
			}
			{
				struct json_object *c0 = json_object_array_get_idx(
					json_object_object_get(o, "choices"), 0);
				struct json_object *fin = json_object_object_get(c0,
					"finish_reason");
				if (fin)
					finish = json_object_get_string(fin);
			}
			json_object_put(o);
			if (!finish || strcmp(finish, "stop") != 0) {
				fprintf(stderr, "FAIL: stream final finish_reason=%s\n",
					finish ? finish : "(none)");
				return 1;
			}
		}
	}
	if (chunks < 1)
		return fail("no streamed oneshot chunks");
	printf("oneshot(stream): %d delta chunks + stop\n", chunks);

	qllm_engine_shutdown();

	for (i = 0; i < g_nrecs; ++i)
		free(g_recs[i].payload);

	printf("PASS: engine embed/chat/oneshot (%d results, %d final)\n",
		g_nrecs, g_finals);
	return 0;
}