/* engine-test.c — standalone Phase 1 driver for the qllm engine.
 *
 * No axil, no HTTP: verifies the worker-thread engine produces correct
 * embeddings (OpenAI-shaped JSON, dim 896, finite) end to end, including the
 * self-pipe wake + completion drain path.
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

struct sample {
	int   job_id;
	int   err;
	char *payload;
	int   dim;
	int   finite;
	int   kind;
	int   ok;
	double max_abs;
	char  model_echo[BUFSIZ];
};

static struct sample g_samples[8];

static void
collect_sample(int idx, const struct qllm_engine_result *r)
{
	struct sample *s;
	int n;
	double mv;

	s = &g_samples[idx];
	s->job_id = r->job_id;
	s->kind = r->kind;
	s->err = r->err;
	s->payload = r->payload ? strdup(r->payload) : NULL;
	s->dim = 0;
	s->finite = 0;
	s->max_abs = 0.0;
	s->ok = 0;

	if (!s->payload || s->err != 0)
		return;

	{
		struct json_object *o, *data, *item, *emb, *model;
		int i;

		o = json_tokener_parse(s->payload);
		if (!o)
			return;

		if (json_object_object_get_ex(o, "data", &data) &&
		    json_object_array_length(data) == 1) {
			item = json_object_array_get_idx(data, 0);
			n = (int)json_object_array_length(
				json_object_object_get(item, "embedding"));
			emb = json_object_object_get(item, "embedding");
		} else {
			n = 0;
			emb = NULL;
		}

		if (n > 0) {
			mv = 0.0;
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
				s->finite = 1;
			s->max_abs = mv;
		}
		s->dim = n;
		s->ok = s->finite && s->dim == DIM_EXPECTED;

		if (json_object_object_get_ex(o, "model", &model))
			snprintf(s->model_echo, sizeof(s->model_echo), "%s",
				json_object_get_string(model));
		json_object_put(o);
	}
}

static void
on_result(void *ud, const struct qllm_engine_result *r)
{
	int *idx = ud;

	(void)r;
	collect_sample(*idx, r);
	++*idx;
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
	struct sample *s;
	int idx = 0, id = 0, i, attempts;

	if (argc != 2) {
		fprintf(stderr, "usage: %s /path/to/model.gguf\n", argv[0]);
		return 1;
	}
	model = argv[1];

	if (qllm_engine_init(model) != 0)
		return fail("engine init");

	/* --- single embed, block via self-pipe + drain --- */
	if (qllm_engine_submit(QE_JOB_EMBED, "puddle reflection", &idx, &id) != 0)
		return fail("submit puddle");
	if (qllm_engine_wait(120000) != 0)
		return fail("wait puddle");
	if (qllm_engine_poll(on_result, &idx) != 1)
		return fail("poll puddle");

	s = &g_samples[0];
	if (s->job_id != id || s->kind != QE_JOB_EMBED || s->err != 0)
		return fail("puddle result fields");
	if (!s->ok)
		return fail("puddle dim/finite");
	printf("puddle: dim=%d max|v|=%.3g finite=%d ok\n", s->dim, s->max_abs,
		!!s->finite);

	/* --- two queued back to back --- */
	if (qllm_engine_submit(QE_JOB_EMBED, "dog", &idx, &id) != 0)
		return fail("submit dog");
	if (qllm_engine_submit(QE_JOB_EMBED, "quantum", &idx, &id) != 0)
		return fail("submit quantum");

	for (attempts = 0; attempts < 4 && idx < 3; ++attempts) {
		if (qllm_engine_wait(120000) != 0)
			return fail("wait batch");
		(void)qllm_engine_poll(on_result, &idx);
	}
	if (idx != 3)
		return fail("batch results count");

	for (i = 1; i < 3; ++i) {
		s = &g_samples[i];
		if (s->kind != QE_JOB_EMBED || s->err != 0 || !s->ok) {
			fprintf(stderr, "FAIL: sample %d bad (err=%d dim=%d ok=%d)\n",
				i, s->err, s->dim, s->ok);
			return 1;
		}
		printf("%s: dim=%d max|v|=%.3g ok\n", i == 1 ? "dog" : "quantum",
			s->dim, s->max_abs);
	}

	if (strcmp(g_samples[0].model_echo, model) != 0)
		return fail("model name echo");

	qllm_engine_shutdown();

	for (i = 0; i < 3; ++i)
		free(g_samples[i].payload);

	printf("PASS: engine embed (%d/896 dims finite across %d samples)\n",
		DIM_EXPECTED, idx);
	return 0;
}