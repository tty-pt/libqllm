/* qllm-engine.c — worker-thread LLM engine (see qllm-engine.h). */

#include "qllm-engine.h"

#include <ttypt/qllm.h>

#include <json-c/json.h>

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define QE_JOB_MAX 64
#define QE_VEC_MAX 4096

struct qllm_engine_job {
	int   kind;
	char *input;  /* owned by engine; freed after processing */
	void *ud;
	int   job_id;
};

struct qllm_engine {
	char model_path[BUFSIZ];

	int   running; /* init done, worker alive */
	int   quit;    /* shutdown requested */

	pthread_t       worker;
	pthread_mutex_t job_mutex;
	pthread_cond_t  job_cond;
	struct qllm_engine_job jobs[QE_JOB_MAX];
	int job_head, job_tail, job_count;

	pthread_mutex_t comp_mutex;
	struct qllm_engine_result comp[QE_JOB_MAX];
	int comp_head, comp_tail, comp_count;

	/* self-pipe: worker writes a byte to wake the consumer */
	int pipe_r, pipe_w;

	int next_job_id;

	struct qllm_context *embed_ctx; /* lazily built on the worker thread */
};

static struct qllm_engine g_engine;

static int
engine_running(void)
{
	return g_engine.running && !g_engine.quit;
}

static int
completion_push(const struct qllm_engine_result *r)
{
	int i;

	pthread_mutex_lock(&g_engine.comp_mutex);
	if (g_engine.comp_count >= QE_JOB_MAX) {
		pthread_mutex_unlock(&g_engine.comp_mutex);
		return -1;
	}

	i = g_engine.comp_tail;
	g_engine.comp_tail = (g_engine.comp_tail + 1) % QE_JOB_MAX;
	g_engine.comp[i] = *r;
	++g_engine.comp_count;
	pthread_mutex_unlock(&g_engine.comp_mutex);

	/* Wake hint only; EAGAIN is fine (a byte is already buffered and the
	 * completion queue holds the data). */
	if (write(g_engine.pipe_w, "\0", 1) < 0 && errno != EAGAIN)
		return -1;
	return 0;
}

/* Build an OpenAI-compatible embeddings payload. Mirrors the shape produced by
 * openai_embed.c; keep in sync until Phase 3 removes that copy. */
static char *
embed_build_payload(const float *vec, int dim, const char *model_path)
{
	struct json_object *resp, *data, *item, *embed, *usage;
	int i;
	const char *s;

	resp = json_object_new_object();
	data = json_object_new_array();
	item = json_object_new_object();
	embed = json_object_new_array();

	for (i = 0; i < dim; ++i)
		json_object_array_add(embed, json_object_new_double(vec[i]));
	json_object_object_add(item, "object", json_object_new_string("embedding"));
	json_object_object_add(item, "embedding", embed);
	json_object_object_add(item, "index", json_object_new_int(0));
	json_object_array_add(data, item);

	json_object_object_add(resp, "data", data);
	json_object_object_add(resp, "model",
		json_object_new_string(model_path));
	json_object_object_add(resp, "object", json_object_new_string("list"));

	usage = json_object_new_object();
	json_object_object_add(usage, "prompt_tokens", json_object_new_int(0));
	json_object_object_add(usage, "total_tokens", json_object_new_int(0));
	json_object_object_add(resp, "usage", usage);

	s = json_object_to_json_string(resp);
	if (!s) {
		json_object_put(resp);
		return NULL;
	}

	{
		char *copy = strdup(s);
		json_object_put(resp);
		return copy;
	}
}

static int
run_embed(struct qllm_engine_job *job, struct qllm_engine_result *out)
{
	float *vec = NULL;
	int dim = 0;

	memset(out, 0, sizeof(*out));
	out->job_id = job->job_id;
	out->kind = job->kind;
	out->ud = job->ud;

	if (!g_engine.embed_ctx) {
		struct qllm_config cfg;

		memset(&cfg, 0, sizeof(cfg));
		cfg.model_path = g_engine.model_path;
		cfg.n_ctx = 0;
		cfg.n_threads = 0;
		cfg.max_offload_bytes = 0;
		cfg.n_contexts = 1;

		g_engine.embed_ctx = qllm_create(&cfg);
		if (!g_engine.embed_ctx) {
			out->err = -2;
			return -1;
		}
	}

	vec = calloc(QE_VEC_MAX, sizeof(*vec));
	if (!vec) {
		out->err = -3;
		return -1;
	}

	dim = qllm_embed(g_engine.embed_ctx, job->input, vec, QE_VEC_MAX);
	if (dim <= 0) {
		free(vec);
		out->err = -4;
		return -1;
	}

	out->payload = embed_build_payload(vec, dim, g_engine.model_path);
	free(vec);

	if (!out->payload) {
		out->err = -5;
		return -1;
	}
	return 0;
}

static void *
worker_main(void *arg)
{
	(void)arg;

	for (;;) {
		struct qllm_engine_job job;
		struct qllm_engine_result res;

		pthread_mutex_lock(&g_engine.job_mutex);
		while (g_engine.job_count == 0 && !g_engine.quit)
			pthread_cond_wait(&g_engine.job_cond, &g_engine.job_mutex);
		if (g_engine.quit) {
			pthread_mutex_unlock(&g_engine.job_mutex);
			break;
		}

		job = g_engine.jobs[g_engine.job_head];
		g_engine.job_head = (g_engine.job_head + 1) % QE_JOB_MAX;
		--g_engine.job_count;
		pthread_mutex_unlock(&g_engine.job_mutex);

		if (job.kind == QE_JOB_EMBED)
			run_embed(&job, &res);
		else {
			memset(&res, 0, sizeof(res));
			res.job_id = job.job_id;
			res.kind = job.kind;
			res.ud = job.ud;
			res.err = -1;
		}

		free(job.input);

		if (completion_push(&res) < 0)
			free(res.payload);
	}

	pthread_mutex_lock(&g_engine.comp_mutex);
	/* ensure any final wake is delivered even if quitting with pending results */
	(void)write(g_engine.pipe_w, "\0", 1);
	pthread_mutex_unlock(&g_engine.comp_mutex);

	return NULL;
}

int
qllm_engine_init(const char *model_path)
{
	if (g_engine.running)
		return -1;
	if (!model_path || !model_path[0])
		return -1;

	memset(&g_engine, 0, sizeof(g_engine));
	snprintf(g_engine.model_path, sizeof(g_engine.model_path), "%s",
		model_path);

	if (pthread_mutex_init(&g_engine.job_mutex, NULL) != 0)
		return -1;
	if (pthread_cond_init(&g_engine.job_cond, NULL) != 0) {
		pthread_mutex_destroy(&g_engine.job_mutex);
		return -1;
	}
	if (pthread_mutex_init(&g_engine.comp_mutex, NULL) != 0) {
		pthread_cond_destroy(&g_engine.job_cond);
		pthread_mutex_destroy(&g_engine.job_mutex);
		return -1;
	}

	{
		int pfd[2];

		if (pipe(pfd) == -1) {
			pthread_mutex_destroy(&g_engine.comp_mutex);
			pthread_cond_destroy(&g_engine.job_cond);
			pthread_mutex_destroy(&g_engine.job_mutex);
			return -1;
		}
		g_engine.pipe_r = pfd[0];
		g_engine.pipe_w = pfd[1];
	}
	fcntl(g_engine.pipe_w, F_SETFL, O_NONBLOCK);

	if (pthread_create(&g_engine.worker, NULL, worker_main, NULL) != 0) {
		close(g_engine.pipe_r);
		close(g_engine.pipe_w);
		pthread_mutex_destroy(&g_engine.comp_mutex);
		pthread_cond_destroy(&g_engine.job_cond);
		pthread_mutex_destroy(&g_engine.job_mutex);
		return -1;
	}

	g_engine.running = 1;
	g_engine.next_job_id = 1;
	return 0;
}

void
qllm_engine_shutdown(void)
{
	if (!g_engine.running)
		return;

	pthread_mutex_lock(&g_engine.job_mutex);
	g_engine.quit = 1;
	pthread_cond_broadcast(&g_engine.job_cond);
	pthread_mutex_unlock(&g_engine.job_mutex);

	pthread_join(g_engine.worker, NULL);

	if (g_engine.embed_ctx) {
		qllm_free(g_engine.embed_ctx);
		g_engine.embed_ctx = NULL;
	}

	/* free anything left in the completion queue */
	while (g_engine.comp_count > 0) {
		int i = g_engine.comp_head;

		free(g_engine.comp[i].payload);
		g_engine.comp_head = (i + 1) % QE_JOB_MAX;
		--g_engine.comp_count;
	}

	close(g_engine.pipe_r);
	close(g_engine.pipe_w);

	pthread_mutex_destroy(&g_engine.comp_mutex);
	pthread_cond_destroy(&g_engine.job_cond);
	pthread_mutex_destroy(&g_engine.job_mutex);

	memset(&g_engine, 0, sizeof(g_engine));
}

int
qllm_engine_submit(int kind, const char *input, void *ud, int *job_id_out)
{
	struct qllm_engine_job *job;
	int i, id;

	if (!engine_running())
		return -1;
	if (!input)
		return -1;

	pthread_mutex_lock(&g_engine.job_mutex);
	if (g_engine.job_count >= QE_JOB_MAX) {
		pthread_mutex_unlock(&g_engine.job_mutex);
		return -1;
	}

	i = g_engine.job_tail;
	g_engine.job_tail = (g_engine.job_tail + 1) % QE_JOB_MAX;
	job = &g_engine.jobs[i];
	job->kind = kind;
	job->input = strdup(input);
	if (!job->input) {
		pthread_mutex_unlock(&g_engine.job_mutex);
		return -1;
	}
	job->ud = ud;
	id = job->job_id = g_engine.next_job_id++;
	++g_engine.job_count;
	pthread_cond_signal(&g_engine.job_cond);
	pthread_mutex_unlock(&g_engine.job_mutex);

	if (job_id_out)
		*job_id_out = id;
	return 0;
}

int
qllm_engine_poll(qllm_engine_result_cb cb, void *cbud)
{
	struct qllm_engine_result pending[QE_JOB_MAX];
	int n = 0, i, r;

	if (!cb)
		return 0;

	pthread_mutex_lock(&g_engine.comp_mutex);
	while (g_engine.comp_count > 0) {
		i = g_engine.comp_head;
		pending[n++] = g_engine.comp[i];
		g_engine.comp_head = (i + 1) % QE_JOB_MAX;
		--g_engine.comp_count;
	}
	pthread_mutex_unlock(&g_engine.comp_mutex);

	for (r = 0; r < n; ++r) {
		cb(cbud, &pending[r]);
		free(pending[r].payload);
	}
	return n;
}

int
qllm_engine_wake_fd(void)
{
	if (!g_engine.running)
		return -1;
	return g_engine.pipe_r;
}

int
qllm_engine_wait(int timeout_ms)
{
	struct pollfd pfd;
	char b;
	ssize_t r;

	if (!g_engine.running)
		return -1;

	pfd.fd = g_engine.pipe_r;
	pfd.events = POLLIN;
	pfd.revents = 0;

	r = poll(&pfd, 1, timeout_ms);
	if (r <= 0)
		return -1;

	r = read(g_engine.pipe_r, &b, 1);
	if (r < 0)
		return -1;
	return 0;
}