/* libqllm.c */

#include "./../include/ttypt/qllm.h"

#include <ctype.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <llama.h>
#include <gguf.h>

#include <ttypt/qsys.h>
#include <ttypt/qmap.h>

struct qllm_context {
	struct llama_model	*model;
	struct llama_context	*ctx;
	struct llama_sampler	*sampler;
	struct llama_context_params params; /* <-- add this */
	const struct llama_vocab *vocab;

	int32_t			 n_embd;
	int32_t			 max_tokens;
	llama_pos		 cur_pos;

	llama_token		*token_buf;
	llama_seq_id		*seq_ids;
};

static int qllm_backend_inited;
static uint32_t qm_model, model_hd;

/* Initialize llama backend exactly once. */
__attribute__((constructor)) void 
qllm_init(void)
{
	qm_model = qmap_reg(sizeof(struct llama_model *));
	model_hd = qmap_open(NULL, NULL, QM_STR, qm_model, 0, 0);
	llama_backend_init();
	qllm_backend_inited = 1;
}

/* Small helper to decode a batch of tokens at the current position. */
static int
qllm_decode_tokens(struct qllm_context *qctx,
		   const llama_token *tokens,
		   int32_t n_tokens)
{
	struct llama_batch batch;
	int32_t i;

	if (!qctx || !qctx->ctx || !tokens || n_tokens <= 0)
		return -1;

	if (n_tokens > qctx->max_tokens)
		return -1;

	batch = llama_batch_init(n_tokens, 0, 1);
	batch.n_tokens = n_tokens;

	for (i = 0; i < n_tokens; ++i) {
		batch.token[i] = tokens[i];
		batch.pos[i] = qctx->cur_pos + i;
		batch.n_seq_id[i] = 1;
		qctx->seq_ids[i] = 0;
		batch.seq_id[i] = &qctx->seq_ids[i];
		batch.logits[i] = (i == n_tokens - 1);
	}

	if (llama_decode(qctx->ctx, batch) != 0)
		return -1;

	qctx->cur_pos += n_tokens;
	return 0;
}

extern void
qllm_backend_mem_check(int gpu, size_t *free_b, size_t *total_b);

static int
auto_ngl(const char *path, int gpu, uint32_t n_ctx, uint32_t max_offload_bytes,
	 int n_layers, int n_embd, int n_contexts)
{
	size_t free_b, total_b;
	struct gguf_init_params ip = { .no_alloc = true };
	struct gguf_context *ctx;
	size_t *layer_sizes;
	size_t usable;
	size_t kv_size_per_ctx;
	size_t workspace_per_ctx;
	size_t this_ctx_cost;
	size_t per_other_ctx_cost = 0;
	size_t other_ctx_cost = 0;
	size_t system_overhead;
	size_t reserve;
	size_t used;
	size_t largest_layer;
	int n_tensors;
	int ngl;
	int i;

	if (n_contexts <= 0)
		n_contexts = 1;

	qllm_backend_mem_check(gpu, &free_b, &total_b);
	if (!total_b || !free_b)
		return 0;

	usable = free_b;
	if (!usable)
		return 0;

	ctx = gguf_init_from_file(path, ip);
	if (!ctx)
		return 0;

	n_tensors = (int)gguf_get_n_tensors(ctx);
	if (n_layers <= 0 || n_embd <= 0 || n_tensors <= 0) {
		gguf_free(ctx);
		return 0;
	}

	layer_sizes = calloc((size_t)n_layers, sizeof(*layer_sizes));
	if (!layer_sizes) {
		gguf_free(ctx);
		return 0;
	}

	for (i = 0; i < n_tensors; i++) {
		const char *name = gguf_get_tensor_name(ctx, i);
		const char *p;
		long layer;

		if (!name)
			continue;

		p = strstr(name, "blk.");
		if (!p) p = strstr(name, "layers.");
		if (!p) p = strstr(name, "block.");
		if (!p)
			continue;

		while (*p && !isdigit((unsigned char)*p))
			p++;
		if (!isdigit((unsigned char)*p))
			continue;

		layer = strtol(p, NULL, 10);
		if (layer < 0 || layer >= n_layers)
			continue;

		layer_sizes[layer] += gguf_get_tensor_size(ctx, i);
	}

	gguf_free(ctx);

	largest_layer = 0;
	for (i = 0; i < n_layers; i++)
		if (layer_sizes[i] > largest_layer)
			largest_layer = layer_sizes[i];

	workspace_per_ctx = largest_layer + (64 * 1024 * 1024);

	/* 2 * n_ctx * n_embd * n_layers * sizeof(f16) == 4 * n_ctx * n_embd * n_layers */
	kv_size_per_ctx =
	    (size_t)n_ctx *
	    (size_t)n_embd *
	    25ULL *
	    (size_t)n_layers;

	system_overhead = kv_size_per_ctx / 10;

	this_ctx_cost = kv_size_per_ctx + workspace_per_ctx;

	if (n_contexts > 1) {
		per_other_ctx_cost = this_ctx_cost + (this_ctx_cost >> 1);
		other_ctx_cost = per_other_ctx_cost * (size_t)(n_contexts - 1);
	}

	/* reserva fixa para driver/SO */
	reserve = 128 * 1024 * 1024ULL;

	if (usable <= reserve + this_ctx_cost + other_ctx_cost + system_overhead) {
		free(layer_sizes);
		return 0;
	}

	usable -= reserve;
	usable -= this_ctx_cost + other_ctx_cost + system_overhead;

	if (max_offload_bytes > 0 && usable > max_offload_bytes)
		usable = max_offload_bytes;

	used = 0;
	ngl = 0;

	for (i = 0; i < n_layers; i++) {
		size_t weight = layer_sizes[i];
		size_t need   = weight * 5 / 2;

		if (used + need > usable)
			break;

		used += need;
		ngl++;
	}

	free(layer_sizes);
	return ngl;
}

struct llama_model *model_load(
		const char *path,
		int32_t n_ctx,
		uint32_t ngl_max,
		int32_t n_contexts)
{
	struct llama_model_params model_params;
	struct llama_model ** model_r, *model;
	int n_layers, n_embd, ngl;

	model_r = (struct llama_model **) qmap_get(model_hd, path);
	if (model_r)
		return *model_r;

	if (!n_contexts)
		n_contexts = 1;

	model_params = llama_model_default_params();

	model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;

	model_params.n_gpu_layers = 0;
	if (!(model = llama_model_load_from_file(
			path,
			model_params)))
		return NULL;

	n_layers = llama_model_n_layer(model);
	n_embd = llama_model_n_embd(model);
	llama_model_free(model);
	ngl = auto_ngl(path, 0, n_ctx, ngl_max, n_layers, n_embd, n_contexts);

	if (ngl > 0)
		model_params.n_gpu_layers = ngl;
	else
		model_params.n_gpu_layers = 0;

	if (!(model = llama_model_load_from_file(
			path,
			model_params)))
		return NULL;

	qmap_put(model_hd, path, &model);
	return model;
}

void
qllm_position(struct qllm_context *ctx,
	uint64_t pos)
{
	llama_pos pos_min, pos_max;
	llama_memory_t mem;

	mem = llama_get_memory(ctx->ctx);

	pos_min = llama_memory_seq_pos_min(mem, 0);
	pos_max = llama_memory_seq_pos_max(mem, 0);

	if (pos_min < 0 || pos_max < 0) {
		ctx->cur_pos = 0;
		return;
	}

	/* remove tokens antes de new_pos */
	llama_memory_seq_rm(mem, 0, pos_min, pos);

	/* desloca os restantes para começar em 0 */
	llama_memory_seq_add(mem, 0, pos, -1, -pos);
	ctx->cur_pos = pos_max - pos + 1;
}

struct qllm_context *
qllm_create(const struct qllm_config *cfg)
{
	struct qllm_context *qctx;
	struct llama_context_params ctx_params;
	struct llama_sampler_chain_params chain_params;
	int32_t n_threads;

	if (!cfg || !cfg->model_path)
		return NULL;

	ctx_params = llama_context_default_params();
	chain_params = llama_sampler_chain_default_params();

	if (cfg->n_ctx > 0)
		ctx_params.n_ctx = (uint32_t) cfg->n_ctx;
	else
		ctx_params.n_ctx = 512;

	ctx_params.n_batch = ctx_params.n_ctx;
	ctx_params.n_ubatch = 0;
	ctx_params.n_seq_max = 1;

	/* Enable embeddings and mean pooling so qllm_embed() works. */
	ctx_params.embeddings = true;
	ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

	if (cfg->n_threads > 0) {
		n_threads = cfg->n_threads;
	} else {
		long ncpu;

		ncpu = sysconf(_SC_NPROCESSORS_ONLN);
		if (ncpu <= 0)
			n_threads = 1;
		else
			n_threads = (int32_t)(ncpu > 1 ? ncpu / 2 : 1);
	}

	ctx_params.n_threads = n_threads;
	ctx_params.n_threads_batch = n_threads;

	qctx = calloc(1, sizeof(*qctx));
	if (!qctx)
		return NULL;

	qctx->max_tokens = (int32_t)ctx_params.n_ctx;
	qctx->params = ctx_params;	/* <-- important: save params */

	qctx->model = model_load(cfg->model_path, ctx_params.n_ctx, cfg->max_offload_bytes, cfg->n_contexts);

	if (!qctx->model)
		goto fail;

	qctx->ctx = llama_init_from_model(qctx->model, ctx_params);
	if (!qctx->ctx)
		goto fail;

	qctx->vocab = llama_model_get_vocab(qctx->model);
	qctx->n_embd = llama_model_n_embd(qctx->model);

	qctx->sampler = llama_sampler_chain_init(chain_params);
	if (!qctx->sampler)
		goto fail;

	llama_sampler_chain_add(qctx->sampler,
	    llama_sampler_init_greedy());

	qctx->token_buf = calloc((size_t)qctx->max_tokens,
	    sizeof(*qctx->token_buf));
	qctx->seq_ids = calloc((size_t)qctx->max_tokens,
	    sizeof(*qctx->seq_ids));
	if (!qctx->token_buf || !qctx->seq_ids)
		goto fail;

	qctx->cur_pos = 0;

	return qctx;

fail:
	qllm_free(qctx);
	return NULL;
}

void
qllm_free(struct qllm_context *qctx)
{
	if (!qctx)
		return;

	if (qctx->sampler)
		llama_sampler_free(qctx->sampler);
	if (qctx->ctx)
		llama_free(qctx->ctx);
	if (qctx->model)
		llama_model_free(qctx->model);

	free(qctx->token_buf);
	free(qctx->seq_ids);

	free(qctx);
}

/* Internal streaming helper: runs generation and calls cb() for each piece. */
static int
qllm_generate_stream_internal(struct qllm_context *qctx,
			      const char *prompt,
			      qllm_token_cb cb,
			      void *user)
{
	int32_t n_prompt;
	llama_token tok;
	char piece[256];
	int n_piece;
	int32_t step;
	const int32_t max_gen = qctx->max_tokens;

	if (!qctx || !prompt || !cb)
		return -1;

	llama_free(qctx->ctx);
	qctx->ctx = llama_init_from_model(qctx->model, qctx->params);

	qctx->cur_pos = 0;

	n_prompt = llama_tokenize(qctx->vocab,
				  prompt,
				  (int32_t) strlen(prompt),
				  qctx->token_buf,
				  qctx->max_tokens,
				  true,
				  true);
	if (n_prompt < 0)
		return -1;

	if (n_prompt == 0)
		return 0;

	if (qllm_decode_tokens(qctx, qctx->token_buf, n_prompt) != 0)
		return -1;

	for (step = 0; step < max_gen; ++step) {
		tok = llama_sampler_sample(qctx->sampler, qctx->ctx, -1);
		llama_sampler_accept(qctx->sampler, tok);

		if (llama_vocab_is_eog(qctx->vocab, tok))
			break;

		qctx->token_buf[0] = tok;
		if (qllm_decode_tokens(qctx, qctx->token_buf, 1) != 0)
			break;

		memset(piece, 0, sizeof(piece));
		n_piece = llama_token_to_piece(qctx->vocab,
					       tok,
					       piece,
					       (int) sizeof(piece),
					       false,
					       true);
		if (n_piece <= 0)
			continue;

		cb(user, piece, (size_t) n_piece);
	}

	return 0;
}

int
qllm_generate_stream(struct qllm_context *qctx,
		     const char *prompt,
		     qllm_token_cb cb,
		     void *user)
{
	if (!qctx || !prompt || !cb)
		return -1;

	return qllm_generate_stream_internal(qctx, prompt, cb, user);
}

/* Accumulator used by qllm_generate() to build a string. */
struct qllm_accum {
	char	*buf;
	size_t	 cap;
	size_t	 len;
};

static void
qllm_accum_cb(void *user, const char *chunk, size_t len)
{
	struct qllm_accum *acc = user;
	size_t n;

	if (!acc || !acc->buf || acc->cap == 0)
		return;

	if (acc->len >= acc->cap)
		return;

	n = len;
	if (acc->len + n >= acc->cap) {
		if (acc->cap <= acc->len + 1)
			return;
		n = acc->cap - acc->len - 1;
	}

	if (!n)
		return;

	memcpy(acc->buf + acc->len, chunk, n);
	acc->len += n;
	acc->buf[acc->len] = '\0';
}

long
qllm_generate(struct qllm_context *qctx,
	      const char *prompt,
	      char *out,
	      size_t out_size)
{
	struct qllm_accum acc;
	int ret;

	if (!qctx || !prompt || !out || out_size == 0)
		return -1;

	out[0] = '\0';

	acc.buf = out;
	acc.cap = out_size;
	acc.len = 0;

	ret = qllm_generate_stream_internal(qctx,
					    prompt,
					    qllm_accum_cb,
					    &acc);
	if (ret != 0)
		return -1;

	return (ssize_t) acc.len;
}

/*
 * qllm_embed — return a single embedding vector for the whole text.
 * Returns number of dims on success, < 0 on error.
 */
int
qllm_embed(struct qllm_context *qctx,
	   const char *text,
	   float *out,
	   size_t out_dim)
{
	int32_t n_tokens;
	const float *embd;
	int32_t i;

	if (!qctx || !text || !out)
		return -1;

	llama_free(qctx->ctx);
	qctx->ctx = llama_init_from_model(qctx->model, qctx->params);
	qctx->cur_pos = 0;

	n_tokens = llama_tokenize(qctx->vocab,
				  text,
				  (int32_t) strlen(text),
				  qctx->token_buf,
				  qctx->max_tokens,
				  true,
				  true);
	if (n_tokens < 0)
		return -1;

	if (n_tokens == 0)
		return -1;

	if (qllm_decode_tokens(qctx, qctx->token_buf, n_tokens) != 0)
		return -1;

	embd = llama_get_embeddings(qctx->ctx);
	if (!embd)
		return -1;

	if (out_dim < (size_t) qctx->n_embd)
		return -1;

	for (i = 0; i < qctx->n_embd; ++i)
		out[i] = embd[i];

	return qctx->n_embd;
}

int
qllm_prime(struct qllm_context *qctx,
	   const char *prompt)
{
	int32_t n_prompt;

	if (!qctx || !qctx->ctx || !prompt)
		return -1;

	n_prompt = llama_tokenize(qctx->vocab,
				  prompt,
				  (int32_t)strlen(prompt),
				  qctx->token_buf,
				  qctx->max_tokens,
				  true,
				  true);
	if (n_prompt < 0)
		return -1;

	if (n_prompt == 0)
		return 0;

	if (qllm_decode_tokens(qctx, qctx->token_buf, n_prompt) != 0)
		return -1;

	return 0;
}

int
qllm_next(struct qllm_context *qctx,
	  char *out,
	  size_t out_size)
{
	llama_token tok;
	char piece[256];
	int n_piece;

	if (!qctx || !qctx->ctx || !out || out_size == 0)
		return -1;

	/* Sample one token */
	tok = llama_sampler_sample(qctx->sampler, qctx->ctx, -1);
	llama_sampler_accept(qctx->sampler, tok);

	/* Treat any EOG/EOS as end-of-generation */
	if (llama_vocab_is_eog(qctx->vocab, tok))
		return 0;

	/* Advance KV with this token */
	qctx->token_buf[0] = tok;
	if (qllm_decode_tokens(qctx, qctx->token_buf, 1) != 0)
		return -1;

	/* Convert token to text piece */
	memset(piece, 0, sizeof(piece));
	n_piece = llama_token_to_piece(qctx->vocab,
				       tok,
				       piece,
				       (int)sizeof(piece),
				       false,
				       true);
	if (n_piece <= 0)
		return -1;

	if ((size_t)n_piece >= out_size)
		n_piece = (int)(out_size - 1);

	memcpy(out, piece, (size_t)n_piece);
	out[n_piece] = '\0';

	return n_piece;
}
