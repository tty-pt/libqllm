#include "llama_mock.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#define MOCK_MAX_TOKENS 4096
#define MOCK_MAX_MODELS 16
#define MOCK_MAX_CONTEXTS 16

static mock_llama_config_t _mock_config = {0};
static int32_t _mock_n_layer = 32;
static int32_t _mock_n_embd = 4096;
static int32_t _mock_vocab_size = 32000;

static int _mock_model_load_fail = 0;
static int _mock_context_create_fail = 0;
static int _mock_sampler_create_fail = 0;
static int _mock_tokenize_fail = 0;
static int _mock_decode_fail = 0;

static llama_token _mock_eos_token = 2;
static llama_token _mock_eot_token = 3;
static llama_token _mock_tokenizer_tokens[MOCK_MAX_TOKENS];
static int32_t _mock_tokenizer_n_tokens = 0;
static llama_token _mock_next_tokens[MOCK_MAX_TOKENS];
static int32_t _mock_next_n_tokens = 0;
static int32_t _mock_next_token_idx = 0;

static float _mock_embeddings[4096];

static int _model_load_count = 0;
static int _context_create_count = 0;
static int _decode_count = 0;
static int _sample_count = 0;

static int _model_counter = 0;
static int _context_counter = 0;
static int _model_free_count = 0;

static struct llama_model_params _last_model_params = {0};
static struct llama_context_params _last_context_params = {0};

struct llama_model {
	int id;
	int n_layer;
	int n_embd;
	char path[1024];
};

struct llama_vocab {
	int id;
	int size;
};

struct llama_context {
	int id;
	int pos;
	struct llama_model *model;
	struct llama_vocab *vocab;
	float *embeddings;
	llama_token *tokens;
	int n_tokens;
};

void
mock_llama_init(void)
{
	memset(&_mock_config, 0, sizeof(_mock_config));
	_mock_n_layer = 32;
	_mock_n_embd = 4096;
	_mock_vocab_size = 32000;
	_mock_model_load_fail = 0;
	_mock_context_create_fail = 0;
	_mock_sampler_create_fail = 0;
	_mock_tokenize_fail = 0;
	_mock_decode_fail = 0;
	_mock_tokenizer_n_tokens = 0;
	_mock_next_n_tokens = 0;
	_mock_next_token_idx = 0;
	_model_load_count = 0;
	_context_create_count = 0;
	_decode_count = 0;
	_sample_count = 0;
	_model_counter = 0;
    _context_counter = 0;
    _model_free_count = 0;

	memset(_mock_embeddings, 0, sizeof(_mock_embeddings));
}

void
mock_llama_cleanup(void)
{
}

void
mock_llama_set_config(mock_llama_config_t *config)
{
	if (config)
		_mock_config = *config;
}

void
mock_llama_set_n_layer(int32_t n_layer)
{
	_mock_n_layer = n_layer;
}

void
mock_llama_set_n_embd(int32_t n_embd)
{
	_mock_n_embd = n_embd;
}

void
mock_llama_set_vocab_size(int32_t size)
{
	_mock_vocab_size = size;
}

void
mock_llama_set_model_load_fail(int should_fail)
{
	_mock_model_load_fail = should_fail;
}

void
mock_llama_set_context_create_fail(int should_fail)
{
	_mock_context_create_fail = should_fail;
}

void
mock_llama_set_sampler_create_fail(int should_fail)
{
	_mock_sampler_create_fail = should_fail;
}

void
mock_llama_set_tokenize_fail(int should_fail)
{
	_mock_tokenize_fail = should_fail;
}

void
mock_llama_set_decode_fail(int should_fail)
{
	_mock_decode_fail = should_fail;
}

void
mock_llama_set_tokenizer_result(const llama_token *tokens, int32_t n_tokens)
{
	int32_t i;
	if (n_tokens > MOCK_MAX_TOKENS)
		n_tokens = MOCK_MAX_TOKENS;
	for (i = 0; i < n_tokens; i++)
		_mock_tokenizer_tokens[i] = tokens[i];
	_mock_tokenizer_n_tokens = n_tokens;
}

void
mock_llama_set_next_tokens(const llama_token *tokens, int32_t n_tokens)
{
	int32_t i;
	if (n_tokens > MOCK_MAX_TOKENS)
		n_tokens = MOCK_MAX_TOKENS;
	for (i = 0; i < n_tokens; i++)
		_mock_next_tokens[i] = tokens[i];
	_mock_next_n_tokens = n_tokens;
	_mock_next_token_idx = 0;
}

void
mock_llama_set_eos_token(llama_token tok)
{
	_mock_eos_token = tok;
}

void
mock_llama_set_eot_token(llama_token tok)
{
	_mock_eot_token = tok;
}

int
mock_llama_get_model_load_count(void)
{
	return _model_load_count;
}

int
mock_llama_get_context_create_count(void)
{
	return _context_create_count;
}

int
mock_llama_get_decode_count(void)
{
	return _decode_count;
}

int
mock_llama_get_sample_count(void)
{
    return _sample_count;
}

int
mock_llama_get_model_free_count(void)
{
    return _model_free_count;
}

struct llama_model_params
mock_llama_get_last_model_params(void)
{
    return _last_model_params;
}

struct llama_context_params
mock_llama_get_last_context_params(void)
{
    return _last_context_params;
}

void
mock_llama_reset_counts(void)
{
	_model_load_count = 0;
	_context_create_count = 0;
	_decode_count = 0;
	_sample_count = 0;
    memset(&_last_model_params, 0, sizeof(_last_model_params));
    memset(&_last_context_params, 0, sizeof(_last_context_params));
}

struct llama_model_params
llama_model_default_params(void)
{
	struct llama_model_params params;
	memset(&params, 0, sizeof(params));
	params.n_gpu_layers = 0;
	params.split_mode = LLAMA_SPLIT_MODE_LAYER;
	params.tensor_split = NULL;
	params.vocab_only = false;
	params.use_mmap = true;
	params.use_mlock = false;
	params.check_tensors = false;
	return params;
}

size_t llama_max_devices(void)
{
	return 16;
}

struct llama_context_params
llama_context_default_params(void)
{
	struct llama_context_params params;
	memset(&params, 0, sizeof(params));
	params.n_ctx = 512;
	params.n_batch = 512;
	params.n_ubatch = 512;
	params.n_seq_max = 1;
	params.n_threads = 4;
	params.n_threads_batch = 4;
	params.embeddings = false;
    params.offload_kqv = true;
	params.pooling_type = LLAMA_POOLING_TYPE_NONE;
	return params;
}

struct llama_sampler_chain_params
llama_sampler_chain_default_params(void)
{
	struct llama_sampler_chain_params params;
	memset(&params, 0, sizeof(params));
	params.no_perf = false;
	return params;
}

void
llama_backend_init(void)
{
}

void
llama_backend_free(void)
{
}

struct llama_model *
llama_model_load_from_file(const char *path, struct llama_model_params params)
{
	struct llama_model *model;

	_model_load_count++;
    _last_model_params = params;

	if (_mock_model_load_fail)
		return NULL;

	model = calloc(1, sizeof(*model));
	if (!model)
		return NULL;

	model->id = ++_model_counter;
	model->n_layer = _mock_n_layer;
	model->n_embd = _mock_n_embd;
	strncpy(model->path, path, sizeof(model->path) - 1);

	return model;
}

void
llama_model_free(struct llama_model *model)
{
    _model_free_count++;
    free(model);
}

int32_t
llama_model_n_layer(const struct llama_model *model)
{
	if (!model)
		return 0;
	return model->n_layer;
}

int32_t
llama_model_n_embd(const struct llama_model *model)
{
	if (!model)
		return 0;
	return model->n_embd;
}

const struct llama_vocab *
llama_model_get_vocab(const struct llama_model *model)
{
	static struct llama_vocab vocab;
	if (!model)
		return NULL;
	vocab.id = model->id;
	vocab.size = _mock_vocab_size;
	return &vocab;
}

struct llama_context *
llama_init_from_model(struct llama_model *model, struct llama_context_params params)
{
	struct llama_context *ctx;

	_context_create_count++;
    _last_context_params = params;

	if (_mock_context_create_fail)
		return NULL;

	if (!model)
		return NULL;

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;

	ctx->id = ++_context_counter;
	ctx->model = model;
	ctx->pos = 0;
	ctx->vocab = (struct llama_vocab *)llama_model_get_vocab(model);
	ctx->tokens = calloc(params.n_ctx, sizeof(llama_token));
	ctx->n_tokens = 0;

	for (int i = 0; i < model->n_embd && i < 4096; i++)
		_mock_embeddings[i] = (float)(i % 100) / 100.0f;
	ctx->embeddings = _mock_embeddings;

	return ctx;
}

void
llama_free(struct llama_context *ctx)
{
	if (ctx) {
		free(ctx->tokens);
		free(ctx);
	}
}

llama_memory_t
llama_get_memory(struct llama_context *ctx)
{
	return ctx;
}

llama_pos
llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq_id)
{
	struct llama_context *ctx = (struct llama_context *)mem;
	if (!ctx)
		return 0;
	return ctx->n_tokens - 1;
}

void
llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id,
		    llama_pos p0, llama_pos p1)
{
	struct llama_context *ctx = (struct llama_context *)mem;
	if (!ctx)
		return;

	if (p1 > ctx->n_tokens)
		p1 = ctx->n_tokens;

	int32_t len = p1 - p0;
	if (len <= 0)
		return;

	int32_t remaining = ctx->n_tokens - p1;
	if (remaining > 0)
		memmove(ctx->tokens + p0, ctx->tokens + p1, remaining * sizeof(llama_token));

	ctx->n_tokens -= len;
}

void
llama_memory_seq_add(llama_memory_t mem, llama_seq_id seq_id,
		     llama_pos p0, llama_pos p1, llama_pos delta)
{
}

struct llama_batch
llama_batch_init(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
{
    struct llama_batch batch;
    memset(&batch, 0, sizeof(batch));
    batch.n_tokens = n_tokens;
    if (n_tokens > 0) {
        batch.token = calloc((size_t)n_tokens, sizeof(llama_token));
        batch.n_seq_id = calloc((size_t)n_tokens, sizeof(int32_t));
        batch.seq_id = calloc((size_t)n_tokens, sizeof(llama_seq_id *));
        batch.pos = NULL;
        batch.logits = calloc((size_t)n_tokens, sizeof(int32_t));
    }
    return batch;
}

void
llama_batch_free(struct llama_batch batch)
{
    if (batch.n_tokens > 0) {
        free(batch.token);
        free(batch.n_seq_id);
        free(batch.seq_id);
        free(batch.logits);
    }
}

int32_t
llama_tokenize(const struct llama_vocab *vocab,
	       const char *text,
	       int32_t text_len,
	       llama_token *tokens,
	       int32_t n_tokens_max,
	       bool add_bos,
	       bool special)
{
	int32_t i;

	if (_mock_tokenize_fail)
		return -1;

	if (_mock_tokenizer_n_tokens > 0) {
		int32_t n = _mock_tokenizer_n_tokens;
		if (n > n_tokens_max)
			n = n_tokens_max;
		for (i = 0; i < n; i++)
			tokens[i] = _mock_tokenizer_tokens[i];
		return n;
	}

	int32_t n = text_len;
	if (n > n_tokens_max)
		n = n_tokens_max;

	for (i = 0; i < n; i++)
		tokens[i] = (llama_token)(unsigned char)text[i] + 100;

	return n;
}

int32_t
llama_token_to_piece(const struct llama_vocab *vocab,
		     llama_token token,
		     char *buf,
		     int32_t length,
		     bool add_space,
		     bool special)
{
    /* Allow a wider range in the mock so tests that sample synthetic
     * token values still map to visible characters. Real tokenizer/codec
     * handles are more complex; this is sufficient for unit tests. */
    if (token < 0 || token > 255)
        return 0;

    char c = (char)(token - 100);
    if (length >= 1) {
        buf[0] = c;
        if (length >= 2)
            buf[1] = '\0';
        return 1;
    }
    return 0;
}

llama_token
llama_vocab_eos(const struct llama_vocab *vocab)
{
	return _mock_eos_token;
}

llama_token
llama_vocab_eot(const struct llama_vocab *vocab)
{
	return _mock_eot_token;
}

bool
llama_vocab_is_eog(const struct llama_vocab *vocab, llama_token token)
{
	return token == _mock_eos_token || token == _mock_eot_token;
}

bool
llama_vocab_is_control(const struct llama_vocab *vocab, llama_token token)
{
	return token >= 100 && token < 200;
}

int
llama_decode(struct llama_context *ctx, struct llama_batch batch)
{
	_decode_count++;

	if (_mock_decode_fail)
		return -1;

	if (!ctx)
		return -1;

	int32_t i;
	for (i = 0; i < batch.n_tokens; i++) {
		if (ctx->n_tokens < 4096) {
			ctx->tokens[ctx->n_tokens] = batch.token[i];
			ctx->n_tokens++;
		}
	}

	return 0;
}

const float *
llama_get_embeddings(struct llama_context *ctx)
{
	if (!ctx)
		return NULL;
	return ctx->embeddings;
}

struct llama_sampler *
llama_sampler_chain_init(struct llama_sampler_chain_params params)
{
	struct llama_sampler *s;

	if (_mock_sampler_create_fail)
		return NULL;

	s = calloc(1, sizeof(*s));
	return s;
}

void
llama_sampler_chain_add(struct llama_sampler *chain, struct llama_sampler *sampler)
{
}

void
llama_sampler_free(struct llama_sampler *smpl)
{
	free(smpl);
}

struct llama_sampler *
llama_sampler_init(const struct llama_sampler_i *iface, void *ctx)
{
	struct llama_sampler *s = calloc(1, sizeof(*s));
	if (s) {
		s->iface = iface;
		s->ctx = ctx;
	}
	return s;
}

struct llama_sampler *
llama_sampler_init_penalties(int32_t last_n, float repeat, float freq, float present)
{
	return calloc(1, sizeof(struct llama_sampler));
}

struct llama_sampler *
llama_sampler_init_dist(uint64_t seed)
{
	return calloc(1, sizeof(struct llama_sampler));
}

struct llama_sampler *
llama_sampler_init_top_k(int32_t top_k)
{
	return calloc(1, sizeof(struct llama_sampler));
}

struct llama_sampler *
llama_sampler_init_top_p(float top_p, float tail_free_freq_z)
{
	return calloc(1, sizeof(struct llama_sampler));
}

struct llama_sampler *
llama_sampler_init_temp(float temp)
{
	return calloc(1, sizeof(struct llama_sampler));
}

llama_token
llama_sampler_sample(struct llama_sampler *smpl, struct llama_context *ctx, int32_t idx)
{
	_sample_count++;

	if (_mock_next_n_tokens > 0 && _mock_next_token_idx < _mock_next_n_tokens)
		return _mock_next_tokens[_mock_next_token_idx++];

	if (_sample_count > 10)
		return _mock_eos_token;

	return (llama_token)('a' + (_sample_count - 1) % 26) + 100;
}

void
llama_sampler_accept(struct llama_sampler *smpl, llama_token token)
{
}

struct llama_sampler * llama_sampler_init_grammar(
    const struct llama_vocab * vocab,
    const char * grammar_str,
    const char * grammar_root) {
    (void)vocab;
    (void)grammar_str;
    (void)grammar_root;
    return calloc(1, sizeof(struct llama_sampler));
}
