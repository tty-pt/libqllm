#include "gguf_mock.h"
#include <stdlib.h>
#include <string.h>

#define MAX_MOCK_TENSORS 1024
#define MAX_MOCK_KV_PAIRS 64

static int _mock_n_tensors = 0;
static mock_tensor_info _mock_tensors[MAX_MOCK_TENSORS];
static int _mock_init_fail = 0;
static int _mock_n_kv = 0;
static mock_kv_pair _mock_kv[MAX_MOCK_KV_PAIRS];

struct gguf_context {
	int n_tensors;
	mock_tensor_info *tensors;
	int n_kv;
	mock_kv_pair *kv;
};

void
mock_gguf_init(void)
{
    /* Free any existing mock data before reinitializing to avoid leaks */
    mock_gguf_cleanup();
    _mock_init_fail = 0;
    memset(_mock_tensors, 0, sizeof(_mock_tensors));
}

void
mock_gguf_cleanup(void)
{
	int i;
	for (i = 0; i < _mock_n_tensors; i++) {
		free((void*)_mock_tensors[i].name);
	}
	_mock_n_tensors = 0;

	for (i = 0; i < _mock_n_kv; i++) {
		if (_mock_kv[i].type == GGUF_TYPE_STRING && _mock_kv[i].val.str) {
			free(_mock_kv[i].val.str);
		}
	}
	_mock_n_kv = 0;
}

void
mock_gguf_set_kv_string(const char *key, const char *val)
{
	if (_mock_n_kv >= MAX_MOCK_KV_PAIRS) return;
	strncpy(_mock_kv[_mock_n_kv].key, key, sizeof(_mock_kv[_mock_n_kv].key) - 1);
	_mock_kv[_mock_n_kv].type = GGUF_TYPE_STRING;
	_mock_kv[_mock_n_kv].val.str = strdup(val);
	_mock_n_kv++;
}

void
mock_gguf_set_kv_int32(const char *key, int32_t val)
{
	if (_mock_n_kv >= MAX_MOCK_KV_PAIRS) return;
	strncpy(_mock_kv[_mock_n_kv].key, key, sizeof(_mock_kv[_mock_n_kv].key) - 1);
	_mock_kv[_mock_n_kv].type = GGUF_TYPE_INT32;
	_mock_kv[_mock_n_kv].val.i32 = val;
	_mock_n_kv++;
}

void
mock_gguf_set_kv_uint32(const char *key, uint32_t val)
{
	if (_mock_n_kv >= MAX_MOCK_KV_PAIRS) return;
	strncpy(_mock_kv[_mock_n_kv].key, key, sizeof(_mock_kv[_mock_n_kv].key) - 1);
	_mock_kv[_mock_n_kv].type = GGUF_TYPE_UINT32;
	_mock_kv[_mock_n_kv].val.u32 = val;
	_mock_n_kv++;
}

void
mock_gguf_set_n_tensors(int n)
{
	_mock_n_tensors = n;
}

void
mock_gguf_set_tensor_name(int idx, const char *name)
{
	if (idx >= 0 && idx < MAX_MOCK_TENSORS) {
		free((void*)_mock_tensors[idx].name);
		_mock_tensors[idx].name = strdup(name);
		if (idx >= _mock_n_tensors)
			_mock_n_tensors = idx + 1;
	}
}

void
mock_gguf_set_tensor_size(int idx, size_t size)
{
	if (idx >= 0 && idx < MAX_MOCK_TENSORS) {
		_mock_tensors[idx].size = size;
		if (idx >= _mock_n_tensors)
			_mock_n_tensors = idx + 1;
	}
}

void
mock_gguf_set_init_fail(int should_fail)
{
	_mock_init_fail = should_fail;
}

void
mock_gguf_add_tensor(const char *name, size_t size)
{
    if (_mock_n_tensors < MAX_MOCK_TENSORS) {
        static int _atexit_registered = 0;
        if (!_atexit_registered) {
            /* Ensure names are freed at process exit if tests don't call cleanup. */
            atexit(mock_gguf_cleanup);
            _atexit_registered = 1;
        }
        _mock_tensors[_mock_n_tensors].name = strdup(name);
        _mock_tensors[_mock_n_tensors].size = size;
        _mock_n_tensors++;
    }
}

struct gguf_context *
gguf_init_from_file(const char *path, struct gguf_init_params params)
{
	struct gguf_context *ctx;
	int i;

	if (_mock_init_fail)
		return NULL;

	ctx = calloc(1, sizeof(*ctx));
	if (!ctx)
		return NULL;

	ctx->n_tensors = _mock_n_tensors;
	ctx->tensors = calloc(_mock_n_tensors, sizeof(mock_tensor_info));
	if (!ctx->tensors) {
		free(ctx);
		return NULL;
	}

    for (i = 0; i < _mock_n_tensors; i++) {
        /* Duplicate the mock name into the ctx so the ctx owns its copy. */
        if (_mock_tensors[i].name)
            ctx->tensors[i].name = strdup(_mock_tensors[i].name);
        else
            ctx->tensors[i].name = NULL;
        ctx->tensors[i].size = _mock_tensors[i].size;
    }

	/* Copy KV pairs */
	ctx->n_kv = _mock_n_kv;
	ctx->kv = calloc(_mock_n_kv, sizeof(mock_kv_pair));
	if (!ctx->kv) {
		free(ctx->tensors);
		free(ctx);
		return NULL;
	}
	for (i = 0; i < _mock_n_kv; i++) {
		ctx->kv[i] = _mock_kv[i];
		if (ctx->kv[i].type == GGUF_TYPE_STRING && ctx->kv[i].val.str) {
			ctx->kv[i].val.str = strdup(_mock_kv[i].val.str);
		}
	}

	return ctx;
}

void
gguf_free(struct gguf_context *ctx)
{
    int i;
    if (ctx) {
        for (i = 0; i < ctx->n_tensors; i++)
            free((void*)ctx->tensors[i].name);
        free(ctx->tensors);

		for (i = 0; i < ctx->n_kv; i++) {
			if (ctx->kv[i].type == GGUF_TYPE_STRING && ctx->kv[i].val.str) {
				free(ctx->kv[i].val.str);
			}
		}
		free(ctx->kv);

        free(ctx);
    }
}

int64_t
gguf_find_key(struct gguf_context *ctx, const char *key)
{
	int i;
	if (!ctx || !key) return -1;
	for (i = 0; i < ctx->n_kv; i++) {
		if (strcmp(ctx->kv[i].key, key) == 0)
			return i;
	}
	return -1;
}

enum gguf_type gguf_get_kv_type(struct gguf_context *ctx, int64_t key_id)
{
	if (!ctx || key_id < 0 || key_id >= ctx->n_kv)
		return 0;
	return (enum gguf_type)ctx->kv[key_id].type;
}

const char *gguf_get_val_str(struct gguf_context *ctx, int64_t key_id)
{
	if (!ctx || key_id < 0 || key_id >= ctx->n_kv)
		return NULL;
	if (ctx->kv[key_id].type != GGUF_TYPE_STRING)
		return NULL;
	return ctx->kv[key_id].val.str;
}

uint8_t gguf_get_val_u8(struct gguf_context *ctx, int64_t key_id) {
	if (!ctx || key_id < 0 || key_id >= ctx->n_kv) return 0;
	return ctx->kv[key_id].val.u8;
}
int8_t gguf_get_val_i8(struct gguf_context *ctx, int64_t key_id) {
	if (!ctx || key_id < 0 || key_id >= ctx->n_kv) return 0;
	return ctx->kv[key_id].val.i8;
}
uint16_t gguf_get_val_u16(struct gguf_context *ctx, int64_t key_id) {
	if (!ctx || key_id < 0 || key_id >= ctx->n_kv) return 0;
	return ctx->kv[key_id].val.u16;
}
int16_t gguf_get_val_i16(struct gguf_context *ctx, int64_t key_id) {
	if (!ctx || key_id < 0 || key_id >= ctx->n_kv) return 0;
	return ctx->kv[key_id].val.i16;
}
uint32_t gguf_get_val_u32(struct gguf_context *ctx, int64_t key_id) {
	if (!ctx || key_id < 0 || key_id >= ctx->n_kv) return 0;
	return ctx->kv[key_id].val.u32;
}
int32_t gguf_get_val_i32(struct gguf_context *ctx, int64_t key_id) {
	if (!ctx || key_id < 0 || key_id >= ctx->n_kv) return 0;
	return ctx->kv[key_id].val.i32;
}
uint64_t gguf_get_val_u64(struct gguf_context *ctx, int64_t key_id) {
	if (!ctx || key_id < 0 || key_id >= ctx->n_kv) return 0;
	return ctx->kv[key_id].val.u64;
}
int64_t gguf_get_val_i64(struct gguf_context *ctx, int64_t key_id) {
	if (!ctx || key_id < 0 || key_id >= ctx->n_kv) return 0;
	return ctx->kv[key_id].val.i64;
}

int
gguf_get_n_tensors(struct gguf_context *ctx)
{
	if (!ctx)
		return 0;
	return ctx->n_tensors;
}

const char *
gguf_get_tensor_name(struct gguf_context *ctx, int i)
{
	if (!ctx || i < 0 || i >= ctx->n_tensors)
		return NULL;
	return ctx->tensors[i].name;
}

size_t
gguf_get_tensor_size(struct gguf_context *ctx, int i)
{
	if (!ctx || i < 0 || i >= ctx->n_tensors)
		return 0;
	return ctx->tensors[i].size;
}
