#include "gguf_mock.h"
#include <stdlib.h>
#include <string.h>

#define MAX_MOCK_TENSORS 1024

static int _mock_n_tensors = 0;
static mock_tensor_info _mock_tensors[MAX_MOCK_TENSORS];
static int _mock_init_fail = 0;

struct gguf_context {
	int n_tensors;
	mock_tensor_info *tensors;
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

        /* If strdup failed, clean up previously-allocated names and return NULL. */
        if (_mock_tensors[i].name && ctx->tensors[i].name == NULL) {
            int j;
            for (j = 0; j < i; j++)
                free((void*)ctx->tensors[j].name);
            free(ctx->tensors);
            free(ctx);
            return NULL;
        }
        /* Free the global mock name now that the ctx owns a copy. */
        free((void*)_mock_tensors[i].name);
        _mock_tensors[i].name = NULL;
    }

	return ctx;
}

void
gguf_free(struct gguf_context *ctx)
{
    if (ctx) {
        int i;
        for (i = 0; i < ctx->n_tensors; i++)
            free((void*)ctx->tensors[i].name);
        free(ctx->tensors);
        free(ctx);
    }
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
