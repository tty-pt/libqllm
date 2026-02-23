#ifndef GGUF_MOCK_H
#define GGUF_MOCK_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct gguf_context;

typedef struct gguf_init_params {
	bool no_alloc;
} gguf_init_params;

typedef struct {
	const char *name;
	int64_t n_tensors;
	size_t size;
} mock_tensor_info;

void mock_gguf_init(void);
void mock_gguf_cleanup(void);

void mock_gguf_set_n_tensors(int n);
void mock_gguf_set_tensor_name(int idx, const char *name);
void mock_gguf_set_tensor_size(int idx, size_t size);
void mock_gguf_set_init_fail(int should_fail);

void mock_gguf_add_tensor(const char *name, size_t size);

struct gguf_context *gguf_init_from_file(const char *path, struct gguf_init_params params);
void gguf_free(struct gguf_context *ctx);

int gguf_get_n_tensors(struct gguf_context *ctx);
const char *gguf_get_tensor_name(struct gguf_context *ctx, int i);
size_t gguf_get_tensor_size(struct gguf_context *ctx, int i);

#ifdef __cplusplus
}
#endif

#endif
