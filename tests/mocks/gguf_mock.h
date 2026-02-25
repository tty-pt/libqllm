#ifndef GGUF_MOCK_H
#define GGUF_MOCK_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

enum gguf_type {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

struct gguf_context;

typedef struct gguf_init_params {
	bool no_alloc;
} gguf_init_params;

typedef struct {
	const char *name;
	int64_t n_tensors;
	size_t size;
} mock_tensor_info;

typedef struct {
    char key[64];
    enum gguf_type type;
    union {
        uint8_t u8;
        int8_t i8;
        uint16_t u16;
        int16_t i16;
        uint32_t u32;
        int32_t i32;
        uint64_t u64;
        int64_t i64;
        float f32;
        double f64;
        char *str;
    } val;
} mock_kv_pair;

void mock_gguf_init(void);
void mock_gguf_cleanup(void);

void mock_gguf_set_n_tensors(int n);
void mock_gguf_set_tensor_name(int idx, const char *name);
void mock_gguf_set_tensor_size(int idx, size_t size);
void mock_gguf_set_init_fail(int should_fail);

void mock_gguf_add_tensor(const char *name, size_t size);
void mock_gguf_set_kv_string(const char *key, const char *val);
void mock_gguf_set_kv_int32(const char *key, int32_t val);
void mock_gguf_set_kv_uint32(const char *key, uint32_t val);

struct gguf_context *gguf_init_from_file(const char *path, struct gguf_init_params params);
void gguf_free(struct gguf_context *ctx);

int gguf_get_n_tensors(struct gguf_context *ctx);
const char *gguf_get_tensor_name(struct gguf_context *ctx, int i);
size_t gguf_get_tensor_size(struct gguf_context *ctx, int i);

int64_t gguf_find_key(struct gguf_context *ctx, const char *key);
enum gguf_type gguf_get_kv_type(struct gguf_context *ctx, int64_t key_id);
const char *gguf_get_val_str(struct gguf_context *ctx, int64_t key_id);
uint8_t gguf_get_val_u8(struct gguf_context *ctx, int64_t key_id);
int8_t gguf_get_val_i8(struct gguf_context *ctx, int64_t key_id);
uint16_t gguf_get_val_u16(struct gguf_context *ctx, int64_t key_id);
int16_t gguf_get_val_i16(struct gguf_context *ctx, int64_t key_id);
uint32_t gguf_get_val_u32(struct gguf_context *ctx, int64_t key_id);
int32_t gguf_get_val_i32(struct gguf_context *ctx, int64_t key_id);
uint64_t gguf_get_val_u64(struct gguf_context *ctx, int64_t key_id);
int64_t gguf_get_val_i64(struct gguf_context *ctx, int64_t key_id);

#ifdef __cplusplus
}
#endif

#endif
