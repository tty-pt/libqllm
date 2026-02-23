#ifndef MOCK_REGISTRY_H
#define MOCK_REGISTRY_H

#include <stddef.h>
#include <stdint.h>

#define MOCK_REGISTRY_SIZE 256

typedef enum {
	MOCK_TYPE_PTR,
	MOCK_TYPE_INT,
	MOCK_TYPE_UINT,
	MOCK_TYPE_LONG,
	MOCK_TYPE_ULONG,
	MOCK_TYPE_FLOAT,
	MOCK_TYPE_DOUBLE,
	MOCK_TYPE_STRING,
} mock_type_t;

typedef struct {
	const char *name;
	void *original;
	void *replacement;
	mock_type_t return_type;
	int call_count;
	int enabled;
} mock_entry_t;

typedef struct {
	void *data;
	size_t size;
} mock_return_value_t;

void mock_init(void);
void mock_cleanup(void);

void mock_set(const char *name, void *original, void *replacement);
void *mock_get(const char *name);
void mock_enable(const char *name);
void mock_disable(const char *name);
void mock_disable_all(void);
int mock_call_count(const char *name);
void mock_reset_call_counts(void);

#define MOCK_SET(fn, impl) mock_set(#fn, (void*)(fn), (void*)(impl))
#define MOCK_ENABLE(fn) mock_enable(#fn)
#define MOCK_DISABLE(fn) mock_disable(#fn)
#define MOCK_CALL_COUNT(fn) mock_call_count(#fn)

typedef struct {
	int64_t int_val;
	uint64_t uint_val;
	double double_val;
	float float_val;
	void *ptr_val;
	char str_val[1024];
} mock_value_t;

typedef struct {
	const char *name;
	mock_value_t return_value;
	int should_fail;
	int fail_count;
} mock_config_t;

void mock_configure(const char *name, mock_config_t *config);
mock_config_t *mock_get_config(const char *name);

typedef struct {
	const char *name;
	void (*on_call)(void *user_data, int call_num);
	void *user_data;
} mock_callback_t;

void mock_set_callback(const char *name, mock_callback_t *callback);

#endif
