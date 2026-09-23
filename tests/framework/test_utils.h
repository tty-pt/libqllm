#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <setjmp.h>

extern int _tests_run;
extern int _tests_passed;
extern int _tests_failed;
extern int _current_test_failed;
extern char _current_test_name[256];
extern jmp_buf _test_jmp_buf;

#define TEST(name) \
	static void test_##name(void); \
	static void __attribute__((constructor)) _register_##name(void) { \
		_register_test(#name, test_##name); \
	} \
	static void test_##name(void)

#define ASSERT(cond) do { \
	if (!(cond)) { \
		_fail_assertion(#cond, __FILE__, __LINE__); \
	} \
} while(0)

#define ASSERT_EQ(a, b) do { \
	__typeof__(a) _a = (a); \
	__typeof__(b) _b = (b); \
	if (_a != _b) { \
		_fail_eq((long long)_a, (long long)_b, #a, #b, __FILE__, __LINE__); \
	} \
} while(0)

#define ASSERT_NE(a, b) do { \
	__typeof__(a) _a = (a); \
	__typeof__(b) _b = (b); \
	if (_a == _b) { \
		_fail_ne((long long)_a, (long long)_b, #a, #b, __FILE__, __LINE__); \
	} \
} while(0)

#define ASSERT_NULL(p) do { \
	if ((p) != NULL) { \
		_fail_null(#p, __FILE__, __LINE__); \
	} \
} while(0)

#define ASSERT_NOT_NULL(p) do { \
	if ((p) == NULL) { \
		_fail_not_null(#p, __FILE__, __LINE__); \
	} \
} while(0)

#define ASSERT_STR_EQ(a, b) do { \
	const char *_a = (a); \
	const char *_b = (b); \
	if (_a == NULL || _b == NULL || strcmp(_a, _b) != 0) { \
		_fail_str_eq(_a, _b, #a, #b, __FILE__, __LINE__); \
	} \
} while(0)

#define ASSERT_STR_NE(a, b) do { \
	const char *_a = (a); \
	const char *_b = (b); \
	if (_a == NULL || _b == NULL || strcmp(_a, _b) == 0) { \
		_fail_str_ne(_a, _b, #a, #b, __FILE__, __LINE__); \
	} \
} while(0)

#define ASSERT_MEM_EQ(a, b, n) do { \
	const void *_a = (a); \
	const void *_b = (b); \
	if (_a == NULL || _b == NULL || memcmp(_a, _b, (n)) != 0) { \
		_fail_mem_eq(_a, _b, (n), #a, #b, __FILE__, __LINE__); \
	} \
} while(0)

#define ASSERT_FLOAT_EQ(a, b) do { \
	float _a = (a); \
	float _b = (b); \
	float _diff = _a - _b; \
	if (_diff < 0) _diff = -_diff; \
	if (_diff > 0.0001f) { \
		_fail_float_eq(_a, _b, #a, #b, __FILE__, __LINE__); \
	} \
} while(0)

#define ASSERT_TRUE(cond) ASSERT(cond)
#define ASSERT_FALSE(cond) ASSERT(!(cond))

#define SKIP(msg) do { \
	printf("  SKIP: %s\n", msg); \
	longjmp(_test_jmp_buf, 2); \
} while(0)

typedef void (*test_func_t)(void);

void _register_test(const char *name, test_func_t func);
void _fail_assertion(const char *cond, const char *file, int line);
void _fail_eq(long long a, long long b, const char *a_str, const char *b_str, const char *file, int line);
void _fail_ne(long long a, long long b, const char *a_str, const char *b_str, const char *file, int line);
void _fail_null(const char *expr, const char *file, int line);
void _fail_not_null(const char *expr, const char *file, int line);
void _fail_str_eq(const char *a, const char *b, const char *a_str, const char *b_str, const char *file, int line);
void _fail_str_ne(const char *a, const char *b, const char *a_str, const char *b_str, const char *file, int line);
void _fail_mem_eq(const void *a, const void *b, size_t n, const char *a_str, const char *b_str, const char *file, int line);
void _fail_float_eq(float a, float b, const char *a_str, const char *b_str, const char *file, int line);

int run_all_tests(void);
void test_summary(void);

/* Test harness freed-pointer tracking (enabled only for MOCK_BUILD).
 * Implemented in the test harness so production code doesn't get test-only
 * globals. */
#if defined(MOCK_BUILD)
int qllm_ptr_freed(const void *p);
void qllm_record_freed(void *p);
void qllm_unrecord_freed(void *p);
#endif

#define TEST_SUITE(name) void register_suite_##name(void)
#define RUN_SUITE(name) register_suite_##name()

typedef struct {
	const char *name;
	test_func_t func;
} test_entry_t;

#define MAX_TESTS 1024

#endif
