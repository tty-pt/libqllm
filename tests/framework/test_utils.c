#include "test_utils.h"
#include <stdarg.h>

int _tests_run = 0;
int _tests_passed = 0;
int _tests_failed = 0;
int _current_test_failed = 0;
char _current_test_name[256] = {0};
jmp_buf _test_jmp_buf;

static test_entry_t _test_registry[MAX_TESTS];
static int _test_count = 0;

void
_register_test(const char *name, test_func_t func)
{
	if (_test_count >= MAX_TESTS) {
		fprintf(stderr, "ERROR: Test registry full (max %d tests)\n", MAX_TESTS);
		return;
	}
	_test_registry[_test_count].name = name;
	_test_registry[_test_count].func = func;
	_test_count++;
}

static void
_print_diff_hex(const void *a, const void *b, size_t n)
{
	size_t i;
	fprintf(stderr, "  Expected: ");
	for (i = 0; i < n && i < 32; i++)
		fprintf(stderr, "%02x ", ((unsigned char*)a)[i]);
	fprintf(stderr, "\n  Actual:   ");
	for (i = 0; i < n && i < 32; i++)
		fprintf(stderr, "%02x ", ((unsigned char*)b)[i]);
	if (n > 32)
		fprintf(stderr, "... (truncated)");
	fprintf(stderr, "\n");
}

void
_fail_assertion(const char *cond, const char *file, int line)
{
	fprintf(stderr, "  FAIL: %s:%d: assertion failed: %s\n", file, line, cond);
	_current_test_failed = 1;
	longjmp(_test_jmp_buf, 1);
}

void
_fail_eq(long long a, long long b, const char *a_str, const char *b_str,
	const char *file, int line)
{
	fprintf(stderr, "  FAIL: %s:%d: %s != %s\n", file, line, a_str, b_str);
	fprintf(stderr, "        Expected: %lld\n", a);
	fprintf(stderr, "        Actual:   %lld\n", b);
	_current_test_failed = 1;
	longjmp(_test_jmp_buf, 1);
}

void
_fail_ne(long long a, long long b, const char *a_str, const char *b_str,
	const char *file, int line)
{
	fprintf(stderr, "  FAIL: %s:%d: %s == %s (expected different)\n",
		file, line, a_str, b_str);
	fprintf(stderr, "        Value: %lld\n", a);
	_current_test_failed = 1;
	longjmp(_test_jmp_buf, 1);
}

void
_fail_null(const char *expr, const char *file, int line)
{
	fprintf(stderr, "  FAIL: %s:%d: %s is not NULL\n", file, line, expr);
	_current_test_failed = 1;
	longjmp(_test_jmp_buf, 1);
}

void
_fail_not_null(const char *expr, const char *file, int line)
{
	fprintf(stderr, "  FAIL: %s:%d: %s is NULL\n", file, line, expr);
	_current_test_failed = 1;
	longjmp(_test_jmp_buf, 1);
}

void
_fail_str_eq(const char *a, const char *b, const char *a_str, const char *b_str,
	     const char *file, int line)
{
	fprintf(stderr, "  FAIL: %s:%d: %s != %s\n", file, line, a_str, b_str);
	fprintf(stderr, "        Expected: \"%s\"\n", a ? a : "(null)");
	fprintf(stderr, "        Actual:   \"%s\"\n", b ? b : "(null)");
	_current_test_failed = 1;
	longjmp(_test_jmp_buf, 1);
}

void
_fail_str_ne(const char *a, const char *b, const char *a_str, const char *b_str,
	     const char *file, int line)
{
	fprintf(stderr, "  FAIL: %s:%d: %s == %s (expected different)\n",
		file, line, a_str, b_str);
	fprintf(stderr, "        Value: \"%s\"\n", a ? a : "(null)");
	_current_test_failed = 1;
	longjmp(_test_jmp_buf, 1);
}

void
_fail_mem_eq(const void *a, const void *b, size_t n, const char *a_str,
	     const char *b_str, const char *file, int line)
{
	fprintf(stderr, "  FAIL: %s:%d: %s != %s (size %zu)\n",
		file, line, a_str, b_str, n);
	if (a && b)
		_print_diff_hex(a, b, n);
	else
		fprintf(stderr, "        One or both pointers are NULL\n");
	_current_test_failed = 1;
	longjmp(_test_jmp_buf, 1);
}

void
_fail_float_eq(float a, float b, const char *a_str, const char *b_str,
	       const char *file, int line)
{
	fprintf(stderr, "  FAIL: %s:%d: %s != %s\n", file, line, a_str, b_str);
	fprintf(stderr, "        Expected: %f\n", a);
	fprintf(stderr, "        Actual:   %f\n", b);
	_current_test_failed = 1;
	longjmp(_test_jmp_buf, 1);
}

int
run_all_tests(void)
{
	int i;
	int result;

	printf("\n=== Running %d tests ===\n\n", _test_count);

	for (i = 0; i < _test_count; i++) {
		_tests_run++;
		_current_test_failed = 0;
		strncpy(_current_test_name, _test_registry[i].name,
			sizeof(_current_test_name) - 1);

		printf("  [%03d] %s... ", i + 1, _test_registry[i].name);
		fflush(stdout);

		result = setjmp(_test_jmp_buf);
		if (result == 0) {
			_test_registry[i].func();
			if (!_current_test_failed) {
				printf("OK\n");
				_tests_passed++;
			}
		} else if (result == 2) {
			/* Skipped */
		} else {
			_tests_failed++;
		}
	}

	printf("\n");
	test_summary();

	return _tests_failed > 0 ? 1 : 0;
}

void
test_summary(void)
{
	printf("=== Test Summary ===\n");
	printf("  Total:  %d\n", _tests_run);
	printf("  Passed: %d\n", _tests_passed);
	printf("  Failed: %d\n", _tests_failed);
	printf("  Skipped: %d\n", _tests_run - _tests_passed - _tests_failed);
	printf("\n");

	if (_tests_failed == 0)
		printf("All tests passed!\n");
	else
		printf("SOME TESTS FAILED!\n");
}

#if defined(MOCK_BUILD)
/* Simple freed-pointer tracker used only during MOCK_BUILD. Kept in the test
 * harness so production sources don't contain test globals. This mirrors the
 * earlier behavior that lived inside src/libqllm.c. */
static void *freed_ctxs[1024];
static size_t freed_ctxs_n = 0;

int qllm_ptr_freed(const void *p)
{
    for (size_t i = 0; i < freed_ctxs_n; ++i) {
        if (freed_ctxs[i] == p)
            return 1;
    }
    return 0;
}

void qllm_record_freed(void *p)
{
    if (freed_ctxs_n < (sizeof(freed_ctxs)/sizeof(freed_ctxs[0])))
        freed_ctxs[freed_ctxs_n++] = p;
}

void qllm_unrecord_freed(void *p)
{
    for (size_t i = 0; i < freed_ctxs_n; ++i) {
        if (freed_ctxs[i] == p) {
            for (size_t j = i + 1; j < freed_ctxs_n; ++j)
                freed_ctxs[j - 1] = freed_ctxs[j];
            freed_ctxs_n--;
            break;
        }
    }
}
#endif
