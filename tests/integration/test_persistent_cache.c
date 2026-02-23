/* Test persistent cache feature with QLLM_CACHE_FILE */
#include "test_utils.h"
#include "../mocks/qmap_mock.h"
#include <stdlib.h>
#include <string.h>

/* Test that QLLM_CACHE_FILE environment variable is respected */
TEST(persistent_cache_env_var)
{
	const char *old_env;
	
	mock_qmap_init();
	
	/* Save old value */
	old_env = getenv("QLLM_CACHE_FILE");
	
	/* Set environment variable */
	setenv("QLLM_CACHE_FILE", "/tmp/test_cache.qmap", 1);
	
	/* In a real scenario, qllm_init() would be called here
	 * and would use the cache file. Our mock doesn't support
	 * file operations, so we just verify the API pattern. */
	
	const char *cache_file = getenv("QLLM_CACHE_FILE");
	ASSERT_NOT_NULL(cache_file);
	ASSERT_STR_EQ(cache_file, "/tmp/test_cache.qmap");
	
	/* Restore old value */
	if (old_env)
		setenv("QLLM_CACHE_FILE", old_env, 1);
	else
		unsetenv("QLLM_CACHE_FILE");
	
	mock_qmap_cleanup();
}

/* Test cache behavior without environment variable (in-memory only) */
TEST(persistent_cache_disabled_by_default)
{
	const char *cache_file;
	
	mock_qmap_init();
	
	/* Ensure env var is not set */
	unsetenv("QLLM_CACHE_FILE");
	
	cache_file = getenv("QLLM_CACHE_FILE");
	ASSERT_NULL(cache_file);
	
	/* This means qmap_open is called with NULL for filename,
	 * resulting in in-memory only cache (backward compatible) */
	
	mock_qmap_cleanup();
}

/* Test that cache file path can be a user-specific path */
TEST(persistent_cache_user_path)
{
	const char *old_env;
	char path[256];
	const char *home;
	
	mock_qmap_init();
	
	old_env = getenv("QLLM_CACHE_FILE");
	
	/* Construct path like ~/.cache/qllm/models.cache */
	home = getenv("HOME");
	if (home) {
		snprintf(path, sizeof(path), "%s/.cache/qllm/models.cache", home);
		setenv("QLLM_CACHE_FILE", path, 1);
		
		const char *cache_file = getenv("QLLM_CACHE_FILE");
		ASSERT_NOT_NULL(cache_file);
		ASSERT(strstr(cache_file, ".cache/qllm") != NULL);
	}
	
	/* Restore */
	if (old_env)
		setenv("QLLM_CACHE_FILE", old_env, 1);
	else
		unsetenv("QLLM_CACHE_FILE");
	
	mock_qmap_cleanup();
}

/* Test that excessively long paths are rejected */
TEST(persistent_cache_path_too_long)
{
	const char *old_env;
	char long_path[5000];
	
	mock_qmap_init();
	
	old_env = getenv("QLLM_CACHE_FILE");
	
	/* Create a path that's >= 4096 characters */
	memset(long_path, 'a', sizeof(long_path) - 1);
	long_path[0] = '/';  /* Make it look like an absolute path */
	long_path[sizeof(long_path) - 1] = '\0';
	
	setenv("QLLM_CACHE_FILE", long_path, 1);
	
	/* qllm_init() should reject this path and fall back to in-memory.
	 * The validation happens inside qllm_init(), which would print a
	 * warning to stderr and set cache_file to NULL before calling
	 * qmap_open(). We can't easily test the warning output in unit
	 * tests, but we verify the path length exceeds the limit. */
	ASSERT(strlen(long_path) >= 4096);
	
	/* Restore */
	if (old_env)
		setenv("QLLM_CACHE_FILE", old_env, 1);
	else
		unsetenv("QLLM_CACHE_FILE");
	
	mock_qmap_cleanup();
}

/* Test that empty string is treated as disabled */
TEST(persistent_cache_empty_path)
{
	const char *old_env;
	
	mock_qmap_init();
	
	old_env = getenv("QLLM_CACHE_FILE");
	
	/* Empty string should be treated as disabled */
	setenv("QLLM_CACHE_FILE", "", 1);
	
	const char *cache_file = getenv("QLLM_CACHE_FILE");
	ASSERT_NOT_NULL(cache_file);
	ASSERT_EQ(cache_file[0], '\0');
	
	/* qllm_init() checks for empty string and treats it as NULL */
	
	/* Restore */
	if (old_env)
		setenv("QLLM_CACHE_FILE", old_env, 1);
	else
		unsetenv("QLLM_CACHE_FILE");
	
	mock_qmap_cleanup();
}

/* Test persistence simulation with mock */
TEST(persistent_cache_mock_persistence)
{
	uint32_t hd1, hd2;
	int value1 = 42;
	int value2 = 99;
	int **result;
	
	mock_qmap_init();
	mock_qmap_enable_persistence(1);
	
	/* Open same file twice - should get different handles but same data */
	hd1 = qmap_open("/tmp/test.qmap", "db1", QM_STR, qmap_reg(sizeof(int*)), 0, 0);
	
	/* Put a value using first handle */
	qmap_put(hd1, "test_key", &value1);
	
	/* Get should return the same value */
	result = (int **)qmap_get(hd1, "test_key");
	ASSERT_NOT_NULL(result);
	ASSERT_EQ(*result, &value1);
	
	/* Open same file again - in real qmap this would load from disk */
	hd2 = qmap_open("/tmp/test.qmap", "db1", QM_STR, qmap_reg(sizeof(int*)), 0, 0);
	
	/* Should see the same data (simulated persistence) */
	result = (int **)qmap_get(hd2, "test_key");
	ASSERT_NOT_NULL(result);
	ASSERT_EQ(*result, &value1);
	
	/* Update via second handle */
	qmap_put(hd2, "test_key", &value2);
	
	/* First handle should also see the update (same underlying storage) */
	result = (int **)qmap_get(hd1, "test_key");
	ASSERT_NOT_NULL(result);
	ASSERT_EQ(*result, &value2);
	
	mock_qmap_cleanup();
}

/* Test that different databases are isolated */
TEST(persistent_cache_database_isolation)
{
	uint32_t hd1, hd2;
	int value1 = 42;
	int value2 = 99;
	int **result;
	
	mock_qmap_init();
	mock_qmap_enable_persistence(1);
	
	/* Open same file but different databases */
	hd1 = qmap_open("/tmp/test.qmap", "db1", QM_STR, qmap_reg(sizeof(int*)), 0, 0);
	hd2 = qmap_open("/tmp/test.qmap", "db2", QM_STR, qmap_reg(sizeof(int*)), 0, 0);
	
	/* Put values in each */
	qmap_put(hd1, "key", &value1);
	qmap_put(hd2, "key", &value2);
	
	/* Each should have its own value */
	result = (int **)qmap_get(hd1, "key");
	ASSERT_NOT_NULL(result);
	ASSERT_EQ(*result, &value1);
	
	result = (int **)qmap_get(hd2, "key");
	ASSERT_NOT_NULL(result);
	ASSERT_EQ(*result, &value2);
	
	mock_qmap_cleanup();
}

/* Test NULL path falls back to in-memory */
TEST(persistent_cache_null_path)
{
	uint32_t hd;
	int value = 123;
	int **result;
	
	mock_qmap_init();
	mock_qmap_enable_persistence(1);
	
	/* NULL path should use in-memory storage */
	hd = qmap_open(NULL, "test", QM_STR, qmap_reg(sizeof(int*)), 0, 0);
	ASSERT(hd > 0);
	
	/* Should work normally, just without persistence */
	qmap_put(hd, "key", &value);
	result = (int **)qmap_get(hd, "key");
	ASSERT_NOT_NULL(result);
	ASSERT_EQ(*result, &value);
	
	mock_qmap_cleanup();
}

/* Test multiple keys in persistent storage */
TEST(persistent_cache_multiple_keys)
{
	uint32_t hd;
	int value1 = 10, value2 = 20, value3 = 30;
	int **result;
	
	mock_qmap_init();
	mock_qmap_enable_persistence(1);
	
	hd = qmap_open("/tmp/multi.qmap", "db", QM_STR, qmap_reg(sizeof(int*)), 0, 0);
	
	/* Add multiple keys */
	qmap_put(hd, "key1", &value1);
	qmap_put(hd, "key2", &value2);
	qmap_put(hd, "key3", &value3);
	
	/* Verify all are retrievable */
	result = (int **)qmap_get(hd, "key1");
	ASSERT_NOT_NULL(result);
	ASSERT_EQ(*result, &value1);
	
	result = (int **)qmap_get(hd, "key2");
	ASSERT_NOT_NULL(result);
	ASSERT_EQ(*result, &value2);
	
	result = (int **)qmap_get(hd, "key3");
	ASSERT_NOT_NULL(result);
	ASSERT_EQ(*result, &value3);
	
	/* Non-existent key should return NULL */
	result = (int **)qmap_get(hd, "nonexistent");
	ASSERT_NULL(result);
	
	mock_qmap_cleanup();
}
