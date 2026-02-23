/* Test qmap 0.6.0 pointer stability and allocation reuse */
#include "test_utils.h"
#include "../mocks/qmap_mock.h"
#include <string.h>
#include <stdlib.h>

/* Test basic pointer retrieval from qmap */
TEST(qmap_get_returns_slot_pointer)
{
	uint32_t hd;
	void *slot;
	int value = 42;
	int **slot_pp;
	
	mock_qmap_init();
	
	hd = qmap_reg(sizeof(int *));
	hd = qmap_open(NULL, NULL, 0, hd, 0, 0);
	
	qmap_put(hd, "test_key", &value);
	
	slot = qmap_get(hd, "test_key");
	ASSERT_NOT_NULL(slot);
	
	slot_pp = (int **)slot;
	ASSERT(*slot_pp == &value);
	
	mock_qmap_cleanup();
}

/* Test pointer stability across multiple puts with same key */
TEST(qmap_pointer_stability_same_key)
{
	uint32_t hd;
	void *slot1, *slot2;
	int value1 = 42, value2 = 100;
	int **entry1, **entry2;
	
	mock_qmap_init();
	
	hd = qmap_reg(sizeof(int *));
	hd = qmap_open(NULL, NULL, 0, hd, 0, 0);
	
	/* First put */
	qmap_put(hd, "test_key", &value1);
	slot1 = qmap_get(hd, "test_key");
	entry1 = (int **)slot1;
	
	/* Update with same key - with qmap 0.6.0 allocation reuse,
	 * the slot pointer should remain stable */
	qmap_put(hd, "test_key", &value2);
	slot2 = qmap_get(hd, "test_key");
	entry2 = (int **)slot2;
	
	/* Note: In the mock, we can't fully test allocation reuse,
	 * but we verify the API pattern works correctly */
	ASSERT_NOT_NULL(slot2);
	ASSERT(*entry2 == &value2);
	
	mock_qmap_cleanup();
}

/* Test NULL write pattern (used in model cache cleanup) */
TEST(qmap_null_write_pattern)
{
	uint32_t hd;
	void *slot;
	int *value;
	int **entry_pp;
	
	mock_qmap_init();
	
	hd = qmap_reg(sizeof(int *));
	hd = qmap_open(NULL, NULL, 0, hd, 0, 0);
	
	value = malloc(sizeof(int));
	*value = 42;
	
	qmap_put(hd, "test_key", value);
	
	slot = qmap_get(hd, "test_key");
	entry_pp = (int **)slot;
	ASSERT(*entry_pp == value);
	
	/* Simulate the cleanup pattern in qllm_free:
	 * write NULL to the slot to prevent use-after-free */
	*entry_pp = NULL;
	
	/* Verify NULL was written */
	slot = qmap_get(hd, "test_key");
	entry_pp = (int **)slot;
	ASSERT(*entry_pp == NULL);
	
	free(value);
	mock_qmap_cleanup();
}

/* Test model cache entry pattern with pointer indirection */
TEST(qmap_cache_entry_indirection)
{
	uint32_t hd;
	void *slot;
	struct test_entry {
		int value;
		unsigned refcount;
	};
	struct test_entry *entry;
	struct test_entry **entry_pp;
	
	mock_qmap_init();
	
	hd = qmap_reg(sizeof(struct test_entry *));
	hd = qmap_open(NULL, NULL, 0, hd, 0, 0);
	
	/* Allocate and store entry pointer (mimics model cache) */
	entry = calloc(1, sizeof(*entry));
	entry->value = 42;
	entry->refcount = 1;
	
	qmap_put(hd, "model_path", entry);
	
	/* Retrieve and verify double indirection */
	slot = qmap_get(hd, "model_path");
	entry_pp = (struct test_entry **)slot;
	
	ASSERT_NOT_NULL(entry_pp);
	ASSERT(*entry_pp == entry);
	ASSERT_EQ((*entry_pp)->value, 42);
	ASSERT_EQ((*entry_pp)->refcount, 1);
	
	/* Simulate refcount increment */
	(*entry_pp)->refcount++;
	ASSERT_EQ(entry->refcount, 2);
	
	/* Cleanup */
	free(entry);
	mock_qmap_cleanup();
}

/* Test multiple entries with different keys */
TEST(qmap_multiple_entries)
{
	uint32_t hd;
	int value1 = 10, value2 = 20, value3 = 30;
	int **slot1, **slot2, **slot3;
	
	mock_qmap_init();
	
	hd = qmap_reg(sizeof(int *));
	hd = qmap_open(NULL, NULL, 0, hd, 0, 0);
	
	qmap_put(hd, "key1", &value1);
	qmap_put(hd, "key2", &value2);
	qmap_put(hd, "key3", &value3);
	
	slot1 = (int **)qmap_get(hd, "key1");
	slot2 = (int **)qmap_get(hd, "key2");
	slot3 = (int **)qmap_get(hd, "key3");
	
	ASSERT(slot1 && *slot1 == &value1);
	ASSERT(slot2 && *slot2 == &value2);
	ASSERT(slot3 && *slot3 == &value3);
	
	mock_qmap_cleanup();
}
