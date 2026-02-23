#include "../framework/test_utils.h"
#include "../mocks/ndc_mock.h"
#include "../mocks/llama_mock.h"
#include "../mocks/qmap_mock.h"

TEST(ndc_write_capture)
{
	char buf[256] = {0};

	mock_ndc_init();
	mock_ndc_set_write_capture(buf, sizeof(buf));

	ndc_write(0, "hello", 5);

	ASSERT_STR_EQ(buf, "hello");

	mock_ndc_cleanup();
}

TEST(ndc_write_multiple)
{
	char buf[256] = {0};

	mock_ndc_init();
	mock_ndc_set_write_capture(buf, sizeof(buf));

	ndc_write(0, "hello", 5);
	ndc_write(0, " world", 6);

	ASSERT_STR_EQ(buf, "hello world");

	mock_ndc_cleanup();
}

TEST(ndc_write_overflow_protection)
{
	char buf[10] = {0};

	mock_ndc_init();
	mock_ndc_set_write_capture(buf, sizeof(buf));

	ndc_write(0, "this is a very long string", 26);

	ASSERT_EQ(strlen(buf), 9);
	ASSERT_STR_EQ(buf, "this is a");

	mock_ndc_cleanup();
}

TEST(ndc_write_empty)
{
	char buf[256] = {0};

	mock_ndc_init();
	mock_ndc_set_write_capture(buf, sizeof(buf));

	ndc_write(0, "", 0);

	ASSERT_STR_EQ(buf, "");

	mock_ndc_cleanup();
}

TEST(ndc_write_null_capture)
{
	mock_ndc_init();
	mock_ndc_set_write_capture(NULL, 0);

	ndc_write(0, "test", 4);

	ASSERT_TRUE(1);

	mock_ndc_cleanup();
}

TEST(ndc_writef_basic)
{
	char buf[256] = {0};

	mock_ndc_init();
	mock_ndc_set_write_capture(buf, sizeof(buf));

	ndc_writef(0, "value: %d", 42);

	ASSERT_STR_EQ(buf, "value: 42");

	mock_ndc_cleanup();
}

TEST(ndc_register_cmd)
{
	mock_ndc_init();

	ndc_register("test_cmd", NULL, 0);
	ndc_register("another_cmd", NULL, CF_NOAUTH);

	ASSERT_TRUE(1);

	mock_ndc_cleanup();
}

TEST(ndc_mmap_returns_null)
{
	char *ptr = NULL;
	size_t size;

	mock_ndc_init();

	size = ndc_mmap(&ptr, "nonexistent.txt");

	ASSERT_EQ(size, 0);
	ASSERT_NULL(ptr);

	mock_ndc_cleanup();
}

TEST(ndc_main_returns_zero)
{
	mock_ndc_init();

	int ret = ndc_main();

	ASSERT_EQ(ret, 0);

	mock_ndc_cleanup();
}
