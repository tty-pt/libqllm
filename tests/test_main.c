#include "framework/test_utils.h"
#include "mocks/llama_mock.h"
#include "mocks/gguf_mock.h"
#include "mocks/corm_mock.h"
#include "mocks/vulkan_mock.h"

int
main(int argc, char **argv)
{
	int ret;

	(void)argc;
	(void)argv;

	mock_llama_init();
	mock_gguf_init();
	mock_corm_init();
	mock_vulkan_init();

	printf("=== libqllm Test Suite ===\n\n");

	ret = run_all_tests();

	mock_vulkan_cleanup();
	mock_corm_cleanup();
	mock_gguf_cleanup();
	mock_llama_cleanup();

	return ret;
}