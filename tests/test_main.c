#include "framework/test_utils.h"
#include "framework/mock_registry.h"

#include "mocks/llama_mock.h"
#include "mocks/gguf_mock.h"
#include "mocks/qmap_mock.h"
#include "mocks/ndc_mock.h"
#include "mocks/vulkan_mock.h"

int
main(int argc, char **argv)
{
	int ret;

	(void)argc;
	(void)argv;

	mock_init();
	mock_llama_init();
	mock_gguf_init();
	mock_qmap_init();
	mock_ndc_init();
	mock_vulkan_init();

	printf("=== libqllm Test Suite ===\n\n");

	ret = run_all_tests();

	mock_llama_cleanup();
	mock_gguf_cleanup();
	mock_qmap_cleanup();
	mock_ndc_cleanup();
	mock_vulkan_cleanup();
	mock_cleanup();

	return ret;
}
