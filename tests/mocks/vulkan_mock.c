#include "vulkan_mock.h"
#include <stdlib.h>
#include <string.h>

static size_t _mock_total_vram = 8ULL * 1024 * 1024 * 1024;
static size_t _mock_free_vram = 4ULL * 1024 * 1024 * 1024;
static int _mock_gpu_count = 1;

void
mock_vulkan_init(void)
{
	_mock_total_vram = 8ULL * 1024 * 1024 * 1024;
	_mock_free_vram = 4ULL * 1024 * 1024 * 1024;
	_mock_gpu_count = 1;
}

void
mock_vulkan_cleanup(void)
{
}

void
mock_vulkan_set_total_vram(size_t bytes)
{
	_mock_total_vram = bytes;
}

void
mock_vulkan_set_free_vram(size_t bytes)
{
	_mock_free_vram = bytes;
}

void
mock_vulkan_set_gpu_count(int count)
{
	_mock_gpu_count = count;
}

void
qllm_backend_mem_check(int gpu, size_t *free_b, size_t *total_b)
{
	(void)gpu;
	*free_b = _mock_free_vram;
	*total_b = _mock_total_vram;
}

int
qllm_backend_get_vram(size_t *free_b, size_t *total_b, int max_devices)
{
	int count = _mock_gpu_count;
	int i;

	if (!free_b || !total_b || max_devices <= 0)
		return 0;
	if (count > max_devices)
		count = max_devices;
	if (count < 0)
		count = 0;

	for (i = 0; i < count; i++) {
		free_b[i] = _mock_free_vram;
		total_b[i] = _mock_total_vram;
	}
	return count;
}