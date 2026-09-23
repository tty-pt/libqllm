#ifndef VULKAN_MOCK_H
#define VULKAN_MOCK_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void mock_vulkan_init(void);
void mock_vulkan_cleanup(void);
void mock_vulkan_set_total_vram(size_t bytes);
void mock_vulkan_set_free_vram(size_t bytes);
void mock_vulkan_set_gpu_count(int count);

void qllm_backend_mem_check(int gpu, size_t *free_b, size_t *total_b);
int qllm_backend_get_vram(size_t *free_b, size_t *total_b, int max_devices);

#ifdef __cplusplus
}
#endif

#endif