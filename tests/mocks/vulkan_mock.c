#include "vulkan_mock.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

static size_t _mock_total_vram = 8ULL * 1024 * 1024 * 1024; /* 8GB default */
static size_t _mock_free_vram = 4ULL * 1024 * 1024 * 1024;  /* 4GB default */
static int _mock_gpu_count = 1;
static int _mock_instance_create_fail = 0;
static int _mock_no_device_local = 0;

static int _instance_counter = 0;
static int _device_counter = 0;

void
mock_vulkan_init(void)
{
	_mock_total_vram = 8ULL * 1024 * 1024 * 1024;
	_mock_free_vram = 4ULL * 1024 * 1024 * 1024;
	_mock_gpu_count = 1;
	_mock_instance_create_fail = 0;
	_mock_no_device_local = 0;
	_instance_counter = 0;
	_device_counter = 0;
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
mock_vulkan_set_instance_create_fail(int should_fail)
{
	_mock_instance_create_fail = should_fail;
}

void
mock_vulkan_set_no_device_local_memory(int no_device_local)
{
	_mock_no_device_local = no_device_local;
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
	if (count > max_devices)
		count = max_devices;

	for (int i = 0; i < count; i++) {
		free_b[i] = _mock_free_vram;
		total_b[i] = _mock_total_vram;
	}
	return count;
}

VkResult
vkCreateInstance(const VkInstanceCreateInfo* pCreateInfo,
		 const void* pAllocator,
		 VkInstance* pInstance)
{
	if (_mock_instance_create_fail)
		return 1; /* not VK_SUCCESS */

	_instance_counter++;
	*pInstance = (VkInstance)(uintptr_t)_instance_counter;
	return VK_SUCCESS;
}

void
vkDestroyInstance(VkInstance instance, const void* pAllocator)
{
}

VkResult
vkEnumeratePhysicalDevices(VkInstance instance,
			   uint32_t* pPhysicalDeviceCount,
			   VkPhysicalDevice* pPhysicalDevices)
{
	if (pPhysicalDevices == NULL) {
		*pPhysicalDeviceCount = (uint32_t)_mock_gpu_count;
		return VK_SUCCESS;
	}

	uint32_t count = *pPhysicalDeviceCount;
	if (count > (uint32_t)_mock_gpu_count)
		count = (uint32_t)_mock_gpu_count;

	for (uint32_t i = 0; i < count; i++) {
		_device_counter++;
		pPhysicalDevices[i] = (VkPhysicalDevice)(uintptr_t)_device_counter;
	}

	*pPhysicalDeviceCount = count;
	return VK_SUCCESS;
}

void
vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice,
				    VkPhysicalDeviceMemoryProperties* pMemoryProperties)
{
	memset(pMemoryProperties, 0, sizeof(*pMemoryProperties));

	if (_mock_no_device_local) {
		pMemoryProperties->memoryHeapCount = 1;
		pMemoryProperties->memoryHeaps[0].size = _mock_total_vram;
		pMemoryProperties->memoryHeaps[0].flags = 0;
	} else {
		pMemoryProperties->memoryHeapCount = 2;
		pMemoryProperties->memoryHeaps[0].size = _mock_total_vram;
		pMemoryProperties->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
		pMemoryProperties->memoryHeaps[1].size = 16ULL * 1024 * 1024 * 1024;
		pMemoryProperties->memoryHeaps[1].flags = 0;
	}
}

void
vkGetPhysicalDeviceMemoryProperties2(VkPhysicalDevice physicalDevice,
                                     VkPhysicalDeviceMemoryProperties2* pMemoryProperties2)
{
    (void)pMemoryProperties2;
	VkPhysicalDeviceMemoryBudgetPropertiesEXT* budget = NULL;

    VkBaseOutStructure* chain = (VkBaseOutStructure*)pMemoryProperties2->pNext;
    while (chain) {
        /* Debug: observe extension chain sType values */
        if (chain->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT) {
            budget = (VkPhysicalDeviceMemoryBudgetPropertiesEXT*)chain;
            break;
        }
        chain = (VkBaseOutStructure*)chain->pNext;
    }

    if (!budget && pMemoryProperties2->pNext) {
        /* Be tolerant: if we couldn't find the budget by sType, try treating pNext as the budget struct. */
        budget = (VkPhysicalDeviceMemoryBudgetPropertiesEXT*)pMemoryProperties2->pNext;
    }

    if (budget) {
        memset(budget->heapBudget, 0, sizeof(budget->heapBudget));
        memset(budget->heapUsage, 0, sizeof(budget->heapUsage));

        if (!_mock_no_device_local) {
            budget->heapBudget[0] = _mock_free_vram + _mock_total_vram / 10;
            budget->heapUsage[0] = _mock_total_vram - _mock_free_vram;
        }
    }
    else {
            if (pMemoryProperties2->pNext == NULL) {
                /* nothing */
        } else {
                /* nothing */
        }
    }

    /* Fill base properties into a local struct to avoid overwriting the pNext chain
     * (tests place the VkPhysicalDeviceMemoryBudgetPropertiesEXT directly after
     * the VkPhysicalDeviceMemoryProperties2 on the stack; calling the base
     * function with the pMemoryProperties2 pointer would overwrite that memory). */
    VkPhysicalDeviceMemoryProperties base_props;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &base_props);

    /* As a final fallback, if the test passed a budget struct directly in pNext
     * but we didn't detect it via the sType chain search, populate it anyway. */
    if (pMemoryProperties2->pNext) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT* bud = (VkPhysicalDeviceMemoryBudgetPropertiesEXT*)pMemoryProperties2->pNext;
        if (!_mock_no_device_local) {
            /* Overwrite/initialize budget fields unconditionally as a robust fallback. */
            memset(bud->heapBudget, 0, sizeof(bud->heapBudget));
            memset(bud->heapUsage, 0, sizeof(bud->heapUsage));
            bud->heapBudget[0] = _mock_free_vram + _mock_total_vram / 10;
            bud->heapUsage[0] = _mock_total_vram - _mock_free_vram;
            /* fallback populated */
        }
    }
}
