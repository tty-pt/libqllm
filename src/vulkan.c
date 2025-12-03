/* vulkan.c */

#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* 
 * Minimal Vulkan instance creation. 
 * Only needed to query memory heaps.
 */
static VkInstance
vk_create_instance(void)
{
	VkApplicationInfo app = {
		.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
		.pApplicationName = "qllm",
		.applicationVersion = VK_MAKE_VERSION(1,0,0),
		.pEngineName = "none",
		.engineVersion = VK_MAKE_VERSION(1,0,0),
		.apiVersion = VK_API_VERSION_1_1,
	};

	const char *exts[] = {
		VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
	};

	VkInstanceCreateInfo ci = {
		.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
		.pApplicationInfo = &app,
		.enabledExtensionCount = 1,
		.ppEnabledExtensionNames = exts,
	};

	VkInstance instance;
	if (vkCreateInstance(&ci, NULL, &instance) != VK_SUCCESS)
		return VK_NULL_HANDLE;

	return instance;
}

static VkPhysicalDevice
vk_get_gpu(VkInstance inst, int index)
{
	uint32_t count = 0;
	vkEnumeratePhysicalDevices(inst, &count, NULL);
	if (count == 0)
		return VK_NULL_HANDLE;

	VkPhysicalDevice *list = malloc(sizeof(*list) * count);
	vkEnumeratePhysicalDevices(inst, &count, list);

	VkPhysicalDevice dev = VK_NULL_HANDLE;
	if (index >= 0 && index < (int)count)
		dev = list[index];

	free(list);
	return dev;
}

/*
 * This is the function you asked for.
 *
 * - gpu = GPU index (0 = first GPU)
 * - free_b  = output: free VRAM in bytes (if available)
 * - total_b = output: total VRAM in bytes
 */
void
qllm_backend_mem_check(int gpu, size_t *free_b, size_t *total_b)
{
	*free_b = 0;
	*total_b = 0;

	VkInstance inst = vk_create_instance();
	if (!inst)
		return;

	VkPhysicalDevice dev = vk_get_gpu(inst, gpu);
	if (!dev) {
		vkDestroyInstance(inst, NULL);
		return;
	}

	/* Query memory heaps. */
	VkPhysicalDeviceMemoryProperties mem;
	vkGetPhysicalDeviceMemoryProperties(dev, &mem);

	/* First: find the DEVICE_LOCAL heap (VRAM). */
	uint32_t heap_index = UINT32_MAX;

	for (uint32_t i = 0; i < mem.memoryHeapCount; i++) {
		if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			heap_index = i;
			break;
		}
	}

	if (heap_index == UINT32_MAX) {
		/* No dedicated VRAM (iGPU). Report zero. */
		vkDestroyInstance(inst, NULL);
		return;
	}

	*total_b = mem.memoryHeaps[heap_index].size;

	/*
	 * Attempt to get actual free memory via VK_EXT_memory_budget.
	 * If not available, free_b will remain 0.
	 */

	/* Prepare structure chain. */
	VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
	};

	VkPhysicalDeviceMemoryProperties2 props2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
		.pNext = &budget
	};

	/* Query. Will work only if extension is supported. */
	vkGetPhysicalDeviceMemoryProperties2(dev, &props2);

	/* If heapBudget contains nonzero, extension is supported. */
	if (budget.heapBudget[heap_index] > 0) {
		size_t used = (size_t) budget.heapUsage[heap_index];
		size_t budget_b = (size_t) budget.heapBudget[heap_index];

		/*
		 * Vulkan "budget" is the safe amount you can allocate.
		 * Free VRAM = budget - currently used.
		 */
		if (budget_b > used)
			*free_b = budget_b - used;
		else
			*free_b = 0;
	} else {
		/* No VK_EXT_memory_budget: best fallback is total only. */
		*free_b = 0;
	}

	vkDestroyInstance(inst, NULL);
}
