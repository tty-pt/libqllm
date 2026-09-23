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

/* Query free/total VRAM for a single physical device. */
static void
device_mem_check(VkPhysicalDevice dev, size_t *free_b, size_t *total_b)
{
	VkPhysicalDeviceMemoryProperties mem;
	uint32_t heap_index = UINT32_MAX;
	uint32_t i;

	*free_b = 0;
	*total_b = 0;

	vkGetPhysicalDeviceMemoryProperties(dev, &mem);

	for (i = 0; i < mem.memoryHeapCount; i++) {
		if (mem.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
			heap_index = i;
			break;
		}
	}

	if (heap_index == UINT32_MAX)
		return;

	*total_b = mem.memoryHeaps[heap_index].size;

	/* Attempt to get actual free memory via VK_EXT_memory_budget. */
	VkPhysicalDeviceMemoryBudgetPropertiesEXT budget = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT
	};

	VkPhysicalDeviceMemoryProperties2 props2 = {
		.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2,
		.pNext = &budget
	};

	vkGetPhysicalDeviceMemoryProperties2(dev, &props2);

	if (budget.heapBudget[heap_index] > 0) {
		size_t used = (size_t) budget.heapUsage[heap_index];
		size_t budget_b = (size_t) budget.heapBudget[heap_index];

		if (budget_b > used)
			*free_b = budget_b - used;
		else
			*free_b = 0;
	}
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
	VkInstance inst = vk_create_instance();
	VkPhysicalDevice dev;

	if (!inst)
		return;

	dev = vk_get_gpu(inst, gpu);
	if (!dev) {
		vkDestroyInstance(inst, NULL);
		return;
	}

	device_mem_check(dev, free_b, total_b);

	vkDestroyInstance(inst, NULL);
}

/*
 * Enumerate VRAM across up to max_devices physical GPUs.
 * Fills free_b[i]/total_b[i] per device and returns the device count.
 */
int
qllm_backend_get_vram(size_t *free_b, size_t *total_b, int max_devices)
{
	VkInstance inst = vk_create_instance();
	uint32_t count = 0;
	uint32_t n = 0;

	if (!free_b || !total_b || max_devices <= 0) {
		if (inst)
			vkDestroyInstance(inst, NULL);
		return 0;
	}

	if (!inst)
		return 0;

	vkEnumeratePhysicalDevices(inst, &count, NULL);
	if (count == 0) {
		vkDestroyInstance(inst, NULL);
		return 0;
	}

	if (count > (uint32_t)max_devices)
		count = (uint32_t)max_devices;

	for (n = 0; n < count; ++n) {
		VkPhysicalDevice dev = vk_get_gpu(inst, (int)n);

		if (!dev)
			break;
		device_mem_check(dev, &free_b[n], &total_b[n]);
	}

	vkDestroyInstance(inst, NULL);
	return (int)n;
}
