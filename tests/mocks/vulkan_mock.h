#ifndef VULKAN_MOCK_H
#define VULKAN_MOCK_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VK_SUCCESS 0
#define VK_NULL_HANDLE ((void*)0)

#define VK_STRUCTURE_TYPE_APPLICATION_INFO 0
#define VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO 1
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2 2
#define VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT 3

#define VK_API_VERSION_1_1 0
#define VK_MAKE_VERSION(major, minor, patch) (((major) << 22) | ((minor) << 12) | (patch))

#define VK_MEMORY_HEAP_DEVICE_LOCAL_BIT 0x00000001

#define VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME "VK_KHR_get_physical_device_properties2"

typedef void* VkInstance;
typedef void* VkPhysicalDevice;
typedef uint32_t VkFlags;
typedef VkFlags VkMemoryHeapFlags;

typedef struct {
	uint32_t    sType;
	const void* pNext;
} VkBaseOutStructure;

typedef struct {
	uint32_t    sType;
	const void* pNext;
	const char* pApplicationName;
	uint32_t    applicationVersion;
	const char* pEngineName;
	uint32_t    engineVersion;
	uint32_t    apiVersion;
} VkApplicationInfo;

typedef struct {
	uint32_t    sType;
	const void* pNext;
	const VkApplicationInfo* pApplicationInfo;
	uint32_t    enabledLayerCount;
	const char* const* ppEnabledLayerNames;
	uint32_t    enabledExtensionCount;
	const char* const* ppEnabledExtensionNames;
} VkInstanceCreateInfo;

typedef struct {
	uint64_t     size;
	VkMemoryHeapFlags flags;
} VkMemoryHeap;

typedef struct {
	uint32_t        memoryTypeCount;
	uint32_t        memoryHeapCount;
	VkMemoryHeap    memoryHeaps[16];
} VkPhysicalDeviceMemoryProperties;

typedef struct {
	uint32_t    sType;
	void*       pNext;
} VkPhysicalDeviceMemoryProperties2;

typedef struct {
	uint32_t    sType;
	void*       pNext;
	uint64_t    heapBudget[16];
	uint64_t    heapUsage[16];
} VkPhysicalDeviceMemoryBudgetPropertiesEXT;

typedef int VkResult;

void mock_vulkan_init(void);
void mock_vulkan_cleanup(void);

void mock_vulkan_set_total_vram(size_t bytes);
void mock_vulkan_set_free_vram(size_t bytes);
void mock_vulkan_set_gpu_count(int count);
void mock_vulkan_set_instance_create_fail(int should_fail);
void mock_vulkan_set_no_device_local_memory(int no_device_local);

void qllm_backend_mem_check(int gpu, size_t *free_b, size_t *total_b);

VkResult vkCreateInstance(const VkInstanceCreateInfo* pCreateInfo,
			  const void* pAllocator,
			  VkInstance* pInstance);
void vkDestroyInstance(VkInstance instance, const void* pAllocator);

VkResult vkEnumeratePhysicalDevices(VkInstance instance,
				    uint32_t* pPhysicalDeviceCount,
				    VkPhysicalDevice* pPhysicalDevices);

void vkGetPhysicalDeviceMemoryProperties(VkPhysicalDevice physicalDevice,
					 VkPhysicalDeviceMemoryProperties* pMemoryProperties);

void vkGetPhysicalDeviceMemoryProperties2(VkPhysicalDevice physicalDevice,
					  VkPhysicalDeviceMemoryProperties2* pMemoryProperties2);

#ifdef __cplusplus
}
#endif

#endif
