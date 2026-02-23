#include "../framework/test_utils.h"
#include "../mocks/vulkan_mock.h"

TEST(vulkan_mem_check_basic)
{
	size_t free_b, total_b;

	mock_vulkan_init();
	mock_vulkan_set_total_vram(8ULL * 1024 * 1024 * 1024);
	mock_vulkan_set_free_vram(4ULL * 1024 * 1024 * 1024);

	VkInstance inst;
	VkInstanceCreateInfo ci = {0};
	VkResult res = vkCreateInstance(&ci, NULL, &inst);

	ASSERT_EQ(res, VK_SUCCESS);
	ASSERT_TRUE(inst != VK_NULL_HANDLE);

	vkDestroyInstance(inst, NULL);
}

TEST(vulkan_enumerate_devices)
{
	mock_vulkan_init();
	mock_vulkan_set_gpu_count(2);

	VkInstance inst;
	VkInstanceCreateInfo ci = {0};
	VkResult res = vkCreateInstance(&ci, NULL, &inst);

	ASSERT_EQ(res, VK_SUCCESS);

	uint32_t count = 0;
	res = vkEnumeratePhysicalDevices(inst, &count, NULL);

	ASSERT_EQ(res, VK_SUCCESS);
	ASSERT_EQ(count, 2);

	VkPhysicalDevice devs[2];
	res = vkEnumeratePhysicalDevices(inst, &count, devs);

	ASSERT_EQ(res, VK_SUCCESS);
	ASSERT_EQ(count, 2);

	vkDestroyInstance(inst, NULL);
}

TEST(vulkan_no_gpus)
{
	mock_vulkan_init();
	mock_vulkan_set_gpu_count(0);

	VkInstance inst;
	VkInstanceCreateInfo ci = {0};
	VkResult res = vkCreateInstance(&ci, NULL, &inst);

	ASSERT_EQ(res, VK_SUCCESS);

	uint32_t count = 0;
	res = vkEnumeratePhysicalDevices(inst, &count, NULL);

	ASSERT_EQ(res, VK_SUCCESS);
	ASSERT_EQ(count, 0);

	vkDestroyInstance(inst, NULL);
}

TEST(vulkan_instance_create_fail)
{
	mock_vulkan_init();
	mock_vulkan_set_instance_create_fail(1);

	VkInstance inst;
	VkInstanceCreateInfo ci = {0};
	VkResult res = vkCreateInstance(&ci, NULL, &inst);

	ASSERT_TRUE(res != VK_SUCCESS);

	mock_vulkan_set_instance_create_fail(0);
}

TEST(vulkan_memory_properties)
{
	mock_vulkan_init();
	mock_vulkan_set_total_vram(8ULL * 1024 * 1024 * 1024);
	mock_vulkan_set_free_vram(4ULL * 1024 * 1024 * 1024);

	VkInstance inst;
	VkInstanceCreateInfo ci = {0};
	vkCreateInstance(&ci, NULL, &inst);

	uint32_t count = 0;
	vkEnumeratePhysicalDevices(inst, &count, NULL);

	VkPhysicalDevice dev;
	vkEnumeratePhysicalDevices(inst, &count, &dev);

	VkPhysicalDeviceMemoryProperties props;
	vkGetPhysicalDeviceMemoryProperties(dev, &props);

	ASSERT_TRUE(props.memoryHeapCount > 0);
	ASSERT_TRUE(props.memoryHeaps[0].size > 0);
	ASSERT_TRUE(props.memoryHeaps[0].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);

	vkDestroyInstance(inst, NULL);
}

TEST(vulkan_memory_budget)
{
    mock_vulkan_init();
    mock_vulkan_set_total_vram(8ULL * 1024 * 1024 * 1024);
    mock_vulkan_set_free_vram(4ULL * 1024 * 1024 * 1024);

    VkInstance inst;
    VkInstanceCreateInfo ci = {0};
    VkResult res = vkCreateInstance(&ci, NULL, &inst);
    ASSERT_EQ(res, VK_SUCCESS);

    uint32_t count = 1;
    vkEnumeratePhysicalDevices(inst, &count, NULL);
    VkPhysicalDevice dev;
    vkEnumeratePhysicalDevices(inst, &count, &dev);

    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget;
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    budget.pNext = NULL;

    VkPhysicalDeviceMemoryProperties2 props2;
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    props2.pNext = &budget;

    vkGetPhysicalDeviceMemoryProperties2(dev, &props2);

    /* Basic sanity checks: budget and usage should be populated and budget >= usage */
    ASSERT_TRUE(budget.heapBudget[0] > 0);
    ASSERT_TRUE(budget.heapBudget[0] >= budget.heapUsage[0]);

    vkDestroyInstance(inst, NULL);
}

TEST(vulkan_no_device_local_memory)
{
	mock_vulkan_init();
	mock_vulkan_set_no_device_local_memory(1);

	VkInstance inst;
	VkInstanceCreateInfo ci = {0};
	vkCreateInstance(&ci, NULL, &inst);

	uint32_t count = 1;
	VkPhysicalDevice dev;
	vkEnumeratePhysicalDevices(inst, &count, &dev);

	VkPhysicalDeviceMemoryProperties props;
	vkGetPhysicalDeviceMemoryProperties(dev, &props);

	ASSERT_TRUE(props.memoryHeapCount > 0);
	ASSERT_FALSE(props.memoryHeaps[0].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);

	vkDestroyInstance(inst, NULL);
	mock_vulkan_set_no_device_local_memory(0);
}

TEST(vulkan_small_vram)
{
	mock_vulkan_init();
	mock_vulkan_set_total_vram(256 * 1024 * 1024);
	mock_vulkan_set_free_vram(128 * 1024 * 1024);

	VkInstance inst;
	VkInstanceCreateInfo ci = {0};
	vkCreateInstance(&ci, NULL, &inst);

	uint32_t count = 1;
	VkPhysicalDevice dev;
	vkEnumeratePhysicalDevices(inst, &count, &dev);

	VkPhysicalDeviceMemoryProperties props;
	vkGetPhysicalDeviceMemoryProperties(dev, &props);

	ASSERT_TRUE(props.memoryHeaps[0].size == 256 * 1024 * 1024);

	vkDestroyInstance(inst, NULL);
}

TEST(vulkan_large_vram)
{
	mock_vulkan_init();
	mock_vulkan_set_total_vram(24ULL * 1024 * 1024 * 1024);
	mock_vulkan_set_free_vram(16ULL * 1024 * 1024 * 1024);

	VkInstance inst;
	VkInstanceCreateInfo ci = {0};
	vkCreateInstance(&ci, NULL, &inst);

	uint32_t count = 1;
	VkPhysicalDevice dev;
	vkEnumeratePhysicalDevices(inst, &count, &dev);

	VkPhysicalDeviceMemoryProperties props;
	vkGetPhysicalDeviceMemoryProperties(dev, &props);

	ASSERT_TRUE(props.memoryHeaps[0].size == 24ULL * 1024 * 1024 * 1024);

	vkDestroyInstance(inst, NULL);
}
