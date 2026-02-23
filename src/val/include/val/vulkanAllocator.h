#include <vulkan/vulkan.h>

namespace val
{
	enum class BufferUsage
	{
		Vertex,
		Uniform,
		Storage,
		VertexStorage,
		Indirect,
		Staging
	};

	enum class BufferLifetime
	{
		Static,
		Dynamic
	};

	struct BufferInfo
	{
		VkDeviceSize size;
		BufferUsage usage;
		BufferLifetime lifetime;
	};

	struct AllocatedBuffer
	{
		VkBuffer buffer;
		VkDeviceMemory memory;
		VkDeviceSize size;
		VkDevice device;
		void* mapped;

		void dispose();
	};

	class VulkanAllocator
	{
	public:
		VulkanAllocator(VkDevice device, VkPhysicalDevice physicalDevice);
		VulkanAllocator();

		AllocatedBuffer create(BufferInfo info);

	private:
		struct AllocateInfo
		{
			VkBufferUsageFlags usage;
			VkMemoryPropertyFlags properties;
		};

		VkPhysicalDevice physicalDevice;
		VkDevice device;

		AllocateInfo toAllocInfo(BufferInfo info);

		uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
	};
}