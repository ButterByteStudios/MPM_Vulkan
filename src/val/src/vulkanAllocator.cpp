#include <val/vulkanAllocator.h>
#include <stdexcept>

namespace val
{
	void AllocatedBuffer::dispose()
	{
		if (mapped != nullptr)
		{
			vkUnmapMemory(device, memory);
		}

		vkDestroyBuffer(device, buffer, nullptr);
		vkFreeMemory(device, memory, nullptr);
	}

	VulkanAllocator::VulkanAllocator(VkDevice device, VkPhysicalDevice physicalDevice)
	{
		this->device = device;
		this->physicalDevice = physicalDevice;
	}

	VulkanAllocator::VulkanAllocator()
	{
		this->device = nullptr;
		this->physicalDevice = nullptr;
	}

	AllocatedBuffer VulkanAllocator::create(BufferInfo info)
	{
		AllocatedBuffer alloc{};
		alloc.size = info.size;
		alloc.device = device;

		AllocateInfo allocInfo = toAllocInfo(info);

		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = info.size;
		bufferInfo.usage = allocInfo.usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &alloc.buffer) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create buffer.");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, alloc.buffer, &memRequirements);

		VkMemoryAllocateInfo memAllocInfo{};
		memAllocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		memAllocInfo.allocationSize = memRequirements.size;
		memAllocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, allocInfo.properties);

		if (vkAllocateMemory(device, &memAllocInfo, nullptr, &alloc.memory) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to allocate buffer memory.");
		}

		vkBindBufferMemory(device, alloc.buffer, alloc.memory, 0);

		if ((allocInfo.properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0)
		{
			vkMapMemory(device, alloc.memory, 0, info.size, 0, &alloc.mapped);
		}

		return alloc;
	}

	uint32_t VulkanAllocator::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

		for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
		{
			if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("Failed to find suitable memory type.");
	}

	VulkanAllocator::AllocateInfo VulkanAllocator::toAllocInfo(BufferInfo info)
	{
		AllocateInfo allocInfo{};

		switch (info.usage)
		{
		case BufferUsage::Vertex:
			allocInfo.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
			allocInfo.properties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			break;
		case BufferUsage::Storage:
			allocInfo.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			allocInfo.properties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			break;
		case BufferUsage::VertexStorage:
			allocInfo.usage |= VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
			allocInfo.properties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
			break;
		case BufferUsage::Uniform:
			allocInfo.usage |= VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
			allocInfo.properties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			break;
		case BufferUsage::Staging:
			allocInfo.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
			allocInfo.properties |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
			break;
		case BufferUsage::Indirect:
			allocInfo.usage |= VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
			allocInfo.properties |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
		}

		switch (info.lifetime)
		{
		case BufferLifetime::Static:
			allocInfo.usage |= VK_BUFFER_USAGE_TRANSFER_DST_BIT;
			break;
		}

		return allocInfo;
	}
}