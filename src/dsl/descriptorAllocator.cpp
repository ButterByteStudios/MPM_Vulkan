#include <descriptorAllocator.h>;
#include <stdexcept>
#include <vector>

namespace dsl
{
	void DescriptorAllocator::initPool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios)
	{
		std::vector<VkDescriptorPoolSize> poolSizes;
		for (PoolSizeRatio ratio : poolRatios)
		{
			VkDescriptorPoolSize poolSize{};
			poolSize.type = ratio.type;
			poolSize.descriptorCount = uint32_t(ratio.ratio * maxSets);

			poolSizes.push_back(poolSize);
		}

		VkDescriptorPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		poolInfo.flags = 0;
		poolInfo.maxSets = maxSets;
		poolInfo.poolSizeCount = (uint32_t)poolSizes.size();
		poolInfo.pPoolSizes = poolSizes.data();

		vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool);
	}

	void DescriptorAllocator::clearDescriptors(VkDevice device)
	{
		vkResetDescriptorPool(device, pool, 0);
	}

	void DescriptorAllocator::destroyPool(VkDevice device)
	{
		vkDestroyDescriptorPool(device, pool, nullptr);
	}

	void DescriptorAllocator::allocate(VkDevice device, VkDescriptorSetLayout* layouts, int32_t setCount, VkDescriptorSet* sets)
	{
		VkDescriptorSetAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = pool;
		allocInfo.descriptorSetCount = setCount;
		allocInfo.pSetLayouts = layouts;

		if (vkAllocateDescriptorSets(device, &allocInfo, sets) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to allocate descriptorset");
		}
	}
}