#include <vulkan/vulkan.h>
#include <span>;

namespace dsl
{

	struct DescriptorAllocator
	{
		struct PoolSizeRatio
		{
			VkDescriptorType type;
			float ratio;
		};

		VkDescriptorPool pool;

		void initPool(VkDevice device, uint32_t maxSets, std::span<PoolSizeRatio> poolRatios);
		void clearDescriptors(VkDevice device);
		void destroyPool(VkDevice device);

		void allocate(VkDevice device, VkDescriptorSetLayout* layout, int32_t setCount, VkDescriptorSet* sets);
	};
}