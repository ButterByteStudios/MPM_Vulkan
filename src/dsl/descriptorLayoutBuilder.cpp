#include <descriptorLayoutBuilder.h>;
#include <stdexcept>

namespace dsl
{
	void DescriptorLayoutBuilder::addBinding(uint32_t binding, VkDescriptorType type)
	{
		VkDescriptorSetLayoutBinding newBind{};
		newBind.binding = binding;
		newBind.descriptorCount = 1;
		newBind.descriptorType = type;
		newBind.pImmutableSamplers = nullptr;

		layoutBindings.push_back(newBind);
	}

	void DescriptorLayoutBuilder::clear()
	{
		layoutBindings.clear();
	}

	void DescriptorLayoutBuilder::build(VkDevice device, VkShaderStageFlags shaderStages, VkDescriptorSetLayout& setLayout, void* pNext, VkDescriptorSetLayoutCreateFlags flags)
	{
		for (auto& b : layoutBindings)
		{
			b.stageFlags |= shaderStages;
		}

		VkDescriptorSetLayoutCreateInfo info{};
		info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		info.pNext = pNext;

		info.pBindings = layoutBindings.data();
		info.bindingCount = (uint32_t)layoutBindings.size();
		info.flags = flags;

		if (vkCreateDescriptorSetLayout(device, &info, nullptr, &setLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create descriptor set layout.");
		}
	}
}