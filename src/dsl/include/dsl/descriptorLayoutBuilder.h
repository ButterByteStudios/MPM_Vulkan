#pragma once

#include <vector>;
#include <vulkan/vulkan.h>

namespace dsl
{
	struct DescriptorLayoutBuilder
	{
		std::vector<VkDescriptorSetLayoutBinding> layoutBindings;

		void addBinding(uint32_t binding, VkDescriptorType type);
		void clear();
		void build(VkDevice device, VkShaderStageFlags shaderStages, VkDescriptorSetLayout& setLayout, void* pNext = nullptr, VkDescriptorSetLayoutCreateFlags flags = 0);
	};
}