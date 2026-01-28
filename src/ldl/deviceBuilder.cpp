#include <deviceBuilder.h>
#include <vector>
#include <stdexcept>
#include <set>

namespace ldl
{
	void DeviceBuilder::addQueue(VkQueueFlags flags, VkQueueFlags excludedFlags, float priority, bool supportsPresent)
	{
		QueueRequest queueRequest = {
			flags,
			excludedFlags,
			priority,
			supportsPresent
		};

		queueRequests.push_back(queueRequest);
	}

	bool DeviceBuilder::isDeviceSuitable(VkPhysicalDevice physicalDevice)
	{
		bool extensionsSupported = checkDeviceExtensionSupport(physicalDevice);

		bool swapChainAdequate = false;
		bool subgroupsSupported = false;
		if (extensionsSupported)
		{
			SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);
			swapChainAdequate = swapChainSupport.isComplete();

			subgroupsSupported = checkDeviceSubgroupSupport(physicalDevice);
		}

		bool hasIndices = true;
		for (QueueRequest queueRequest : queueRequests)
		{
			std::optional<int> queueFamily = findQueueFamily(physicalDevice, queueRequest.flags, queueRequest.excludedFlags, queueRequest.supportsPresent);
		}

		return hasIndices && extensionsSupported && swapChainAdequate && subgroupsSupported;
	}

	bool DeviceBuilder::checkDeviceExtensionSupport(VkPhysicalDevice device)
	{
		uint32_t extensionsCount;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionsCount, nullptr);

		std::vector<VkExtensionProperties> availableExtensions(extensionsCount);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionsCount, availableExtensions.data());

		std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

		for (const auto& extension : availableExtensions)
		{
			requiredExtensions.erase(extension.extensionName);
		}

		return requiredExtensions.empty();
	}

	bool DeviceBuilder::checkDeviceSubgroupSupport(VkPhysicalDevice device)
	{
		VkPhysicalDeviceSubgroupProperties subgroupProperties{};
		subgroupProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
		subgroupProperties.pNext = nullptr;

		VkPhysicalDeviceProperties2 physicalDeviceProperties{};
		physicalDeviceProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
		physicalDeviceProperties.pNext = &subgroupProperties;

		// Check for subgroup support. Update support check per feature used (none yet)
		vkGetPhysicalDeviceProperties2(device, &physicalDeviceProperties);

		VkSubgroupFeatureFlags requiredOperationFlags{};
		requiredOperationFlags |= VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT;

		VkShaderStageFlags requiredStageFlags{};
		requiredStageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;

		// subgroupProperties.subgroupSize can be used for particle aosoa struct array size and can be passed to gpu for proper indexing
		return ((subgroupProperties.supportedOperations & requiredOperationFlags) == requiredOperationFlags) &&
			((subgroupProperties.supportedStages & requiredStageFlags) == requiredStageFlags);
	}

	DeviceBuilder::SwapChainSupportDetails DeviceBuilder::querySwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupportDetails details;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

		uint32_t formatCount;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

		if (formatCount != 0)
		{
			details.formats.resize(formatCount);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
		}

		uint32_t presentModeCount;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

		if (presentModeCount != 0)
		{
			details.presentModes.resize(presentModeCount);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
		}

		return details;
	}

	void DeviceBuilder::pickPhysicalDevice(VkPhysicalDevice& physicalDevice)
	{
		uint32_t deviceCount = 0;
		vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

		if (deviceCount == 0)
		{
			throw std::runtime_error("Failed to find GPU's with Vulkan support.");
		}

		std::vector<VkPhysicalDevice> devices(deviceCount);
		vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

		for (const auto& device : devices)
		{
			if (isDeviceSuitable(device))
			{
				physicalDevice = device;
				break;
			}
		}

		if (physicalDevice == VK_NULL_HANDLE)
		{
			throw std::runtime_error("Failed to find a suitable GPU.");
		}

		this->physicalDevice = physicalDevice;
	}

	void DeviceBuilder::getQueue(VkQueueFlags flags, VkQueueFlags excludedFlags, bool supportsPresent, VkQueue& queue)
	{
		std::optional<int> queueFamily = getQueueFamily(flags, excludedFlags, supportsPresent);

		vkGetDeviceQueue(device, queueFamily.value(), queueIndices[queueFamily.value()]++, &queue);
	}

	int DeviceBuilder::getQueueFamily(VkQueueFlags flags, VkQueueFlags excludedFlags, bool supportsPresent)
	{
		std::optional<int> queueFamily = findQueueFamily(physicalDevice, flags, excludedFlags, supportsPresent);

		if (!queueFamily.has_value())
		{
			throw std::runtime_error("Failed to find a queue family with the requested requirements.");
		}

		return queueFamily.value();
	}

	void DeviceBuilder::build(VkDevice& device, bool enableValidationLayers)
	{
		std::unordered_map<uint32_t, QueueFamilyRequest> queueFamilyRequests;

		for (auto& queueRequest : queueRequests)
		{
			std::optional<int> queueFamily = findQueueFamily(physicalDevice, queueRequest.flags, queueRequest.excludedFlags, queueRequest.supportsPresent);

			queueFamilyRequests[queueFamily.value()].queueCount++;
			queueFamilyRequests[queueFamily.value()].priorities.push_back(queueRequest.priority);
		}

		std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
		for (auto& queueFamilyRequest : queueFamilyRequests) {
			VkDeviceQueueCreateInfo queueCreateInfo{};
			queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queueCreateInfo.queueFamilyIndex = queueFamilyRequest.first;
			queueCreateInfo.queueCount = queueFamilyRequest.second.queueCount;
			queueCreateInfo.pQueuePriorities = queueFamilyRequest.second.priorities.data();
			queueCreateInfos.push_back(queueCreateInfo);
		}

		VkPhysicalDeviceFeatures deviceFeatures{};

		VkDeviceCreateInfo deviceInfo{};
		deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

		deviceInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
		deviceInfo.pQueueCreateInfos = queueCreateInfos.data();
		deviceInfo.pEnabledFeatures = &deviceFeatures;
		deviceInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
		deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();

		if (enableValidationLayers) {
			deviceInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			deviceInfo.ppEnabledLayerNames = validationLayers.data();
		}
		else {
			deviceInfo.enabledLayerCount = 0;
		}

		if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
			throw std::runtime_error("failed to create logical device!");
		}

		this->device = device;
	}

	std::optional<int> DeviceBuilder::findQueueFamily(VkPhysicalDevice physicalDevice, VkQueueFlags flags, VkQueueFlags excludedFlags, bool supportsPresent)
	{
		uint32_t queueFamilyCount = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);

		std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

		std::optional<int> queueFamilyIndex;

		int i = 0;
		for (const auto& queueFamily : queueFamilies)
		{
			VkBool32 presentSupport = true;
			if (supportsPresent)
			{
				if (surface == nullptr)
				{
					throw std::runtime_error("Failed to check for present support due to a missing surface.");
				}

				vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, i, surface, &presentSupport);
			}

			if ((queueFamily.queueFlags | flags) == queueFamily.queueFlags && (queueFamily.queueFlags & excludedFlags) == 0 && presentSupport)
			{
				queueFamilyIndex = i;
				return queueFamilyIndex;
			}

			i++;
		}

		return queueFamilyIndex;
	}
}