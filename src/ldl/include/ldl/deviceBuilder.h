#include <vulkan/vulkan.h>
#include <optional>
#include <vector>
#include <unordered_map>

namespace ldl
{
	struct DeviceBuilder
	{
		struct QueueRequest
		{
			VkQueueFlags flags;
			VkQueueFlags excludedFlags;
			float priority;
			bool supportsPresent;
		};

		struct QueueFamilyRequest
		{
			uint32_t queueCount;
			std::vector<float> priorities;
		};

		struct SwapChainSupportDetails
		{
			VkSurfaceCapabilitiesKHR capabilities;
			std::vector<VkSurfaceFormatKHR> formats;
			std::vector<VkPresentModeKHR> presentModes;

			bool isComplete()
			{
				return !presentModes.empty() && !formats.empty();
			}
		};

		VkInstance instance;
		VkPhysicalDevice physicalDevice;
		VkSurfaceKHR surface;
		VkDevice device;

		std::vector<QueueRequest> queueRequests;
		std::unordered_map<uint32_t, uint32_t> queueIndices;

		std::vector<const char*> deviceExtensions;
		std::vector<const char*> validationLayers;

		void addQueue(VkQueueFlags flags, VkQueueFlags excludedFlags, float priority, bool supportsPresent);

		bool isDeviceSuitable(VkPhysicalDevice device);

		bool checkDeviceExtensionSupport(VkPhysicalDevice device);

		bool checkDeviceSubgroupSupport(VkPhysicalDevice device);

		bool checkDeviceTimelineSupport(VkPhysicalDevice device);
		
		bool checkDeviceAtomicsSupport(VkPhysicalDevice device);

		SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);

		void pickPhysicalDevice(VkPhysicalDevice& physicalDevice);

		void getQueue(VkQueueFlags flags, VkQueueFlags excludedFlags, bool supportsPresent, VkQueue& queue);

		int getQueueFamily(VkQueueFlags flags, VkQueueFlags excludedFlags, bool supportsPresent);

		void build(VkDevice& device, bool enableValidationLayers);

		std::optional<int> findQueueFamily(VkPhysicalDevice physicalDevice, VkQueueFlags flags, VkQueueFlags excludedFlags, bool supportsPresent);
	};
}