#define GLFW_INCLUDE_VULKAN

#include <glfw/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/integer.hpp>
#include <vulkan/vulkan.h>
#include <vulkan/vk_enum_string_helper.h>

#include <dsl/descriptorAllocator.h>
#include <dsl/descriptorLayoutBuilder.h>
#include <ldl/deviceBuilder.h>
#include <iml/keyInput.h>
#include <iml/mouseInput.h>
#include <val/vulkanAllocator.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <fstream>
#include <array>
#include <random>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 800;
const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
const uint32_t PARTICLE_COUNT = 1 << 17;
const uint32_t BIN_KERNEL_SIZE = 1;
const uint32_t GRID_KERNEL_SIZE = 32;
const uint32_t BLOCK_KERNEL_SIZE = 16;
const uint32_t SUM_KERNEL_SIZE = 256;
const uint32_t BIN_SIZE = 64; // Change to specialization constant and set to the retrieved subgroupsize

const std::vector<const char*> validationLayers =
{
	"VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> instanceExtensions =
{

};

const std::vector<VkValidationFeatureEnableEXT> validationExtensions =
{
	VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT,
	VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT
};

const std::vector<const char*> deviceExtensions =
{
	VK_KHR_SWAPCHAIN_EXTENSION_NAME,
	VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

struct alignas(8) ParameterUBO
{
	glm::vec2 speed;
	float k;
	float mu;
	float rho;
	float dx;
	float invDx;
	uint32_t quadDimensions;
	uint32_t nodeDimensions;
	uint32_t blockDimensions;
	float dt;
	float invDt;
};

struct alignas(8) CameraUBO
{
	glm::vec2 pos;
	float aspectRatio;
	float zoom;
};

struct ScatterDispatchData
{
	uint32_t dispatchX;
	uint32_t dispatchY;
	uint32_t dispatchZ;
};

// WATCH FOR ALIGNMENT ISSUES. 
// Structs on the gpu need to be a multiple of an alignment value. 
// In std140 this is automatically a multiple of 16.
// In std430 this is the largest alignment in the struct.
// Individual variables (such as float, vec2, vec3, vec4) need to be on an adress divisible by their alignment. They are automatically padded if they dont line up.
// Scalar types (float, int, etc): 4.
// vec2: 8.
// vec3: 16!!
// vec4: 16.
// mat2: 8 per collumn.
// mat3: 16 per collumn (3 vec3s padded to vec4).
// mat4: 16 per collumn.

struct alignas(8) Bin
{
	glm::mat2 F[BIN_SIZE]; // (8 + 8) * 64 = 1024
	glm::vec2 position[BIN_SIZE]; // 8 * 32 = 256
	float mass[BIN_SIZE]; // 4 * 32 = 128
	uint32_t blockParticleIndex[BIN_SIZE]; // 4 * 32 = 128
	uint32_t particleId[BIN_SIZE];
	uint32_t particleCount; // 4
	// Max alignment = 8, so pad till nearest multiple of 8
	// Total size = 512 + 256 + 128 + 128 + 4 + 4(pad) = 1032 = 8 * 129
};

struct alignas(16) Particle
{
	glm::vec4 color; // 16
	glm::vec2 position; // 8
	// Max alignment = 16, so pad till nearest multiple of 16
	// Total size 16 + 8 + 8(pad) = 32

	static VkVertexInputBindingDescription getBindingDescription()
	{
		VkVertexInputBindingDescription bindingDescription{};
		bindingDescription.binding = 0;
		bindingDescription.stride = sizeof(Particle);
		bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bindingDescription;
	}

	static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 2> attributeDescriptions{};

		attributeDescriptions[0].binding = 0;
		attributeDescriptions[0].location = 0;
		attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
		attributeDescriptions[0].offset = offsetof(Particle, position);

		attributeDescriptions[1].binding = 0;
		attributeDescriptions[1].location = 1;
		attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attributeDescriptions[1].offset = offsetof(Particle, color);

		return attributeDescriptions;
	}
};
static_assert(sizeof(Particle) == 32);

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
	const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger)
{
	auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger,
	const VkAllocationCallbacks* pAllocator)
{
	auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
	if (func != nullptr)
	{
		func(instance, debugMessenger, pAllocator);
	}
}

class MpmVulkan
{
public:
	void run()
	{
		initWindow();
		initVulkan();
		mainLoop();
		cleanup();
	}

private:
	std::vector<int> keys{
		GLFW_KEY_W,
		GLFW_KEY_A,
		GLFW_KEY_S,
		GLFW_KEY_D
	};

	std::vector<int> buttons{
		GLFW_MOUSE_BUTTON_1,
		GLFW_MOUSE_BUTTON_2,
		GLFW_MOUSE_BUTTON_3,
		GLFW_MOUSE_BUTTON_4,
		GLFW_MOUSE_BUTTON_5,
		GLFW_MOUSE_BUTTON_6,
		GLFW_MOUSE_BUTTON_7,
		GLFW_MOUSE_BUTTON_8
	};

	GLFWwindow* window;
	VkInstance instance;
	VkDebugUtilsMessengerEXT debugMessenger;
	VkPhysicalDevice physicalDevice;
	VkDevice device;

	iml::KeyInput keyInput{ keys };
	iml::MouseInput mouseInput{ buttons };

	VkQueue computeQueue;
	uint32_t computeFamilyIndex;

	VkQueue graphicsQueue;
	uint32_t graphicsFamilyIndex;

	VkQueue presentQueue;
	uint32_t presentFamilyIndex;

	VkSurfaceKHR surface;

	dsl::DescriptorAllocator substepComputeDescriptorAllocator;
	dsl::DescriptorAllocator transferComputeDescriptorAllocator;
	dsl::DescriptorAllocator graphicsDescriptorAllocator;
	ldl::DeviceBuilder deviceBuilder;

	VkRenderPass renderPass;

	std::vector<VkDescriptorSet> graphicsDescriptorSets;
	VkDescriptorSetLayout graphicsDescriptorSetLayout;
	VkPipelineLayout graphicsPipelineLayout;

	VkPipeline graphicsPipeline;
	
	std::vector<VkDescriptorSet> substepDescriptorSets;
	std::vector<VkDescriptorSet> transferDescriptorSets;
	VkDescriptorSetLayout substepComputeDescriptorSetLayout;
	VkDescriptorSetLayout transferComputeDescriptorSetLayout;
	VkPipelineLayout computePipelineLayout;

	VkPipeline g2p2gComputePipeline;
	VkPipeline processhistogramComputePipeline;
	VkPipeline localsumComputePipeline;
	VkPipeline partialglobalsumComputePipeline;
	VkPipeline globalsumComputePipeline;
	VkPipeline scatterComputePipeline;
	VkPipeline totalextractionComputePipeline;
	VkPipeline clearhistogramComputePipeline;
	VkPipeline graphicsscatterComputePipeline;
	VkPipeline gridComputePipeline;

	VkSwapchainKHR swapChain;
	std::vector<VkImage> swapChainImages;
	VkFormat swapChainImageFormat;
	VkExtent2D swapChainExtent;
	std::vector<VkImageView> swapChainImageViews;
	std::vector<VkFramebuffer> swapChainFramebuffers;

	VkCommandPool graphicsCommandPool;
	VkCommandPool transientGraphicsCommandPool;

	VkCommandPool computeCommandPool;
	VkCommandPool transientComputeCommandPool;

	std::vector<VkCommandBuffer> commandBuffers;
	std::vector<VkCommandBuffer> computeCommandBuffers;

	std::vector<VkSemaphore> imageAquireSemaphores;
	std::vector<VkSemaphore> imageSubmitSemaphores;
	std::vector<VkFence> inFlightFences;

	std::vector<VkSemaphore> computeFinishedSemaphores;
	std::vector<VkFence> computeInFlightFences;

	bool framebufferResized = false;

	uint32_t currentFrame = 0;
	uint32_t currentCompute = 0;

	val::VulkanAllocator bufferAllocator;

	std::vector<val::AllocatedBuffer> binBuffers; // Scatter requires a seperate read and write buffer to prevent race conditions
	std::vector<val::AllocatedBuffer> graphicsBuffers; // Requires stored states per in flight frame

	std::vector<val::AllocatedBuffer> vQuadBuffers; // G2p2g requires a reperate read and write buffer to prevent race conditions
	std::vector<val::AllocatedBuffer> GQuadBuffers;
	val::AllocatedBuffer mQuadBuffer;
	val::AllocatedBuffer vNodeBuffer;

	val::AllocatedBuffer histogramBuffer;
	val::AllocatedBuffer binCountBuffer;
	val::AllocatedBuffer binOffsetsBuffer;
	val::AllocatedBuffer binSumBuffer;

	val::AllocatedBuffer scatterIndirectDispatchBuffer;

	std::vector<val::AllocatedBuffer> parameterBuffers;
	std::vector<val::AllocatedBuffer> cameraBuffers;

	uint32_t quadDimensions = 1 << 7;
	uint32_t quadCount = quadDimensions * quadDimensions;
	uint32_t quadBlockDimensions = quadDimensions >> 2;
	uint32_t quadBlockCount = quadBlockDimensions * quadBlockDimensions;
	uint32_t particleBlockDimensions = quadBlockDimensions - 1;
	uint32_t particleBlockCount = particleBlockDimensions * particleBlockDimensions;
	uint32_t paddedParticleBlockDimensions = particleBlockDimensions + 1;
	uint32_t paddedParticleBlockCount = paddedParticleBlockDimensions * paddedParticleBlockDimensions;
	uint32_t nodeDimensions = quadDimensions + 1;
	uint32_t nodeCount = nodeDimensions * nodeDimensions;

	//uint32_t gridBlockDimensions = quadDimensions >> 2;
	//uint32_t particleBlockDimensions = gridBlockDimensions - 1;
	//uint32_t paddedParticleBlockDimensions = gridBlockDimensions;
	//uint32_t paddedParticleBlockCount = gridBlockDimensions * gridBlockDimensions;

	uint32_t binCount = particleBlockDimensions * particleBlockDimensions + ceilIntDivision(PARTICLE_COUNT, BIN_SIZE); // Impossibly worst case scenario. Every bin is full and all blocks have one non-full bin

	double lastTime = 0;
	double lastFrameTime = 0;

	glm::vec2 lastMousePos = glm::vec2(0);
	float sensitivity = 0.002f;
	float scrollSensitivity = 0.05f;

	float E = 100000;
	float v = 0.40f;
	float rho = 2000;
	float dx = 1.0f / quadDimensions;
	float dt = 0.0001f;
	float size = 0.2f;
	uint32_t substeps = 15;
	glm::vec2 cameraPos = glm::vec2(0);
	float zoom = 1;
	glm::vec2 accel = glm::vec2(0);

	void mainLoop()
	{
		while (!glfwWindowShouldClose(window))
		{
			iml::MouseInput::clearFrameValues();
			glfwPollEvents();
			getInput();
			drawFrame();

			double currentTime = glfwGetTime();
			lastFrameTime = currentTime - lastTime;
			lastTime = currentTime;
		}

		vkDeviceWaitIdle(device);
	}

	void initWindow()
	{
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

		window = glfwCreateWindow(WIDTH, HEIGHT, "MpmVulkan", nullptr, nullptr);
		iml::KeyInput::setupKeyInputs(window);
		iml::MouseInput::setupMouseInputs(window);

		glfwSetWindowUserPointer(window, this);
		glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
		lastTime = glfwGetTime();

		double posX;
		double posY;
		glfwGetCursorPos(window, &posX, &posY);
		lastMousePos = glm::vec2(posX, posY);
	}

	void initVulkan()
	{
		createInstance();
		setupDebugMessenger();
		createSurface();
		createDevices();
		createSwapChain();
		createSwapchainImageViews();
		createRenderPass();
		createUniformBuffers();
		createGraphicsDescriptors();
		createGraphicsPipeline();
		createFrameBuffers();
		createCommandPools();
		createShaderStorageBuffers();
		createComputeDescriptors();
		createComputePipelines();
		createCommandBuffers();
		createComputeCommandBuffers();
		createSyncObjects();
	}

	void createInstance()
	{
		if (enableValidationLayers && !checkValidationLayerSupport())
		{
			throw std::runtime_error("Validation layers requested, but not available.");
		}

		VkApplicationInfo appInfo{};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "MpmVulkan";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.pEngineName = "No Engine";
		appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
		appInfo.apiVersion = VK_API_VERSION_1_1;

		VkValidationFeaturesEXT features{};
		features.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT;
		features.enabledValidationFeatureCount = static_cast<uint32_t>(validationExtensions.size());
		features.pEnabledValidationFeatures = validationExtensions.data();

		VkInstanceCreateInfo createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		createInfo.pApplicationInfo = &appInfo;
		createInfo.pNext = &features;

		auto glfwExtensions = getRequiredExtensions();
		for (auto instanceExtension : instanceExtensions)
		{
			glfwExtensions.push_back(instanceExtension);
		}

		uint32_t glfwExtensionCount = static_cast<uint32_t>(glfwExtensions.size());
		createInfo.enabledExtensionCount = glfwExtensionCount;
		createInfo.ppEnabledExtensionNames = glfwExtensions.data();

		VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
		if (enableValidationLayers)
		{
			createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
			createInfo.ppEnabledLayerNames = validationLayers.data();

			populateDebugMessengerCreateInfo(debugCreateInfo);
			createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
		}
		else
		{
			createInfo.enabledLayerCount = 0;

			createInfo.pNext = nullptr;
		}

		if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
		{
			throw std::runtime_error("failed to create instance.");
		}

		uint32_t extensionCount = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

		std::vector<VkExtensionProperties> extensions(extensionCount);

		vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

		std::cout << "Required extensions:\n";

		for (uint32_t i = 0; i < glfwExtensionCount; i++)
		{
			const char* requiredExtension = glfwExtensions[i];

			bool containsExtension = false;
			for (const auto& extension : extensions)
			{
				if (strcmp(extension.extensionName, requiredExtension) == 0)
				{
					containsExtension = true;
					break;
				}
			}

			std::cout << (containsExtension ? "Available:" : "Unavailable:") << ' ' << requiredExtension << '\n';
		}
	}

	void getInput()
	{
		int y = keyInput.getIsKeyDown(GLFW_KEY_W) - keyInput.getIsKeyDown(GLFW_KEY_S);
		int x = keyInput.getIsKeyDown(GLFW_KEY_D) - keyInput.getIsKeyDown(GLFW_KEY_A);
		
		glm::vec2 mousePos = glm::vec2((float)mouseInput.posX, (float)mouseInput.posY);
		glm::vec2 deltaMouse = mouseInput.getIsButtonDown(GLFW_MOUSE_BUTTON_LEFT) ? lastMousePos - mousePos : glm::vec2(0);
		lastMousePos = mousePos;
		zoom += (mouseInput.scrollDY * scrollSensitivity) * zoom;
		cameraPos += deltaMouse * sensitivity / zoom;
		accel = glm::vec2(x, y) * 20.0f;
	}

	void drawFrame()
	{
		vkWaitForFences(device, 1, &computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		updateUniformBuffer(currentFrame);

		vkResetFences(device, 1, &computeInFlightFences[currentFrame]);

		vkResetCommandBuffer(computeCommandBuffers[currentFrame], 0);
		recordComputeCommandBuffer(computeCommandBuffers[currentFrame]);

		VkSubmitInfo computeSubmitInfo = {};
		computeSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		computeSubmitInfo.commandBufferCount = 1;
		computeSubmitInfo.pCommandBuffers = &computeCommandBuffers[currentFrame];
		computeSubmitInfo.signalSemaphoreCount = 1;
		computeSubmitInfo.pSignalSemaphores = &computeFinishedSemaphores[currentFrame];

		if (vkQueueSubmit(computeQueue, 1, &computeSubmitInfo, computeInFlightFences[currentFrame]) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to submit compute command buffer.");
		}

		vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

		uint32_t imageIndex;
		VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAquireSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			recreateSwapchain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("Failed to aquire swap chain image.");
		}

		vkResetFences(device, 1, &inFlightFences[currentFrame]);

		vkResetCommandBuffer(commandBuffers[currentFrame], 0);
		recordCommandBuffer(commandBuffers[currentFrame], imageIndex);

		// What semaphores to wait on and for what stages
		VkSemaphore waitSemaphores[] = { computeFinishedSemaphores[currentFrame], imageAquireSemaphores[currentFrame] };
		VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

		VkSubmitInfo graphicsSubmitInfo{};
		graphicsSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		graphicsSubmitInfo.commandBufferCount = 1;
		graphicsSubmitInfo.pCommandBuffers = &commandBuffers[currentFrame];
		graphicsSubmitInfo.waitSemaphoreCount = 2;
		graphicsSubmitInfo.pWaitSemaphores = waitSemaphores;
		graphicsSubmitInfo.pWaitDstStageMask = waitStages;
		graphicsSubmitInfo.signalSemaphoreCount = 1;
		graphicsSubmitInfo.pSignalSemaphores = &imageSubmitSemaphores[imageIndex];

		// VKRESULTDEVICELOST (undefined behavior on the gpu) Check out the substep stuff further
		if (vkQueueSubmit(graphicsQueue, 1, &graphicsSubmitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to submit draw command buffer.");
		}

		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &imageSubmitSemaphores[imageIndex];
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &swapChain;
		presentInfo.pImageIndices = &imageIndex;

		result = vkQueuePresentKHR(presentQueue, &presentInfo);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized)
		{
			framebufferResized = false;
			recreateSwapchain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to present swap chain image.");
		}

		currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
	}

	void updateUniformBuffer(uint32_t currentImage)
	{
		ParameterUBO pUBO{};
		pUBO.k = E / (2.0f * (1.0f - v));
		pUBO.mu = E / (2.0f * (1.0f + v));
		pUBO.rho = rho;
		pUBO.dx = dx;
		pUBO.invDx = 1.0f / dx;
		pUBO.quadDimensions = quadDimensions;
		pUBO.nodeDimensions = nodeDimensions;
		pUBO.blockDimensions = particleBlockDimensions;
		pUBO.dt = dt;
		pUBO.invDt = 1.0f / pUBO.dt;
		pUBO.speed = accel;

		memcpy(parameterBuffers[currentImage].mapped, &pUBO, sizeof(pUBO));

		int width;
		int height;
		glfwGetFramebufferSize(window, &width, &height);

		CameraUBO cUBO{};
		cUBO.aspectRatio = (float)height / (float)width;
		cUBO.pos = cameraPos;
		cUBO.zoom = zoom;

		memcpy(cameraBuffers[currentImage].mapped, &cUBO, sizeof(cUBO));
	}

	std::vector<const char*> getRequiredExtensions()
	{
		uint32_t glfwExtensionCount = 0;
		const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

		std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

		if (enableValidationLayers)
		{
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
		}

		return extensions;
	}

	bool checkValidationLayerSupport()
	{
		uint32_t layerCount;
		vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

		std::vector<VkLayerProperties> availableLayers(layerCount);
		vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

		for (const char* layerName : validationLayers)
		{
			bool layerFound = false;

			for (const auto& layerProperties : availableLayers)
			{
				if (strcmp(layerName, layerProperties.layerName) == 0)
				{
					layerFound = true;
					break;
				}
			}

			if (!layerFound)
			{
				return false;
			}
		}

		return true;
	}

	void setupDebugMessenger()
	{
		if (!enableValidationLayers)
			return;

		VkDebugUtilsMessengerCreateInfoEXT createInfo;
		populateDebugMessengerCreateInfo(createInfo);

		if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to set up debug messenger.");
		}
	}

	void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
	{
		createInfo = {};
		createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
		createInfo.messageSeverity =
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
		createInfo.messageType =
			VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
			VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
		createInfo.pfnUserCallback = debugCallback;
	}

	void createSurface()
	{
		if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create window surface.");
		}
	}

	void createDevices()
	{
		deviceBuilder.surface = surface;
		deviceBuilder.instance = instance;
		deviceBuilder.deviceExtensions = deviceExtensions;
		deviceBuilder.validationLayers = validationLayers;

		deviceBuilder.addQueue(VK_QUEUE_GRAPHICS_BIT, 0, 1.0f, false);
		deviceBuilder.addQueue(VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT, 1.0f, false);
		deviceBuilder.addQueue(0, 0, 1.0f, true);

		deviceBuilder.pickPhysicalDevice(physicalDevice);

		deviceBuilder.build(device, enableValidationLayers);

		bufferAllocator = { device, physicalDevice };

		deviceBuilder.getQueue(VK_QUEUE_GRAPHICS_BIT, 0, false, graphicsQueue);
		graphicsFamilyIndex = deviceBuilder.getQueueFamily(VK_QUEUE_GRAPHICS_BIT, 0, false);

		deviceBuilder.getQueue(VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT, false, computeQueue);
		computeFamilyIndex = deviceBuilder.getQueueFamily(VK_QUEUE_COMPUTE_BIT, VK_QUEUE_GRAPHICS_BIT, false);

		deviceBuilder.getQueue(0, 0, true, presentQueue);
		presentFamilyIndex = deviceBuilder.getQueueFamily(0, 0, false);
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback
	(
		VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
		VkDebugUtilsMessageTypeFlagsEXT messageType,
		const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
		void* pUserData
	)
	{
		std::cerr << "Validation layer: " << pCallbackData->pMessage << std::endl;

		return VK_FALSE;
	}

	VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats)
	{
		for (const auto& availableFormat : availableFormats)
		{
			if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
			{
				return availableFormat;
			}
		}

		return availableFormats[0];
	}

	VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes)
	{
		for (const auto& availablePresentMode : availablePresentModes)
		{
			if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				return availablePresentMode;
			}
		}

		return VK_PRESENT_MODE_FIFO_KHR;
	}

	VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities)
	{
		if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
		{
			return capabilities.currentExtent;
		}
		else
		{
			int width, height;
			glfwGetFramebufferSize(window, &width, &height);

			VkExtent2D actualExtent = {
				static_cast<uint32_t>(width),
				static_cast<uint32_t>(height)
			};

			actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
			actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

			return actualExtent;
		}
	}

	void createSwapChain()
	{
		ldl::DeviceBuilder::SwapChainSupportDetails swapChainSupport = deviceBuilder.querySwapChainSupport(physicalDevice);

		VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
		VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
		VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

		uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

		if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
		{
			imageCount = swapChainSupport.capabilities.maxImageCount;
		}

		VkSwapchainCreateInfoKHR createInfo{};
		createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		createInfo.surface = surface;

		createInfo.minImageCount = imageCount;
		createInfo.imageFormat = surfaceFormat.format;
		createInfo.imageColorSpace = surfaceFormat.colorSpace;
		createInfo.imageExtent = extent;
		createInfo.imageArrayLayers = 1;
		createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		if (graphicsFamilyIndex != presentFamilyIndex)
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			createInfo.queueFamilyIndexCount = 2;
			createInfo.pQueueFamilyIndices = std::vector<uint32_t>{ graphicsFamilyIndex, presentFamilyIndex }.data();
		}
		else
		{
			createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
			createInfo.queueFamilyIndexCount = 0;
			createInfo.pQueueFamilyIndices = nullptr;
		}

		createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
		createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

		createInfo.presentMode = presentMode;
		createInfo.clipped = VK_TRUE;

		createInfo.oldSwapchain = VK_NULL_HANDLE;

		if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create swap chain.");
		}

		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
		swapChainImages.resize(imageCount);
		vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());

		swapChainImageFormat = surfaceFormat.format;
		swapChainExtent = extent;
	}

	void createSwapchainImageViews()
	{
		swapChainImageViews.resize(swapChainImages.size());

		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			createImageView(swapChainImages[i], swapChainImageViews[i], swapChainImageFormat);
		}
	}

	void createRenderPass()
	{
		VkAttachmentDescription colorAttachment{};
		colorAttachment.format = swapChainImageFormat;
		colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;

		colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

		colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

		colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		VkAttachmentReference colorAttachmentRef{};
		colorAttachmentRef.attachment = 0;
		colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass{};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &colorAttachmentRef;

		VkRenderPassCreateInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		renderPassInfo.attachmentCount = 1;
		renderPassInfo.pAttachments = &colorAttachment;
		renderPassInfo.subpassCount = 1;
		renderPassInfo.pSubpasses = &subpass;

		VkSubpassDependency dependency{};
		dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		dependency.dstSubpass = 0;
		dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependency.srcAccessMask = 0;
		dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;;
		dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		renderPassInfo.dependencyCount = 1;
		renderPassInfo.pDependencies = &dependency;

		if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create render pass.");
		}
	}

	void createGraphicsDescriptors()
	{
		std::vector<dsl::DescriptorAllocator::PoolSizeRatio> sizes =
		{
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1.0f}
		};

		graphicsDescriptorAllocator.initPool(device, MAX_FRAMES_IN_FLIGHT * 2, sizes);

		dsl::DescriptorLayoutBuilder builder{};
		builder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
		builder.addBinding(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);

		builder.build(device, VK_SHADER_STAGE_VERTEX_BIT, graphicsDescriptorSetLayout);

		std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, graphicsDescriptorSetLayout);

		graphicsDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		graphicsDescriptorAllocator.allocate(device, layouts.data(), (uint32_t)MAX_FRAMES_IN_FLIGHT, graphicsDescriptorSets.data());

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			std::vector<VkWriteDescriptorSet> descriptorWrites(2);

			VkDescriptorBufferInfo parameterBufferInfo{};
			parameterBufferInfo.buffer = parameterBuffers[i].buffer;
			parameterBufferInfo.offset = 0;
			parameterBufferInfo.range = sizeof(ParameterUBO);

			descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[0].dstSet = graphicsDescriptorSets[i];
			descriptorWrites[0].dstBinding = 0;
			descriptorWrites[0].dstArrayElement = 0;
			descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[0].descriptorCount = 1;
			descriptorWrites[0].pBufferInfo = &parameterBufferInfo;

			VkDescriptorBufferInfo cameraBufferInfo{};
			cameraBufferInfo.buffer = cameraBuffers[i].buffer;
			cameraBufferInfo.offset = 0;
			cameraBufferInfo.range = sizeof(CameraUBO);

			descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[1].dstSet = graphicsDescriptorSets[i];
			descriptorWrites[1].dstBinding = 1;
			descriptorWrites[1].dstArrayElement = 0;
			descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[1].descriptorCount = 1;
			descriptorWrites[1].pBufferInfo = &cameraBufferInfo;

			vkUpdateDescriptorSets(device, (uint32_t)descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
		}
	}

	void createGraphicsPipeline()
	{
		auto vertShaderCode = readFile("../shaders/shader.vert.spv");
		auto fragShaderCode = readFile("../shaders/shader.frag.spv");

		VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
		VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

		VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
		vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertShaderStageInfo.module = vertShaderModule;
		vertShaderStageInfo.pName = "main";
		vertShaderStageInfo.pSpecializationInfo = nullptr;

		VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
		fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragShaderStageInfo.module = fragShaderModule;
		fragShaderStageInfo.pName = "main";
		fragShaderStageInfo.pSpecializationInfo = nullptr;

		VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

		std::vector<VkDynamicState> dynamicStates =
		{
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};

		VkPipelineDynamicStateCreateInfo dynamicState{};
		dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
		dynamicState.pDynamicStates = dynamicStates.data();

		VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
		vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

		auto bindingDescription = Particle::getBindingDescription();
		auto attributeDescriptions = Particle::getAttributeDescriptions();

		vertexInputInfo.vertexBindingDescriptionCount = 1;
		vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
		vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
		vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

		VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
		inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
		inputAssembly.primitiveRestartEnable = VK_FALSE;

		VkViewport viewPort{};
		viewPort.x = 0.0f;
		viewPort.y = 0.0f;
		viewPort.width = (float)swapChainExtent.width;
		viewPort.height = (float)swapChainExtent.height;
		viewPort.minDepth = 0.0f;
		viewPort.maxDepth = 1.0f;

		VkPipelineViewportStateCreateInfo viewportState{};
		viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportState.viewportCount = 1;
		viewportState.scissorCount = 1;

		VkPipelineRasterizationStateCreateInfo rasterizer{};
		rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer.depthClampEnable = VK_FALSE;
		rasterizer.rasterizerDiscardEnable = VK_FALSE;
		rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer.lineWidth = 1.0f;

		rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;

		rasterizer.depthBiasEnable = VK_FALSE;
		rasterizer.depthBiasConstantFactor = 0.0f;
		rasterizer.depthBiasClamp = 0.0f;
		rasterizer.depthBiasSlopeFactor = 0.0f;

		VkPipelineMultisampleStateCreateInfo multisampling{};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisampling.minSampleShading = 1.0f;
		multisampling.pSampleMask = nullptr;
		multisampling.alphaToCoverageEnable = VK_FALSE;
		multisampling.alphaToOneEnable = VK_FALSE;

		VkPipelineColorBlendAttachmentState colorBlendAttachment{};
		colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
		colorBlendAttachment.blendEnable = VK_TRUE;
		colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
		colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
		colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

		VkPipelineColorBlendStateCreateInfo colorBlending{};
		colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		colorBlending.logicOpEnable = VK_FALSE;
		colorBlending.logicOp = VK_LOGIC_OP_COPY;
		colorBlending.attachmentCount = 1;
		colorBlending.pAttachments = &colorBlendAttachment;
		colorBlending.blendConstants[0] = 0.0f;
		colorBlending.blendConstants[1] = 0.0f;
		colorBlending.blendConstants[2] = 0.0f;
		colorBlending.blendConstants[3] = 0.0f;

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = 1;
		pipelineLayoutInfo.pSetLayouts = &graphicsDescriptorSetLayout;

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &graphicsPipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create graphics pipeline layout.");
		}

		VkGraphicsPipelineCreateInfo pipelineInfo{};
		pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipelineInfo.stageCount = 2;
		pipelineInfo.pStages = shaderStages;

		pipelineInfo.pVertexInputState = &vertexInputInfo;
		pipelineInfo.pInputAssemblyState = &inputAssembly;
		pipelineInfo.pViewportState = &viewportState;
		pipelineInfo.pRasterizationState = &rasterizer;
		pipelineInfo.pMultisampleState = &multisampling;
		pipelineInfo.pColorBlendState = &colorBlending;
		pipelineInfo.pDepthStencilState = nullptr;
		pipelineInfo.pDynamicState = &dynamicState;

		pipelineInfo.layout = graphicsPipelineLayout;
		pipelineInfo.renderPass = renderPass;
		pipelineInfo.subpass = 0;

		pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
		pipelineInfo.basePipelineIndex = -1;

		if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create graphics pipeline.");
		}

		vkDestroyShaderModule(device, fragShaderModule, nullptr);
		vkDestroyShaderModule(device, vertShaderModule, nullptr);
	}

	void createComputeDescriptors()
	{
		std::vector<dsl::DescriptorAllocator::PoolSizeRatio> substepSizes =
		{
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1.0f }
		};

		substepComputeDescriptorAllocator.initPool(device, 2 * 13, substepSizes);

		dsl::DescriptorLayoutBuilder substepBuilder{};
		substepBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // VQR
		substepBuilder.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // VQW
		substepBuilder.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // M
		substepBuilder.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // H
		substepBuilder.addBinding(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // BC
		substepBuilder.addBinding(5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // BO
		substepBuilder.addBinding(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // BS
		substepBuilder.addBinding(7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // BR
		substepBuilder.addBinding(8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // BW
		substepBuilder.addBinding(9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // SID
		substepBuilder.addBinding(10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // GQR
		substepBuilder.addBinding(11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // GQW
		substepBuilder.addBinding(12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // VN

		substepBuilder.build(device, VK_SHADER_STAGE_COMPUTE_BIT, substepComputeDescriptorSetLayout);

		std::vector<VkDescriptorSetLayout> substepLayouts(2, substepComputeDescriptorSetLayout);

		substepDescriptorSets.resize(2);
		substepComputeDescriptorAllocator.allocate(device, substepLayouts.data(), 2, substepDescriptorSets.data());

		for (size_t i = 0; i < 2; i++)
		{
			uint32_t binding = 0;
			std::array<VkWriteDescriptorSet, 13> descriptorWrites{};
			std::array<VkDescriptorBufferInfo, 13> descriptorInfos{};

			binding = 0;
			descriptorInfos[binding].buffer = vQuadBuffers[(i + 2 - 1) % 2].buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(glm::vec2) * quadDimensions * quadDimensions;
;
			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];
			
			binding = 1;
			descriptorInfos[binding].buffer = vQuadBuffers[i].buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(glm::vec2) * quadDimensions * quadDimensions;

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			binding = 2;
			descriptorInfos[binding].buffer = mQuadBuffer.buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(float) * quadDimensions * quadDimensions;

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			binding = 3;
			descriptorInfos[binding].buffer = histogramBuffer.buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(uint32_t) * paddedParticleBlockCount;

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			binding = 4;
			descriptorInfos[binding].buffer = binCountBuffer.buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(uint32_t) * paddedParticleBlockCount;

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			binding = 5;
			descriptorInfos[binding].buffer = binOffsetsBuffer.buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(uint32_t) * paddedParticleBlockCount;

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			binding = 6;
			descriptorInfos[binding].buffer = binSumBuffer.buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(uint32_t) * BLOCK_KERNEL_SIZE * BLOCK_KERNEL_SIZE;

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			binding = 7;
			descriptorInfos[binding].buffer = binBuffers[(i + 2 - 1) % 2].buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(Bin) * binCount;

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			binding = 8;
			descriptorInfos[binding].buffer = binBuffers[i].buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(Bin) * binCount;

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			binding = 9;
			descriptorInfos[binding].buffer = scatterIndirectDispatchBuffer.buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(ScatterDispatchData);

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			binding = 10;
			descriptorInfos[binding].buffer = GQuadBuffers[(i + 2 - 1) % 2].buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(glm::mat2) * quadCount;

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			binding = 11;
			descriptorInfos[binding].buffer = GQuadBuffers[i].buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(glm::mat2) * quadCount;

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			binding = 12;
			descriptorInfos[binding].buffer = vNodeBuffer.buffer;
			descriptorInfos[binding].offset = 0;
			descriptorInfos[binding].range = sizeof(float) * nodeCount;

			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = substepDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &descriptorInfos[binding];

			vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
		}

		std::vector<dsl::DescriptorAllocator::PoolSizeRatio> transferSizes =
		{
			{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1.0f / 2.0f },
			{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1.0f / 2.0f }
		};
		
		transferComputeDescriptorAllocator.initPool(device, 2 * MAX_FRAMES_IN_FLIGHT, transferSizes);

		dsl::DescriptorLayoutBuilder transferBuilder{};
		transferBuilder.addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER); // UBO
		transferBuilder.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER); // G

		transferBuilder.build(device, VK_SHADER_STAGE_COMPUTE_BIT, transferComputeDescriptorSetLayout);

		std::vector<VkDescriptorSetLayout> transferLayouts(MAX_FRAMES_IN_FLIGHT, transferComputeDescriptorSetLayout);

		transferDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
		transferComputeDescriptorAllocator.allocate(device, transferLayouts.data(), (uint32_t)MAX_FRAMES_IN_FLIGHT, transferDescriptorSets.data());

		for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			uint32_t binding = 0;
			std::array<VkWriteDescriptorSet, 2> descriptorWrites{};

			VkDescriptorBufferInfo uniformBufferInfo{};
			uniformBufferInfo.buffer = parameterBuffers[i].buffer;
			uniformBufferInfo.offset = 0;
			uniformBufferInfo.range = sizeof(ParameterUBO);

			binding = 0;
			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = transferDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &uniformBufferInfo;

			VkDescriptorBufferInfo particleBufferInfo{};
			particleBufferInfo.buffer = graphicsBuffers[i].buffer;
			particleBufferInfo.offset = 0;
			particleBufferInfo.range = sizeof(Particle) * PARTICLE_COUNT;

			binding = 1;
			descriptorWrites[binding].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			descriptorWrites[binding].dstSet = transferDescriptorSets[i];
			descriptorWrites[binding].dstBinding = binding;
			descriptorWrites[binding].dstArrayElement = 0;
			descriptorWrites[binding].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			descriptorWrites[binding].descriptorCount = 1;
			descriptorWrites[binding].pBufferInfo = &particleBufferInfo;

			vkUpdateDescriptorSets(device, descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
		}
	}

	void createComputePipelines()
	{
		auto g2p2gCode = readFile("../shaders/g2p2g.comp.spv");
		auto processhistogramCode = readFile("../shaders/processhistogram.comp.spv");
		auto localsumCode = readFile("../shaders/localsum.comp.spv");
		auto partialglobalsumCode = readFile("../shaders/partialglobalsum.comp.spv");
		auto globalsumCode = readFile("../shaders/globalsum.comp.spv");
		auto scatterCode = readFile("../shaders/scatter.comp.spv");
		auto totalextractionCode = readFile("../shaders/totalextraction.comp.spv");
		auto clearhistogramCode = readFile("../shaders/clearhistogram.comp.spv");
		auto graphicsscatterCode = readFile("../shaders/graphicsscatter.comp.spv");
		auto gridCode = readFile("../shaders/grid.comp.spv");

		std::vector<VkShaderModule> shaderModules =
		{
			createShaderModule(g2p2gCode),
			createShaderModule(processhistogramCode),
			createShaderModule(localsumCode),
			createShaderModule(partialglobalsumCode),
			createShaderModule(globalsumCode),
			createShaderModule(scatterCode),
			createShaderModule(totalextractionCode),
			createShaderModule(clearhistogramCode),
			createShaderModule(graphicsscatterCode),
			createShaderModule(gridCode)
		};

		std::vector<VkPipelineShaderStageCreateInfo> stageInfos{};
		for (VkShaderModule& module : shaderModules)
		{
			VkPipelineShaderStageCreateInfo computeStageInfo{};
			computeStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
			computeStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
			computeStageInfo.module = module;
			computeStageInfo.pName = "main";

			stageInfos.push_back(computeStageInfo);
		}

		std::vector<VkDescriptorSetLayout> setLayouts{
			substepComputeDescriptorSetLayout,
			transferComputeDescriptorSetLayout
		};

		VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
		pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipelineLayoutInfo.setLayoutCount = (uint32_t)setLayouts.size();
		pipelineLayoutInfo.pSetLayouts = setLayouts.data();
		pipelineLayoutInfo.pushConstantRangeCount = 0;
		pipelineLayoutInfo.pPushConstantRanges = nullptr; // Remember if i ever want to use push constants

		if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &computePipelineLayout) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create compute pipeline layout.");
		}

		std::vector<VkComputePipelineCreateInfo> pipelineInfos{};

		for (VkPipelineShaderStageCreateInfo& stageInfo : stageInfos)
		{
			VkComputePipelineCreateInfo pipelineInfo{};
			pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
			pipelineInfo.layout = computePipelineLayout;
			pipelineInfo.stage = stageInfo;

			pipelineInfos.push_back(pipelineInfo);
		}

		std::vector<VkPipeline> pipelines(shaderModules.size());
		if (vkCreateComputePipelines(device, VK_NULL_HANDLE, pipelineInfos.size(), pipelineInfos.data(), nullptr, pipelines.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create compute pipelines.");
		}

		g2p2gComputePipeline = pipelines[0];
		processhistogramComputePipeline = pipelines[1];
		localsumComputePipeline = pipelines[2];
		partialglobalsumComputePipeline = pipelines[3];
		globalsumComputePipeline = pipelines[4];
		scatterComputePipeline = pipelines[5];
		totalextractionComputePipeline = pipelines[6];
		clearhistogramComputePipeline = pipelines[7];
		graphicsscatterComputePipeline = pipelines[8];
		gridComputePipeline = pipelines[9];

		for (VkShaderModule& module : shaderModules)
		{
			vkDestroyShaderModule(device, module, nullptr);
		}
	}

	void createFrameBuffers()
	{
		swapChainFramebuffers.resize(swapChainImageViews.size());

		for (size_t i = 0; i < swapChainImageViews.size(); i++)
		{
			VkImageView attachments[] =
			{
				swapChainImageViews[i]
			};

			VkFramebufferCreateInfo framebufferInfo{};
			framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			framebufferInfo.renderPass = renderPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = attachments;
			framebufferInfo.width = swapChainExtent.width;
			framebufferInfo.height = swapChainExtent.height;
			framebufferInfo.layers = 1;

			if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to create frameBuffer.");
			}
		}
	}

	void createCommandPools()
	{
		createCommandPool(graphicsFamilyIndex, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, &graphicsCommandPool);
		createCommandPool(graphicsFamilyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, &transientGraphicsCommandPool);

		createCommandPool(computeFamilyIndex, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, &computeCommandPool);
		createCommandPool(computeFamilyIndex, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, &transientComputeCommandPool);
	}

	void createCommandPool(uint32_t queueFamilyIndex, VkCommandPoolCreateFlags flags, VkCommandPool* commandPool)
	{
		VkCommandPoolCreateInfo poolInfo{};
		poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		poolInfo.flags = flags;
		poolInfo.queueFamilyIndex = queueFamilyIndex;

		if (vkCreateCommandPool(device, &poolInfo, nullptr, commandPool) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create command pool.");
		}
	}

	void createCommandBuffers()
	{
		commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = graphicsCommandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to allocate command buffers.");
		}
	}

	void createComputeCommandBuffers()
	{
		computeCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = computeCommandPool; // Probably change this for future async compute
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

		if (vkAllocateCommandBuffers(device, &allocInfo, computeCommandBuffers.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to allovate compute command buffers.");
		}
	}

	void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex)
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to begin recording command buffer.");
		}

		VkRenderPassBeginInfo renderPassInfo{};
		renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		renderPassInfo.renderPass = renderPass;
		renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
		renderPassInfo.renderArea.offset = { 0, 0 };
		renderPassInfo.renderArea.extent = swapChainExtent;

		VkClearValue clearColor = { {{0.0f, 0.0f, 0.0f, 1.0f}} };
		renderPassInfo.clearValueCount = 1;
		renderPassInfo.pClearValues = &clearColor;

		vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineLayout, 0, 1, &graphicsDescriptorSets[currentFrame], 0, nullptr);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		VkViewport viewPort{};
		viewPort.x = 0.0f;
		viewPort.y = 0.0f;
		viewPort.width = static_cast<float>(swapChainExtent.width);
		viewPort.height = static_cast<float>(swapChainExtent.height);
		viewPort.minDepth = 0.0f;
		viewPort.maxDepth = 1.0f;
		vkCmdSetViewport(commandBuffer, 0, 1, &viewPort);

		VkRect2D scissor{};
		scissor.offset = { 0, 0 };
		scissor.extent = swapChainExtent;
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

		VkDeviceSize offsets[] = { 0 };
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, &graphicsBuffers[currentFrame].buffer, offsets);

		vkCmdDraw(commandBuffer, PARTICLE_COUNT, 1, 0, 0);

		vkCmdEndRenderPass(commandBuffer);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to record command buffer.");
		}
	}

	void recordComputeCommandBuffer(VkCommandBuffer commandBuffer)
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = 0;
		beginInfo.pInheritanceInfo = nullptr;

		if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to begin recording compute command buffer.");
		}

		VkMemoryBarrier computeToComputeBarrier{};
		computeToComputeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		computeToComputeBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
		computeToComputeBarrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

		VkMemoryBarrier computeToIndirectBarrier{};
		computeToIndirectBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
		computeToIndirectBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;
		computeToIndirectBarrier.dstAccessMask = VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT;

		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 1, 1, &transferDescriptorSets[currentFrame], 0, nullptr);

		for (uint32_t i = 0; i < substeps; i++)
		{
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, computePipelineLayout, 0, 1, &substepDescriptorSets[currentCompute], 0, nullptr);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, g2p2gComputePipeline);
			vkCmdDispatch(commandBuffer, particleBlockDimensions, particleBlockDimensions, 1);

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				1, &computeToComputeBarrier,
				0, nullptr,
				0, nullptr
			);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, processhistogramComputePipeline);
			vkCmdDispatch(commandBuffer, paddedParticleBlockDimensions / BLOCK_KERNEL_SIZE, paddedParticleBlockDimensions / BLOCK_KERNEL_SIZE, 1);

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				1, &computeToComputeBarrier,
				0, nullptr,
				0, nullptr
			);

			// Look into https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, localsumComputePipeline);
			vkCmdDispatch(commandBuffer, paddedParticleBlockCount / SUM_KERNEL_SIZE, 1, 1);

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				1, &computeToComputeBarrier,
				0, nullptr,
				0, nullptr
			);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, partialglobalsumComputePipeline);
			vkCmdDispatch(commandBuffer, 1, 1, 1);

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				1, &computeToComputeBarrier,
				0, nullptr,
				0, nullptr
			);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, globalsumComputePipeline);
			vkCmdDispatch(commandBuffer, paddedParticleBlockCount / SUM_KERNEL_SIZE, 1, 1);

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
				0,
				1, &computeToIndirectBarrier,
				0, nullptr,
				0, nullptr
			);
			//

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scatterComputePipeline);
			vkCmdDispatchIndirect(commandBuffer, scatterIndirectDispatchBuffer.buffer, 0);

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				1, &computeToComputeBarrier,
				0, nullptr,
				0, nullptr
			);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, totalextractionComputePipeline);
			vkCmdDispatch(commandBuffer, 1, 1, 1);

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				1, &computeToComputeBarrier,
				0, nullptr,
				0, nullptr
			);

			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, clearhistogramComputePipeline);
			vkCmdDispatch(commandBuffer, paddedParticleBlockCount / (BLOCK_KERNEL_SIZE * BLOCK_KERNEL_SIZE), 1, 1);

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				0,
				1, &computeToComputeBarrier,
				0, nullptr,
				0, nullptr
			);

			// Theres no dependency between particle reordering and grid so change the positioning
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gridComputePipeline);
			vkCmdDispatch(commandBuffer, quadDimensions / GRID_KERNEL_SIZE, quadDimensions / GRID_KERNEL_SIZE, 1);

			vkCmdPipelineBarrier(commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
				0,
				1, &computeToIndirectBarrier,
				0, nullptr,
				0, nullptr
			);

			currentCompute = (currentCompute + 1) % 2;
		}

		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, graphicsscatterComputePipeline);
		vkCmdDispatchIndirect(commandBuffer, scatterIndirectDispatchBuffer.buffer, 0);

		if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to record compute command buffer");
		}
	}

	void createSyncObjects()
	{
		imageAquireSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
		inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
		computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		VkFenceCreateInfo fenceInfo{};
		fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
		fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAquireSemaphores[i]) != VK_SUCCESS ||
				vkCreateSemaphore(device, &semaphoreInfo, nullptr, &computeFinishedSemaphores[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS ||
				vkCreateFence(device, &fenceInfo, nullptr, &computeInFlightFences[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to create sync objects for a frame.");
			}
		}

		createSwapchainSyncObjects();
	}

	void createSwapchainSyncObjects()
	{
		VkSemaphoreCreateInfo semaphoreInfo{};
		semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

		imageSubmitSemaphores.resize(swapChainImages.size());
		for (size_t i = 0; i < swapChainImages.size(); i++)
		{
			if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageSubmitSemaphores[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed to create swapchain semaphores.");
			}
		}
	}

	void recreateSwapchain()
	{
		int width = 0;
		int height = 0;
		glfwGetFramebufferSize(window, &width, &height);
		while (width == 0 || height == 0)
		{
			glfwGetFramebufferSize(window, &width, &height);
			glfwWaitEvents();
		}

		vkDeviceWaitIdle(device);

		cleanupSwapchain();

		createSwapChain();
		createSwapchainImageViews();
		createFrameBuffers();
		createSwapchainSyncObjects();
	}

	void cleanupSwapchain()
	{
		for (auto submitSemaphore : imageSubmitSemaphores)
		{
			vkDestroySemaphore(device, submitSemaphore, nullptr);
		}

		for (auto framebuffer : swapChainFramebuffers)
		{
			vkDestroyFramebuffer(device, framebuffer, nullptr);
		}

		for (auto imageView : swapChainImageViews)
		{
			vkDestroyImageView(device, imageView, nullptr);
		}

		vkDestroySwapchainKHR(device, swapChain, nullptr);
	}

	// Make a method for these copies or smth
	void createShaderStorageBuffers()
	{
		binBuffers.resize(2);
		graphicsBuffers.resize(2);
		vQuadBuffers.resize(2);
		GQuadBuffers.resize(2);

		float pi = 3.14159265358979323846f;
		float worldR = size * (quadDimensions / 2) * dx;
		float volume = worldR * worldR * pi;
		float totalMass = volume * rho;
		float mass = totalMass / PARTICLE_COUNT;

		//std::default_random_engine rndEngine((unsigned)time(nullptr));
		std::default_random_engine rndEngine((unsigned)2u);
		std::uniform_real_distribution<float> rndDist(0.0f, 1.0f);

		std::vector<uint32_t> histogram(paddedParticleBlockCount);
		std::vector<Particle> particles(PARTICLE_COUNT);
		std::vector<uint32_t> blockIndices(PARTICLE_COUNT);
		std::vector<uint32_t> blockPIndex(PARTICLE_COUNT);
		for (size_t i = 0; i < PARTICLE_COUNT; i++)
		{
			//float r = size * sqrt(rndDist(rndEngine));
			//float theta = rndDist(rndEngine) * 2 * pi;
			//float x = r * cos(theta);
			//float y = r * sin(theta);

			float x = size * rndDist(rndEngine) * quadDimensions + quadDimensions / 2 - size * quadDimensions / 2;
			float y = size * rndDist(rndEngine) * quadDimensions + quadDimensions / 2 - size * quadDimensions / 2;
			//x = (x + 1.0f) / 2.0f * quadDimensions;
			//y = (y + 1.0f) / 2.0f * quadDimensions;

			glm::vec2 cellPos = glm::vec2(x, y);
			glm::vec2 pos = cellPos * dx;
			glm::ivec2 coords = glm::ivec2(cellPos - 2.5f);
			glm::ivec2 blockCoords = coords >> 2;
			uint32_t blockIndex = blockCoords.x + blockCoords.y * particleBlockDimensions;

			blockIndices[i] = blockIndex;
			blockPIndex[i] = histogram[blockIndex]++;
			particles[i].position = pos;
			particles[i].color = glm::vec4(rndDist(rndEngine), rndDist(rndEngine), rndDist(rndEngine), 1.0f);
		}

		std::vector<uint32_t> binCounts(paddedParticleBlockCount);
		std::vector<uint32_t> binOffsets(paddedParticleBlockCount);

		for (size_t i = 1; i < paddedParticleBlockCount; i++)
		{
			uint32_t binCount = ceilIntDivision(histogram[i - 1], BIN_SIZE);
			binCounts[i - 1] = binCount;
			binOffsets[i] = binCount + binOffsets[i - 1];

		}
		binCounts[paddedParticleBlockCount - 1] = ceilIntDivision(histogram[paddedParticleBlockCount - 1], BIN_SIZE);
		uint32_t totalUsedBins = binOffsets[paddedParticleBlockCount - 1] + binCounts[paddedParticleBlockCount - 1];

		std::vector<Bin> bins(binCount);
		for (size_t i = 0; i < PARTICLE_COUNT; i++)
		{
			glm::vec2 pos = particles[i].position;
			uint32_t blockIndex = blockIndices[i];
			uint32_t baseBin = binOffsets[blockIndex];
			uint32_t binIndex = baseBin + (blockPIndex[i] / BIN_SIZE);
			uint32_t pIndex = blockPIndex[i] % BIN_SIZE;

			bins[binIndex].F[pIndex] = glm::mat2(1.0f);
			bins[binIndex].position[pIndex] = pos;
			bins[binIndex].mass[pIndex] = mass;
			bins[binIndex].blockParticleIndex[pIndex] = blockPIndex[i];
			bins[binIndex].particleId[pIndex] = i;
			bins[binIndex].particleCount++;
		}

		// Bins
		VkDeviceSize bufferSize = sizeof(Bin) * binCount;

		for (size_t i = 0; i < 2; i++)
		{
			binBuffers[i] = bufferAllocator.create({
				bufferSize,
				val::BufferUsage::Storage,
				val::BufferLifetime::Static
				});

			cpuToGpuCopy(binBuffers[i].buffer, bufferSize, bins);
		}
		//

		// V
		std::vector<glm::vec2> v(quadCount);
		bufferSize = sizeof(glm::vec2) * quadCount;

		for (size_t i = 0; i < 2; i++)
		{
			vQuadBuffers[i] = bufferAllocator.create({
				bufferSize,
				val::BufferUsage::Storage,
				val::BufferLifetime::Static
				});

			cpuToGpuCopy(vQuadBuffers[i].buffer, bufferSize, v);
		}
		//

		// M
		std::vector<float> m(quadCount);
		bufferSize = sizeof(float) * quadCount;

		mQuadBuffer = bufferAllocator.create({
			bufferSize,
			val::BufferUsage::Storage,
			val::BufferLifetime::Static
			});

		cpuToGpuCopy(mQuadBuffer.buffer, bufferSize, m);
		//

		// Histogram
		std::vector<uint32_t> h(paddedParticleBlockCount);
		bufferSize = sizeof(uint32_t) * paddedParticleBlockCount;

		histogramBuffer = bufferAllocator.create({
			bufferSize,
			val::BufferUsage::Storage,
			val::BufferLifetime::Static
			});

		cpuToGpuCopy(histogramBuffer.buffer, bufferSize, h);
		//

		// BinCount
		bufferSize = sizeof(uint32_t) * paddedParticleBlockCount;

		binCountBuffer = bufferAllocator.create({
			bufferSize,
			val::BufferUsage::Storage,
			val::BufferLifetime::Static
			});

		cpuToGpuCopy(binCountBuffer.buffer, bufferSize, binCounts);
		//

		// BinOffsets
		bufferSize = sizeof(uint32_t) * paddedParticleBlockCount;

		binOffsetsBuffer = bufferAllocator.create({
			bufferSize,
			val::BufferUsage::Storage,
			val::BufferLifetime::Static
			});

		cpuToGpuCopy(binOffsetsBuffer.buffer, bufferSize, binOffsets);
		//

		// local sums
		std::vector<uint32_t> emptySums(BLOCK_KERNEL_SIZE * BLOCK_KERNEL_SIZE);
		bufferSize = sizeof(uint32_t) * BLOCK_KERNEL_SIZE * BLOCK_KERNEL_SIZE;

		binSumBuffer = bufferAllocator.create({
			bufferSize,
			val::BufferUsage::Storage,
			val::BufferLifetime::Static
			});

		cpuToGpuCopy(binSumBuffer.buffer, bufferSize, emptySums);
		//
		
		// Particles
		bufferSize = sizeof(Particle) * PARTICLE_COUNT;

		for (size_t i = 0; i < 2; i++)
		{
			graphicsBuffers[i] = bufferAllocator.create({
				bufferSize,
				val::BufferUsage::VertexStorage,
				val::BufferLifetime::Static
				});

			cpuToGpuCopy(graphicsBuffers[i].buffer, bufferSize, particles);
		}
		//

		// Scatter indirect dispatch
		bufferSize = sizeof(ScatterDispatchData);
		std::vector<ScatterDispatchData> s{};
		ScatterDispatchData scatterDispatchData{};
		scatterDispatchData.dispatchX = ceilIntDivision(totalUsedBins, BIN_KERNEL_SIZE);
		scatterDispatchData.dispatchY = 1;
		scatterDispatchData.dispatchZ = 1;
		s.push_back(scatterDispatchData);

		scatterIndirectDispatchBuffer = bufferAllocator.create({
			bufferSize,
			val::BufferUsage::Indirect,
			val::BufferLifetime::Static
			});

		cpuToGpuCopy(scatterIndirectDispatchBuffer.buffer, bufferSize, s);
		//

		// G buffers
		std::vector<glm::mat2> G(quadCount);
		bufferSize = sizeof(glm::mat2) * quadCount;

		for (size_t i = 0; i < 2; i++)
		{
			GQuadBuffers[i] = bufferAllocator.create({
				bufferSize,
				val::BufferUsage::Indirect,
				val::BufferLifetime::Static
				});

			cpuToGpuCopy(GQuadBuffers[i].buffer, bufferSize, G);
		}
		//
		
		// v Node buffer
		std::vector<float> vN(nodeCount);
		bufferSize = sizeof(float) * nodeCount;

		vNodeBuffer = bufferAllocator.create({
			bufferSize,
			val::BufferUsage::Indirect,
			val::BufferLifetime::Static
			});

		cpuToGpuCopy(vNodeBuffer.buffer, bufferSize, vN);

		//
		
		// Maybe create seperate transferqueue?
		vkQueueWaitIdle(graphicsQueue);
		//
	}

	template<typename T>
	void cpuToGpuCopy(VkBuffer copyTo, VkDeviceSize size, std::vector<T> data)
	{
		val::AllocatedBuffer stagingBuffer = createStagingBuffer(size);
		memcpy(stagingBuffer.mapped, data.data(), (size_t)size);
		copyBuffer(stagingBuffer.buffer, copyTo, size);

		stagingBuffer.dispose();
	}

	val::AllocatedBuffer createStagingBuffer(VkDeviceSize bufferSize)
	{
		val::AllocatedBuffer stagingBuffer = bufferAllocator.create({
			bufferSize,
			val::BufferUsage::Staging,
			val::BufferLifetime::Dynamic
			});

		return stagingBuffer;
	}

	uint32_t ceilIntDivision(uint32_t a, uint32_t b)
	{
		return a / b + (a % b != 0);
	}

	void createUniformBuffers()
	{
		VkDeviceSize bufferSize = sizeof(ParameterUBO);

		parameterBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		cameraBuffers.resize(MAX_FRAMES_IN_FLIGHT);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			parameterBuffers[i] = bufferAllocator.create({
				bufferSize,
				val::BufferUsage::Uniform,
				val::BufferLifetime::Dynamic
				});

			cameraBuffers[i] = bufferAllocator.create({
				bufferSize,
				val::BufferUsage::Uniform,
				val::BufferLifetime::Dynamic
				});
		}
	}

	void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.extent.width = width;
		imageInfo.extent.height = height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;

		imageInfo.format = format;
		imageInfo.tiling = tiling;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		imageInfo.usage = usage;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.flags = 0;

		if (vkCreateImage(device, &imageInfo, nullptr, &image) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create image.");
		}

		VkMemoryRequirements memRequirements;
		vkGetImageMemoryRequirements(device, image, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to allocate image memory.");
		}

		vkBindImageMemory(device, image, imageMemory, 0);
	}

	void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkImageMemoryBarrier barrier{};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = oldLayout;
		barrier.newLayout = newLayout;

		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

		barrier.image = image;
		barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;

		VkPipelineStageFlags srcStageMask;
		VkPipelineStageFlags dstStageMask;

		if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_GENERAL)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;

			srcStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
			dstStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
		}
		else if (oldLayout == VK_IMAGE_LAYOUT_GENERAL && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

			srcStageMask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
			dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else
		{
			throw std::invalid_argument("Unsupported layout transition.");
		}

		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = 0;

		vkCmdPipelineBarrier(
			commandBuffer,
			srcStageMask, dstStageMask,
			0,
			0, nullptr,
			0, nullptr,
			1, &barrier
		);

		endSingleTimeCommands(commandBuffer);
	}

	void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferImageCopy region{};
		region.bufferOffset = 0;
		region.bufferRowLength = 0;
		region.bufferImageHeight = 0;

		region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		region.imageSubresource.mipLevel = 0;
		region.imageSubresource.baseArrayLayer = 0;
		region.imageSubresource.layerCount = 1;

		region.imageOffset = { 0, 0, 0 };
		region.imageExtent = {
			width,
			height,
			1
		};

		vkCmdCopyBufferToImage(
			commandBuffer,
			buffer,
			image,
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
			1,
			&region
		);

		endSingleTimeCommands(commandBuffer);
	}

	void clearImage(VkImage image)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkClearColorValue clear{};
		clear.float32[0] = 0.0f;
		clear.float32[1] = 0.0f;
		clear.float32[2] = 0.0f;
		clear.float32[3] = 0.0f;

		VkImageSubresourceRange range{};
		range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		range.baseArrayLayer = 0;
		range.layerCount = 1;
		range.baseMipLevel = 0;
		range.levelCount = 1;

		vkCmdClearColorImage(
			commandBuffer, 
			image, 
			VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 
			&clear, 
			1, 
			&range);

		endSingleTimeCommands(commandBuffer);
	}

	void createImageView(VkImage image, VkImageView& imageView, VkFormat format)
	{
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = format;

		viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		viewInfo.subresourceRange.baseMipLevel = 0;
		viewInfo.subresourceRange.levelCount = 1;
		viewInfo.subresourceRange.baseArrayLayer = 0;
		viewInfo.subresourceRange.layerCount = 1;

		viewInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
		viewInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

		if (vkCreateImageView(device, &viewInfo, nullptr, &imageView) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create image view.");
		}
	}

	void createBuffer(VkDeviceSize bufferSize, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory)
	{
		VkBufferCreateInfo bufferInfo{};
		bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		bufferInfo.size = bufferSize;
		bufferInfo.usage = usage;
		bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create vertex buffer.");
		}

		VkMemoryRequirements memRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

		VkMemoryAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memRequirements.size;
		allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

		if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to allocate vertex buffer memory.");
		}

		vkBindBufferMemory(device, buffer, bufferMemory, 0);
	}

	void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
	{
		VkCommandBuffer commandBuffer = beginSingleTimeCommands();

		VkBufferCopy copyRegion{};
		copyRegion.srcOffset = 0;
		copyRegion.dstOffset = 0;
		copyRegion.size = size;
		vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

		endSingleTimeCommands(commandBuffer);
	}

	VkCommandBuffer beginSingleTimeCommands()
	{
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = transientGraphicsCommandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeCommands(VkCommandBuffer commandBuffer)
	{
		vkEndCommandBuffer(commandBuffer);
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, transientGraphicsCommandPool, 1, &commandBuffer);
	}

	VkCommandBuffer beginSingleTimeComputeCommands()
	{
		VkCommandBufferAllocateInfo allocInfo{};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandPool = transientComputeCommandPool;
		allocInfo.commandBufferCount = 1;

		VkCommandBuffer commandBuffer;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(commandBuffer, &beginInfo);

		return commandBuffer;
	}

	void endSingleTimeComputeCommands(VkCommandBuffer commandBuffer)
	{
		vkEndCommandBuffer(commandBuffer);

		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
		vkQueueWaitIdle(graphicsQueue);

		vkFreeCommandBuffers(device, transientComputeCommandPool, 1, &commandBuffer);
	}

	uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
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

	VkShaderModule createShaderModule(const std::vector<char>& code)
	{
		VkShaderModuleCreateInfo createInfo{};

		createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		createInfo.codeSize = code.size();
		createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

		VkShaderModule shaderModule;
		if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed to create shader module.");
		}

		return shaderModule;
	}

	void cleanup()
	{
		cleanupSwapchain();

		vkDestroyPipeline(device, graphicsPipeline, nullptr);
		vkDestroyPipelineLayout(device, graphicsPipelineLayout, nullptr);

		vkDestroyPipeline(device, g2p2gComputePipeline, nullptr);
		vkDestroyPipeline(device, processhistogramComputePipeline, nullptr);
		vkDestroyPipeline(device, localsumComputePipeline, nullptr);
		vkDestroyPipeline(device, partialglobalsumComputePipeline, nullptr);
		vkDestroyPipeline(device, globalsumComputePipeline, nullptr);
		vkDestroyPipeline(device, totalextractionComputePipeline, nullptr);
		vkDestroyPipeline(device, scatterComputePipeline, nullptr);
		vkDestroyPipeline(device, clearhistogramComputePipeline, nullptr);
		vkDestroyPipeline(device, graphicsscatterComputePipeline, nullptr);
		vkDestroyPipeline(device, gridComputePipeline, nullptr);
		vkDestroyPipelineLayout(device, computePipelineLayout, nullptr);

		vkDestroyRenderPass(device, renderPass, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			parameterBuffers[i].dispose();
			cameraBuffers[i].dispose();
		}

		graphicsDescriptorAllocator.destroyPool(device);
		substepComputeDescriptorAllocator.destroyPool(device);
		transferComputeDescriptorAllocator.destroyPool(device);

		vkDestroyDescriptorSetLayout(device, graphicsDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, substepComputeDescriptorSetLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, transferComputeDescriptorSetLayout, nullptr);

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vQuadBuffers[i].dispose();
			GQuadBuffers[i].dispose();
			binBuffers[i].dispose();
		}

		mQuadBuffer.dispose();
		histogramBuffer.dispose();
		binCountBuffer.dispose();
		binOffsetsBuffer.dispose();
		binSumBuffer.dispose();
		scatterIndirectDispatchBuffer.dispose();
		vNodeBuffer.dispose();

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			graphicsBuffers[i].dispose();
		}

		for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
		{
			vkDestroySemaphore(device, imageAquireSemaphores[i], nullptr);
			vkDestroySemaphore(device, computeFinishedSemaphores[i], nullptr);
			vkDestroyFence(device, inFlightFences[i], nullptr);
			vkDestroyFence(device, computeInFlightFences[i], nullptr);
		}

		vkDestroyCommandPool(device, graphicsCommandPool, nullptr);
		vkDestroyCommandPool(device, transientGraphicsCommandPool, nullptr);

		vkDestroyCommandPool(device, computeCommandPool, nullptr);
		vkDestroyCommandPool(device, transientComputeCommandPool, nullptr);

		vkDestroyDevice(device, nullptr);

		if (enableValidationLayers)
		{
			DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);
		}

		vkDestroySurfaceKHR(instance, surface, nullptr);
		vkDestroyInstance(instance, nullptr);

		glfwDestroyWindow(window);

		glfwTerminate();
	}

	static std::vector<char> readFile(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("Failed to open file " + filename + ".");
		}

		size_t fileSize = (size_t)file.tellg();
		std::vector<char> buffer(fileSize);

		file.seekg(0);
		file.read(buffer.data(), fileSize);

		file.close();

		return buffer;
	}

	static void framebufferResizeCallback(GLFWwindow* window, int width, int height)
	{
		auto app = reinterpret_cast<MpmVulkan*>(glfwGetWindowUserPointer(window));
		app->framebufferResized = true;
	}
};

int main()
{
	MpmVulkan app;

	try
	{
		app.run();
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
