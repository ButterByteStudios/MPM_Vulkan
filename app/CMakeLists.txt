add_executable(MpmVulkan "MpmVulkan.cpp")

add_dependencies(MpmVulkan shaders)

target_link_libraries(MpmVulkan 
	PRIVATE 
		glfw
		glm
		Vulkan::Vulkan
		dsl
		ldl
		kil
)
