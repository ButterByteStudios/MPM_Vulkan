include(FetchContent)

FetchContent_Declare(
	glfw
	GIT_REPOSITORY https://github.com/glfw/glfw
	GIT_TAG 3.4
)

FetchContent_MakeAvailable(glfw)

FetchContent_Declare(
	glm
	GIT_REPOSITORY https://github.com/g-truc/glm
	GIT_TAG 1.0.3
)

FetchContent_MakeAvailable(glm)

add_executable(MpmVulkan "MpmVulkan.cpp")

target_link_libraries(MpmVulkan 
	PRIVATE 
		glfw
		glm
		Vulkan::Vulkan
		dsl
)
