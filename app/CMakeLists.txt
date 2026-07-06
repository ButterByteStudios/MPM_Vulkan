add_executable(MpmVulkan "MpmVulkan.cpp" "materials.cpp" "include/materials.h")

target_include_directories(MpmVulkan
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_dependencies(MpmVulkan shaders)

target_link_libraries(MpmVulkan 
	PRIVATE 
		glfw
		glm
		imgui
		Vulkan::Vulkan
		dsl
		ldl
		iml
		val
)
