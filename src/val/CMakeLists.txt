add_library(val STATIC
	"src/vulkanAllocator.cpp"
)

target_include_directories(val
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(val
	PRIVATE
	Vulkan::Vulkan
)