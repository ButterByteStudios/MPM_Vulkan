add_library(dsl STATIC
	"descriptorLayoutBuilder.cpp"
	"descriptorAllocator.cpp")

target_include_directories(dsl
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dsl
	PRIVATE 
		Vulkan::Vulkan
)