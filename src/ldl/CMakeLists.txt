add_library(ldl STATIC
	"deviceBuilder.cpp"
)

target_include_directories(ldl
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(ldl
	PRIVATE
	Vulkan::Vulkan
)