add_library(pbl STATIC
	"src/barrierBuilder.cpp"
)

target_include_directories(pbl
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pbl
	PRIVATE
	Vulkan::Vulkan
)