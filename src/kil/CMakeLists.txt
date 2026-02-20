add_library(kil STATIC
	"keyInput.cpp"
)

target_include_directories(kil
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(kil
	PRIVATE 
		glfw
)