add_library(iml STATIC
	"src/keyInput.cpp"
	"src/mouseInput.cpp"
)

target_include_directories(iml
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(iml
	PRIVATE 
		glfw
)