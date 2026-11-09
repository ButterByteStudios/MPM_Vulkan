add_library(uil STATIC
	"src/UIManager.cpp"
	"src/MaterialsMenu.cpp"
	"src/MaterialsManager.cpp")

target_include_directories(uil
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(uil
	PRIVATE
	Vulkan::Vulkan
	imgui
)