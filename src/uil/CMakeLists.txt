add_library(uil STATIC
	"src/UI/UIManager.cpp"
	"src/UI/MaterialsMenu.cpp"
	"src/Materials/MaterialsManager.cpp" 
	"src/UI/ToolsMenu.cpp"
	"src/Tools/ToolsManager.cpp" "src/Tools/Tool.cpp")

target_include_directories(uil
	PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(uil
	PRIVATE
	Vulkan::Vulkan
	imgui
)