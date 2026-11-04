#include <glfw/glfw3.h>
#include <vector>
#include <map>

// https://stackoverflow.com/questions/55573238/how-do-i-do-a-proper-input-class-in-glfw-for-a-game-engine

namespace iml
{
	struct KeyState
	{
		bool isDown = false;
		bool wasPressed = false;
		bool wasReleased = false;
	};

	class KeyInput
	{
	public:
		KeyInput(std::vector<int> keysToMonitor);
		~KeyInput();

		bool isKeyDown(int key);

		bool wasKeyPressed(int key);

		bool wasKeyReleased(int key);

		bool getIsEnabled() { return isEnabled; }
		void setIsEnabled(bool value) { isEnabled = value; }
	private:
		void setKeyState(int key, int action);

		std::map<int, KeyState> keys;
		bool isEnabled;

	public:
		static void setupKeyInputs(GLFWwindow* window);
		static void clearFrameValues();
	private:
		static void callback(GLFWwindow* window, int key, int scancode, int action, int mods);

		static std::vector<KeyInput*> instances;
	};
}