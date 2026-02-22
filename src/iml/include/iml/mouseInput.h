#include <glfw/glfw3.h>
#include <vector>
#include <map>

namespace iml
{
	class MouseInput
	{
	public:
		MouseInput(std::vector<int> buttonsToMonitor);
		~MouseInput();

		bool getIsButtonDown(int button);

		bool getIsEnabled() { return isEnabled; }
		void setIsEnabled(bool value) { isEnabled = value; }

	private:
		void setIsButtonDown(int button, bool isDown);

		std::map<int, bool> buttons;
		bool isEnabled;

	public:
		static void setupMouseInputs(GLFWwindow* window);
		static void clearFrameValues(); // Use for "waspressedthisframe"?

		static double scrollDX;
		static double scrollDY;
		static double posX;
		static double posY;
	private:
		static void buttonCallback(GLFWwindow* window, int button, int action, int mods);
		static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
		static void moveCallback(GLFWwindow* window, double xpos, double ypos);

		static std::vector<MouseInput*> instances;
	};
}