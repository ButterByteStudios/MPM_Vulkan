#include <iml/mouseInput.h>
#include <algorithm>

namespace iml
{
	std::vector<MouseInput*> MouseInput::instances;

	double MouseInput::posX;
	double MouseInput::posY;
	double MouseInput::scrollDX;
	double MouseInput::scrollDY;

	MouseInput::MouseInput(std::vector<int> buttonsToMonitor) : isEnabled(true)
	{
		for (int button : buttonsToMonitor)
		{
			buttons[button] = false;
		}

		MouseInput::instances.push_back(this);
	}

	MouseInput::~MouseInput()
	{
		instances.erase(std::remove(instances.begin(), instances.end(), this), instances.end());
	}

	void MouseInput::setupMouseInputs(GLFWwindow* window)
	{
		glfwSetMouseButtonCallback(window, MouseInput::buttonCallback);
		glfwSetScrollCallback(window, MouseInput::scrollCallback);
		glfwSetCursorPosCallback(window, MouseInput::moveCallback);
	}

	void MouseInput::clearFrameValues()
	{
		MouseInput::scrollDX = 0;
		MouseInput::scrollDY = 0;
	}

	bool MouseInput::getIsButtonDown(int button)
	{
		bool result = false;
		if (isEnabled)
		{
			std::map<int, bool>::iterator it = buttons.find(button);
			if (it != buttons.end())
			{
				result = buttons[button];
			}
		}

		return result;
	}

	void MouseInput::setIsButtonDown(int key, bool isDown)
	{
		std::map<int, bool>::iterator it = buttons.find(key);
		if (it != buttons.end())
		{
			buttons[key] = isDown;
		}
	}

	void MouseInput::buttonCallback(GLFWwindow* window, int button, int action, int mods)
	{
		for (MouseInput* mouseInput : instances)
		{
			mouseInput->setIsButtonDown(button, action != GLFW_RELEASE);
		}
	}

	void MouseInput::scrollCallback(GLFWwindow* window, double xoffset, double yoffset)
	{
		MouseInput::scrollDX = xoffset;
		MouseInput::scrollDY = yoffset;
	}

	void MouseInput::moveCallback(GLFWwindow* window, double xpos, double ypos)
	{
		MouseInput::posX = xpos;
		MouseInput::posY = ypos;
	}
}