#include <iml/keyInput.h>
#include <algorithm>

namespace iml
{
	std::vector<KeyInput*> KeyInput::instances;

	KeyInput::KeyInput(std::vector<int> keysToMonitor) : isEnabled(true)
	{
		for (int key : keysToMonitor)
		{
			keys[key] = false;
		}

		KeyInput::instances.push_back(this);
	}

	KeyInput::~KeyInput()
	{
		instances.erase(std::remove(instances.begin(), instances.end(), this), instances.end());
	}

	bool KeyInput::getIsKeyDown(int key)
	{
		bool result = false;
		if (isEnabled)
		{
			std::map<int, bool>::iterator it = keys.find(key);
			if (it != keys.end())
			{
				result = keys[key];
			}
		}

		return result;
	}

	void KeyInput::setIsKeyDown(int key, bool isDown)
	{
		std::map<int, bool>::iterator it = keys.find(key);
		if (it != keys.end())
		{
			keys[key] = isDown;
		}
	}

	void KeyInput::setupKeyInputs(GLFWwindow* window)
	{
		glfwSetKeyCallback(window, KeyInput::callback);
	}

	void KeyInput::callback(GLFWwindow* window, int key, int scancode, int action, int mods)
	{
		for (KeyInput* keyInput : instances)
		{
			keyInput->setIsKeyDown(key, action != GLFW_RELEASE);
		}
	}
}