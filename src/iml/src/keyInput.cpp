#include <iml/keyInput.h>
#include <algorithm>

namespace iml
{
	std::vector<KeyInput*> KeyInput::instances;

	KeyInput::KeyInput(std::vector<int> keysToMonitor) : isEnabled(true)
	{
		for (int key : keysToMonitor)
		{
			keys[key] = KeyState();
		}

		KeyInput::instances.push_back(this);
	}

	KeyInput::~KeyInput()
	{
		instances.erase(std::remove(instances.begin(), instances.end(), this), instances.end());
	}

	void KeyInput::clearFrameValues()
	{
		for (KeyInput* instance : instances)
		{
			KeyInput& input = *instance;
			for (auto& [key, state] : input.keys)
			{
				state.wasPressed = false;
				state.wasReleased = false;
			}
		}
	}

	bool KeyInput::isKeyDown(int key)
	{
		bool result = false;
		if (isEnabled)
		{
			std::map<int, KeyState>::iterator it = keys.find(key);
			if (it == keys.end())
				return false;

			result = it->second.isDown;
		}

		return result;
	}

	bool KeyInput::wasKeyPressed(int key)
	{
		bool result = false;
		if (isEnabled)
		{
			std::map<int, KeyState>::iterator it = keys.find(key);
			if (it == keys.end())
				return false;

			result = it->second.wasPressed;
		}

		return result;
	}

	bool KeyInput::wasKeyReleased(int key)
	{
		bool result = false;
		if (isEnabled)
		{
			std::map<int, KeyState>::iterator it = keys.find(key);
			if (it == keys.end())
				return false;

			result = it->second.wasReleased;
		}

		return result;
	}

	void KeyInput::setKeyState(int key, int action)
	{
		std::map<int, KeyState>::iterator it = keys.find(key);
		if (it == keys.end())
			return;
		
		KeyState& state = it->second;

		if (action == GLFW_RELEASE && state.isDown)
		{
			state.wasReleased = true;
			state.isDown = false;
		}
		if (action == GLFW_PRESS && !state.isDown)
		{
			state.wasPressed = true;
			state.isDown = true;
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
			keyInput->setKeyState(key, action);
		}
	}
}