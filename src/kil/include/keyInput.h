#include <glfw/glfw3.h>
#include <vector>
#include <map>

// https://stackoverflow.com/questions/55573238/how-do-i-do-a-proper-input-class-in-glfw-for-a-game-engine
class KeyInput
{
public:
	KeyInput(std::vector<int> keysToMonitor);
	~KeyInput();

	// Add stuff like is down and waspressedthisframe
	bool getIsKeyDown(int key);

	bool getIsEnabled() { return isEnabled; }
	void setIsEnabled(bool value) { isEnabled = value; }
private:
	void setIsKeyDown(int key, bool isDown);

	std::map<int, bool> keys;
	bool isEnabled;

public:
	static void setupKeyInputs(GLFWwindow* window);
private:
	static void callback(GLFWwindow* window, int key, int scancode, int action, int mods);

	static std::vector<KeyInput*> instances;
};