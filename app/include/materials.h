#include <cstdint>
#include <string>

class Material
{
public:
	float E = 100000;
	float v = 0.45f;
	float rho = 2000;
};

class MaterialNode
{
public:
	MaterialNode(uint32_t id);

	bool renderNode();

private:
	Material mat;
	std::string name;
	char buf[255]{};
	uint32_t id;
};