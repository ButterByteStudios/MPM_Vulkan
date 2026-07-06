#include <cstdint>
#include <string>

class Material
{
public:
	float k = 0;
	float mu = 0;
	float rho = 0;
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