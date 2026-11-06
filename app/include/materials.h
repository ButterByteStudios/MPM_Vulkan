#include <cstdint>
#include <string>
#include <functional>
#include <array>
#include <vector>

// The display of one material input
struct InputData
{
public:
	enum class InputType
	{
		INPUT_FLOAT,
		SLIDER_FLOAT,
		INPUT_INT,
		SLIDER_INT
	};

	InputType inputType;
	std::string inputName;
	uint32_t flags;
	float param1;
	float param2;
	float defaultValue;
	std::string description;

	InputData(std::string inputName, InputType inputType = InputType::INPUT_FLOAT, float param1 = 0.0f, float param2 = 0.0f, float defaultValue = 0.0f, uint32_t flags = 0u, std::string description = "");

	bool renderInput(void* param, uint32_t& uiId);
};

struct alignas(4) MaterialLayout
{
	// i0 = rho
	// i1 = 0: lambda, 1: k
	// i2 = 0: mu, 1: mu
	// i3 = 1: beta
	// i4 = 1: M
	// i5 = 1: xi
	float params[6];
	int materialType;
};

using Parameters = std::array<float, 6>;
using Conversion = std::function<float(const Parameters&)>;

class Material;

// Define a material type using a number of inputs
class MaterialType
{
public:
	std::vector<InputData> inputs;
	std::vector<Conversion> conversions;
	uint32_t index;

	MaterialType(uint32_t index, std::vector<InputData> inputs, std::vector<Conversion> conversions);

	MaterialLayout toMaterialLayout(const Material& mat);

	Material initializeMaterial();
};

// Material to material layout: on the mpm-vulkan side store per material a function that maps the parameters array to material layouts
class Material
{
public:
	Parameters params;
	uint32_t materialTypeIndex;

	Material(uint32_t materialTypeIndex);
};

class MaterialInstance
{
public:
	MaterialInstance(uint32_t id, Material mat);
	bool renderInstance(uint32_t& uiId, std::vector<MaterialType>& materialTypes, std::vector<std::string>& typeNames);

	Material mat;
private:
	std::string name;
	char buf[255]{};
	uint32_t id;
};

class MaterialsManager
{
public:
	std::vector<MaterialLayout> materialLayouts;

	bool isDirty();
	void setDirty();
	void setClean();

	void addMaterialType(std::string, std::vector<InputData> inputs, std::vector<Conversion> conversions);

	void createMaterial(uint32_t type);

	void render();

	MaterialsManager(uint32_t materialCount);

	MaterialsManager();

private:
	void updateMaterialLayout(uint32_t index);

	std::vector<MaterialType> materialTypes;
	std::vector<std::string> typeNames;
	std::vector<MaterialInstance> items;
	bool dirty = false; // Can be changed to a layout version int64 if there are multiple consumers (like for rendering)
	uint32_t maxMaterialCount;

private:
};