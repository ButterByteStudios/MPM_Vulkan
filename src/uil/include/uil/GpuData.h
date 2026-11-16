#pragma once

#include <variant>
#include <glm/glm.hpp>

namespace uil
{
	struct BrushToolData
	{
		glm::vec2 pos;
		glm::vec2 previousPos;
		float brushSize;
		uint32_t brushShape;
		uint32_t particleCount;
		uint32_t seed;
		float mass;
		uint32_t materialId;
	};

	struct EraseToolData
	{

	};

	using ToolGPUData = std::variant<BrushToolData, EraseToolData>;
}