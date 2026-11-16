#pragma once

#include <uil/GpuData.h>
#include <variant>
#include <glm/glm.hpp>

namespace uil
{
	class Tool
	{
	public:
		virtual ~Tool() = default;

		virtual void Enable() = 0;
		virtual void Disable() = 0;

		virtual void update(float deltaTime) = 0;
		virtual void handleInput() = 0;

		virtual const char* getName() const = 0;

		virtual ToolGPUData getGPUData() const = 0;
	};
}