#include "barrierBuilder.h"
#include "unordered_map"

namespace pbl
{
	UseDetails toUsageInfo(BufferUse use)
	{
		switch (use)
		{
		case BufferUse::ComputeRead:
			return { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT };

		case BufferUse::ComputeWrite:
			return { VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT };

		case BufferUse::IndirectRead:
			return { VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT };

		case BufferUse::VertexRead:
			return { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_ACCESS_SHADER_READ_BIT };
		}
	}

	std::unordered_map<VkBuffer, TrackInfo> trackedBufferMap;

	void BarrierBuilder::addBuffer(VkBuffer& buffer, VkDeviceSize size, BufferState state)
	{
		TrackInfo info{};
		info.buffer = buffer;
		info.size = size;
		info.state = state;

		trackedBufferMap.try_emplace(buffer, info);
	}
}