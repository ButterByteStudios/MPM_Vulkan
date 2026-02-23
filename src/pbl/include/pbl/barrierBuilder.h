#include <vulkan/vulkan.h>

namespace pbl
{
	enum class BufferUse
	{
		ComputeRead,
		ComputeWrite,
		VertexRead,
		IndirectRead
	};

	struct UseDetails
	{
		VkPipelineStageFlags stage;
		VkAccessFlags mask;
	};

	struct BufferState
	{
		VkPipelineStageFlags stage;
		VkAccessFlags mask;
	};

	struct TrackInfo
	{
		VkBuffer buffer;
		VkDeviceSize size;
		BufferState state;
	};

	struct Sequence
	{

	};

	class Info
	{
		
	};

	UseDetails toUsageInfo(BufferUse use);

	class BarrierBuilder
	{
		void addBuffer(VkBuffer& buffer, VkDeviceSize size, BufferState state);
	};
}