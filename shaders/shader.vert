#version 450

#define BIN_SIZE (64)

#define LOG2_BIN_SIZE (6)
#define BIN_MASK (BIN_SIZE - 1u)

struct Bin
{
	mat2[BIN_SIZE] F;
	vec2[BIN_SIZE] position;
	float[BIN_SIZE] mass;
	float[BIN_SIZE] hardening;
	uint[BIN_SIZE] particleInBlock;
	uint[BIN_SIZE] particleId;
	uint[BIN_SIZE] materialId;
	uint particleCount;
};

layout(set = 1, std430, binding = 0) readonly buffer BinSSBO1
{
	Bin binsIn[];
};

layout(set = 1, std430, binding = 1) readonly buffer BinSSBO2
{
	Bin binsOut[];
};

layout(set = 0, binding = 0) uniform ParameterUBO
{
	vec2 speed;
	float dx;
	float invDx;
	uint dimensions;
	uint blockDimensions;
	float dt;
	float invDt;
} pubo;

layout(set = 0, binding = 1) uniform CameraUBO
{
	vec2 pos;
	float invAspectRatio;
	float zoom;
} cam;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

void main()
{
	int particle = gl_VertexIndex;
	int bin = particle >> LOG2_BIN_SIZE;
	uint particleInBin = particle & BIN_MASK;
	uint particleCount = binsIn[particleInBin].particleCount;

	if (particleInBin >= particleCount)
	{
		gl_PointSize = 0.0;
		// Map position from 0-dimensions to 0-1
		gl_Position = vec4(100000, 100000, 0.0, 1.0);
		fragColor = vec3(0, 0, 0);
	}
	else
	{
		gl_PointSize = 2.0;
		// Map position from 0-dimensions to 0-1
		vec2 pos = binsIn[bin].position[particleInBin] / pubo.dimensions * 2.0 - 1.0;
		vec2 offset = pos - cam.pos;
		vec2 scaled = offset * cam.zoom;
		gl_Position = vec4(scaled.x * cam.invAspectRatio, -scaled.y, 0.0, 1.0);
		fragColor = vec3(1, 1, 1);
	}
}