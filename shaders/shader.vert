#version 450

#define BIN_SIZE (64)

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

layout(set = 1, std430, binding = 8) readonly buffer BinSSBOIn
{
	Bin binsIn[];
};

// Render straight out of bins. Start binsused*binsize vertices using the indirect buffer with 3*32 bit offset. Every vertex is a particle in the used bins and we can just discard the particles outside out the particlecount

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
	gl_PointSize = 2.0;
	
	// Map position from 0-dimensions to 0-1
	vec2 pos = inPosition / pubo.dimensions * 2.0 - 1.0;
	vec2 offset = pos - cam.pos;
	vec2 scaled = offset * cam.zoom;
	gl_Position = vec4(scaled.x * cam.invAspectRatio, -scaled.y, 0.0, 1.0);
	fragColor = inColor.rgb;
}