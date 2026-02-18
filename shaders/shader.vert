#version 450

layout(binding = 0) uniform ParameterUBO
{
	float k;
	float mu;
	float rho;
	float dx;
	float invDx;
	uint dimensions;
	uint blockDimensions;
	float dt;
	float invDt;
} ubo;

layout(binding = 1) uniform CameraUBO
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
	gl_PointSize = 1.0;
	
	// Map the position from 0 - dimensions to 0 - (dimensions - 2);
	vec2 pos = (inPosition * ubo.invDx - 1) / (ubo.dimensions - 2) * 2.0 - 1.0;
	vec2 offset = pos - cam.pos;
	vec2 scaled = offset * cam.zoom;
	gl_Position = vec4(scaled.x * cam.invAspectRatio, -scaled.y, 0.0, 1.0);
	fragColor = inColor.rgb;
}