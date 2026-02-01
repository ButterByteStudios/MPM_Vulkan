#version 450

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec3 inColor;

layout(location = 0) out vec3 fragColor;

layout(push_constant) uniform PushConstants {
	float dt;
	float dx;
	float invDx;
	int dimensions;
} pc;

void main()
{
	gl_PointSize = 14.0;
	gl_Position = vec4(-(inPosition.xy * pc.invDx / pc.dimensions * 2.0 - 1.0), 0.0, 1.0);
	fragColor = inColor.rgb;
}