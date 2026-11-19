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

layout(location = 0) out vec3 fragColor;

vec3 OklabToRgb(vec3 lab)
{
	// 1. Convert Oklab to LMS
	float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
	float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
	float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;

	// 2. Cube values
	float l = l_ * l_ * l_;
	float m = m_ * m_ * m_;
	float s = s_ * s_ * s_;

	// 3. LMS to linear sRGB
	float R = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s;
	float G = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s;
	float B = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s;

	// 4. Clamp to [0,1]
	R = clamp(R, 0, 1);
	G = clamp(G, 0, 1);
	B = clamp(B, 0, 1);

	return vec3(R, G, B);
}

vec3 RgbToOklab(vec3 rgb)
{
	// 1. Linear RGB to LMS
	float r = pow(rgb.r, 1.0f / 2.2f);
	float g = pow(rgb.g, 1.0f / 2.2f);
	float b = pow(rgb.b, 1.0f / 2.2f);

	float l = 0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b;
	float m = 0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b;
	float s = 0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b;

	// 2. Cube roots
	float l_ = pow(l, 1.0 / 3.0);
	float m_ = pow(m, 1.0 / 3.0);
	float s_ = pow(s, 1.0 / 3.0);

	// 3. LMS to Oklab
	float L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_;
	float a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_;
	float b_ = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_;

	return vec3(L, a, b_);
}

void main()
{
	int particle = gl_VertexIndex;
	int bin = particle >> LOG2_BIN_SIZE;
	uint particleInBin = particle & BIN_MASK;
	uint particleCount = binsIn[bin].particleCount;

	if (particleInBin >= particleCount)
	{
		gl_PointSize = 0.0;
		// Map position from 0-dimensions to 0-1
		gl_Position = vec4(100000, 100000, 0.0, 0.0);
		fragColor = vec3(0, 0, 0);
	}
	else
	{
		float cflSpeed = 0.1 * pubo.invDt;
		gl_PointSize = 4.0;
		// Map position from 0-dimensions to -0.5-0.5
		vec2 pos = binsIn[bin].position[particleInBin] / pubo.dimensions - 0.5;
		vec2 relative = pos - cam.pos;
		vec2 scaled = relative * cam.zoom;
		vec2 clipPos = scaled * 2.0;
		gl_Position = vec4(clipPos.x * cam.invAspectRatio, -clipPos.y, 0.0, 1.0);

		vec2 col = binsIn[bin].position[particleInBin] / pubo.dimensions;

		fragColor = OklabToRgb(vec3(1, col - 0.5));
	}
}