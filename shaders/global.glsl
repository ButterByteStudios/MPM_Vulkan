#define BIND_UBO (0)
#define BIND_VR (1)
#define BIND_VW (2)
#define BIND_M (3)
#define BIND_H (4)
#define BIND_BC (5)
#define BIND_BO (6)
#define BIND_PO (7)
#define BIND_BS (8)
#define BIND_PS (9)
#define BIND_BR (10)
#define BIND_BW (11)
#define BIND_G (12)
#define BIND_SIDR (13)
#define BIND_SIDW (14)

#define BIN_SIZE (32)

#define LOG2_BIN_SIZE (5)
#define BIN_MASK (BIN_SIZE - 1u)

struct Bin
{
	mat2[BIN_SIZE] F;
	vec2[BIN_SIZE] position;
	float[BIN_SIZE] mass;
	uint[BIN_SIZE] blockParticleIndex;
	uint particleCount;
};

struct Particle
{
	vec4 color;
	vec2 position;
};

layout(binding = BIND_UBO) uniform ParameterUBO
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

uint blockIndex(uint x, uint y)
{
	return x + y * ubo.blockDimensions;
}

uint part1by1(uint x)
{
	x = (x | (x << 8)) & 0x00FF00FF;
	x = (x | (x << 4)) & 0x0F0F0F0F;
	x = (x | (x << 2)) & 0x33333333;
	x = (x | (x << 1)) & 0x55555555;
	return x;
}

uint partialMorton(uint x, uint y)
{
	uint perserve = (((x & 0x3) << 2) | (y & 0x3));
	
	return (((part1by1(x) << 1) | part1by1(y)) & 0xFFFFFFF0) | perserve;
}