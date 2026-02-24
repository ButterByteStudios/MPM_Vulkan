#define BIND_VQR (0)
#define BIND_VQW (1)
#define BIND_MQ (2)
#define BIND_H (3)
#define BIND_BC (4)
#define BIND_BO (5)
#define BIND_BS (6)
#define BIND_BR (7)
#define BIND_BW (8)
#define BIND_SID (9)
#define BIND_GQR (10)
#define BIND_GQW (11)
#define BIND_VOLQ (12)
#define BIND_TQ (13)
#define BIND_VN (14)
#define BIND_MN (15)

#define BIND_UBO (0)
#define BIND_G (1)

#define BIN_SIZE (64)

#define LOG2_BIN_SIZE (6)
#define BIN_MASK (BIN_SIZE - 1u)

struct Bin
{
	mat2[BIN_SIZE] F;
	vec2[BIN_SIZE] position;
	float[BIN_SIZE] mass;
	uint[BIN_SIZE] blockParticleIndex;
	uint[BIN_SIZE] particleId;
	uint particleCount;
};

struct Particle
{
	vec4 color;
	vec2 position;
};

layout(set = 1, binding = BIND_UBO) uniform ParameterUBO
{
	vec2 speed;
	float k;
	float mu;
	float rho;
	float dx;
	float invDx;
	uint quadratureDimensions;
	uint nodeDimensions;
	uint blockDimensions;
	float dt;
	float invDt;
} ubo;

uint blockIndex(uint x, uint y)
{
	return x + y * ubo.blockDimensions;
}

uint nodeIndex(uint x, uint y)
{
	return x + y * ubo.nodeDimensions;
}

uvec2 toBlockCoords(vec2 a)
{
	return uvec2(a - 2) >> 2;
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