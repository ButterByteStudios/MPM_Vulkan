#define BIND_VR (0)
#define BIND_VW (1)
#define BIND_M (2)
#define BIND_H (3)
#define BIND_BC (4)
#define BIND_BO (5)
#define BIND_BS (6)
#define BIND_SID (7)
#define BIND_MAT (8)

#define BIND_PUBO (0)
#define BIND_TUBO (1)

#define BIND_BR (0)
#define BIND_BW (1)


#define BIN_SIZE (64)

#define LOG2_BIN_SIZE (6)
#define BIN_MASK (BIN_SIZE - 1u)


#define NEO_HOOKEAN (0)
#define NACC (1)
#define NADP (2)


#define G (4)
#define LOG2_G (2)
#define G_MASK (G - 1)


#define I_RHO (0)

#define I_LAMBDA (1)
#define I_K (1)

#define I_MU (2)

#define I_BETA (3)
#define I_CF (3)

#define I_M (4)
#define I_CC (4)

#define I_XI (5)


#define CIRCLE (0)
#define SQUARE (1)

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

struct Particle
{
	vec4 color;
	vec2 position;
};

struct Material
{
	float[6] params;
	int materialType;
};

layout(set = 1, binding = BIND_PUBO) uniform ParameterUBO
{
	vec2 speed;
	float dx;
	float invDx;
	uint dimensions;
	uint blockDimensions;
	uint blocks;
	float dt;
	float invDt;
} pubo;

uint blockIndex(uint x, uint y)
{
	return x + y * pubo.blockDimensions;
}

uint part1by1(uint x)
{
	x = (x | (x << 8)) & 0x00FF00FF;
	x = (x | (x << 4)) & 0x0F0F0F0F;
	x = (x | (x << 2)) & 0x33333333;
	x = (x | (x << 1)) & 0x55555555;
	return x;
}

uint morton(uint x, uint y)
{
	return (part1by1(x) << 1) | part1by1(y);
}

uint partialMorton(uint x, uint y)
{
	uint perserve = (((x & 0x3)) | ((y & 0x3) << 2));
	
	return (((part1by1(x)) | (part1by1(y) << 1)) & 0xFFFFFFF0) | perserve;
}

uint lossyMorton(uint x, uint y)
{
	return (((part1by1(x)) | (part1by1(y) << 1)) & 0xFFFFFFF0);
}


vec2 hash22(uvec2 q)
{
	q *= uvec2(1597334673U, 3812015801U);
	q = (q.x ^ q.y) * uvec2(1597334673U, 3812015801U);
	return vec2(q) * 2.328306437080797e-10;
}

vec3 hash32(uvec2 q)
{
	uvec3 n = q.xyx * uvec3(1597334673U, 3812015801U, 2798796415U);
	n = (n.x ^ n.y ^n.z) * uvec3(1597334673U, 3812015801U, 2798796415U);
	return vec3(n) * 2.328306437080797e-10;
}


vec2 circleDistribution(vec2 random, float radius)
{
    const float pi2 = 6.2831854820251465;
    float r = radius * sqrt(random.x);
	float angle = random.y * pi2;
    return vec2(cos(angle), sin(angle)) * r;
}
