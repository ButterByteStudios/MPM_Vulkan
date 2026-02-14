#define BIND_UBO (0)
#define BIND_VR (1)
#define BIND_VW (2)
#define BIND_M (3)
#define BIND_H (4)
#define BIND_BC (5)
#define BIND_WC (6)
#define BIND_BO (7)
#define BIND_WO (8)
#define BIND_PO (9)
#define BIND_BS (10)
#define BIND_WS (11)
#define BIND_PS (12)
#define BIND_BR (13)
#define BIND_BW (14)
#define BIND_G (15)
#define BIND_WD (16)
#define BIND_SIDR (17)
#define BIND_SIDW (18)
#define BIND_GID (19)

#define BIN_SIZE (32)

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
} ubo;