// https://scicomp.stackexchange.com/a/18766
void svd(mat2 A, out mat2 U, out vec2 s, out mat2 V) 
{
    s[0] = 0.5 * (length(vec2(A[0][0] - A[1][1], A[1][0] + A[0][1])) + length(vec2(A[0][0] + A[1][1], A[1][0] - A[0][1])));
    s[1] = abs(determinant(A)) / s[0];
    V[0][1] = (s[0] > s[1]) ? sin((atan(2 * (A[0][0] * A[1][0] + A[0][1] * A[1][1]), A[0][0] * A[0][0] - A[1][0] * A[1][0] + A[0][1] * A[0][1] - A[1][1] * A[1][1])) / 2) : 0;
    V[0][0] = sqrt(1 - V[0][1] * V[0][1]);
    V[1][0] = -V[0][1];
    V[1][1] = V[0][0];
    U[0][0] = (s[0] != 0) ? (A[0][0] * V[0][0] + A[1][0] * V[0][1]) / s[0] : 1;
    U[0][1] = (s[0] != 0) ? (A[0][1] * V[0][0] + A[1][1] * V[0][1]) / s[0] : 0;
    U[1][0] = (s[1] != 0) ? (A[0][0] * V[1][0] + A[1][0] * V[1][1]) / s[1] : -U[0][1];
    U[1][1] = (s[1] != 0) ? (A[0][1] * V[1][0] + A[1][1] * V[1][1]) / s[1] : U[0][0];
}

mat2 neohookeanKirchoff(mat2 F, float lambda, float mu)
{
	float J = determinant(F);
	mat2 F_T = transpose(F);
	return mu * (F * F_T - mat2(1.0)) + (lambda * log(J)) * mat2(1.0);	
}


// Model uses neo-hookean stress. Can be changed to 
void naccProject(
    inout mat2 Fe_trial,
    float k,
    float mu,
    float beta,
    float M,
    float xi,
    inout float alpha,
    out mat2 tau
)
{
    mat2 I = mat2(1.0);

	mat2 U, V;
    vec2 sigma;

    svd(Fe_trial, U, sigma, V);
	float p0 = k * (0.00001 + sinh(xi * max(-alpha, 0)));

	// Calculating trial values
	float Je_tr = sigma[0] * sigma[1];
	vec2 sigmaSqr = sigma * sigma;
	float sigmaSqr_trace = (sigmaSqr[0] + sigmaSqr[1]) * 0.5;

	vec2 s_tr = (mu / Je_tr) * (sigmaSqr - sigmaSqr_trace);
	float p_tr = -0.5 * k * (Je_tr * Je_tr - 1.0);

    // Case 1: Project to max tip of the YS
    if (p_tr > p0)
    {
        float Je_new = sqrt(1.0 - 2.0 * p0 / k);

        alpha += log(Je_tr / Je_new);

        float sigma_components = sqrt(Je_new);

        Fe_trial = U * sigma_components * transpose(V);

        tau = -p0 * I; // p_tr gets projected to p0 and the deviatoric part of the matrix sigma is a null matrix

        return;
    }

    // Case 2: Project to min tip of the YS
    if (p_tr < -beta * p0)
    {
        float Je_new = sqrt(1.0 + 2.0 * beta * p0 / k);

        alpha += log(Je_tr / Je_new);

        float sigma_components = sqrt(Je_new);

        Fe_trial = U * sigma_components * transpose(V);

        tau = beta * p0 * I;

        return;
    }

    // Yield test before case 3
	float sqr_s_norm = dot(s_tr, s_tr);
	float s_norm = sqrt(sqr_s_norm);

    float yield =
        (1.0 + 2.0 * beta) * 2.0 * sqr_s_norm + M * M * (p_tr + beta * p0) * (p_tr - p0);

    if (yield < 0.0001) // Inside ys
    {
        tau = U * mat2(s_tr[0] - p_tr, 0.0, 0.0, s_tr[1] - p_tr) * transpose(U);

        return;
    }

    // Case 3: Hardening routine
    if (p0 > 0.0001 &&
        p_tr < p0 - 0.0001 &&
        p_tr > -beta * p0 + 0.0001)
    {
        float pc = 0.5 * (1.0 - beta) * p0;
		float q_tr = sqrt(2.0) * s_norm;

        float dx = pc - p_tr;
        float dy = -q_tr;

        float len = sqrt(dx * dx + dy * dy);

        float Dp = dx / len;
        float Dq = dy / len;

        float A = M * M * Dp * Dp + (1.0 + 2.0 * beta) * Dq * Dq;

        float B = M * M * Dp * (2.0 * pc - p0 + beta * p0);

        float C = M * M * (pc + beta * p0) * (pc - p0);

        float root_discriminant = sqrt(B * B - 4.0 * A * C);

        float l1 = (-B + root_discriminant) / (2.0 * A);
        float l2 = (-B - root_discriminant) / (2.0 * A);

        float p1 = pc + l1 * Dp;
        float p2 = pc + l2 * Dp;

        float px = ((p_tr - pc) * (p1 - pc) > 0.0) ? p1 : p2;

        float Je_cross = sqrt(1.0 - 2.0 * px / k);

        if (Je_cross > 0.0001)
        {
            alpha += log(Je_tr / Je_cross);
        }
    }

    // Case 3: Yield Surface projection
    vec2 be_new = sqrt((-M * M * (p_tr + beta * p0) * (p_tr - p0)) / (2 + 4 * beta)) * (Je_tr / mu) * (s_tr / s_norm) + sigmaSqr_trace;

	mat2 sigma_new = mat2(sqrt(be_new[0]), 0, 0, sqrt(be_new[1]));
	// Reconstruct elastic deformation gradient
	Fe_trial = U * sigma_new * transpose(V);

	float be_trace = 0.5 * (be_new[0] + be_new[1]);
	vec2 s_new = (mu / Je_tr) * (be_new - be_trace);
	tau = U * mat2(s_new[0] - p_tr, 0.0, 0.0, s_new[1] - p_tr) * transpose(U);
}