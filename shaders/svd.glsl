vec2 SVD_AbsScaleOnly(mat2 matrixIn)
{
    vec4 vectorized = vec4(matrixIn[0], matrixIn[1]);
    float Q = dot(vectorized, vectorized);
    float R = determinant(matrixIn); //ad-bc
    float discriminant = sqrt(Q*Q-4*R*R);

    vec2 scale = sqrt(vec2(Q+discriminant, Q-discriminant) / 2);

    return scale;
}

mat2 SVD_EigenvectorGivenEigenvalue(mat2 matrixIn, vec2 eigenvalue)
{
    eigenvalue = eigenvalue * eigenvalue;

    float M01 = dot(matrixIn[0], matrixIn[1]);
    float M00 = dot(matrixIn[0], matrixIn[0]);

    vec2 eigenVectorNumerator = vec2(M01, M01);
    vec2 eigenVectorDenominator = eigenvalue - vec2(M00, M00);

    vec2 eigenFraction = eigenVectorNumerator / eigenVectorDenominator;

    mat2 eigenMatrix = {
    normalize(vec2(eigenFraction.x, 1)),
    normalize(vec2(eigenFraction.y, 1)),
    };

    eigenMatrix = transpose(eigenMatrix);

    return eigenMatrix;
}

void SVD_PartialDecompose(mat2 matrixIn, out mat2 U, out vec2 scale)
{
    scale = SVD_AbsScaleOnly(matrixIn);

    U = SVD_EigenvectorGivenEigenvalue(matrixIn, scale);
}

void SVD_FullDecompose(mat2 matrixIn, out mat2 U, out vec2 scale, out mat2 V)
{
    SVD_PartialDecompose(matrixIn, U, scale);

    // A = U * E * V
    // U^T * A = E * V
    // E^-1 * U^T * A = V

    vec2 inverseScale = 1.0 / scale;

    mat2 E = mat2(
        inverseScale.x, 0,
        0, inverseScale.y
    );

    V = E * transpose(U) * matrixIn;
}


void symmetricEigenDecomposition(
    mat2 A,
    out mat2 R,
    out vec2 eigenvalues
) 
{
    float a = A[0][0];
    float b = 0.5 * (A[0][1] + A[1][0]);
    float d = A[1][1];

    // Eigenvalues
    float halfTrace = 0.5 * (a + d);
    float radius = sqrt(
        0.25 * (a - d) * (a - d) + b * b
    );

    eigenvalues = vec2(
        halfTrace + radius,
        halfTrace - radius
    );

    // Eigenvectors
    float theta = 0.5 * atan(2.0 * b, a - d);

    float c = cos(theta);
    float s = sin(theta);

    // GLSL matrices are column-major
    R = mat2(
        c, s,
        -s, c
    );
}