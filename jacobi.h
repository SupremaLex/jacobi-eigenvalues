#ifndef JACOBI
#define JACOBI

typedef float** Matrix;

void SerialJacobi(Matrix mat, const int n, const float eps);

void ParallelJacobi(Matrix mat, const int n, const float eps);

void PrintMatrix(Matrix mat, const int i);
#endif // JACOBI

