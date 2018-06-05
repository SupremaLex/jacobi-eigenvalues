#include "mpi.h"
#include "stdio.h"
#include "time.h"
#include <cstdlib>
#include <cstring>
#include "jacobi.h"

Matrix SquareMatrix(const int n){
  Matrix matrix = new float*[n];
  for (int i = 0; i < n; i++) {
    matrix[i] = new float[n];
  }
  srandom(time(0)+clock()+random());
  for(int i = 0; i < n; i++){
    for(int j = 0; j < n; j++){
      matrix[i][j] = i + j + 1;//rand() % 4 + 1;
      matrix[j][i] = matrix[i][j];
    }
  }
  return matrix;
}

int main(int argc, char** argv){
  //  размерность передаем через аргументы командной строки
  int n = std::atoi(argv[1]);
  float start, end;
  start = 0;
  end = 0;
  double elapsed_time = 0;

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  elapsed_time -= MPI_Wtime();
  Matrix m = SquareMatrix(n);
    PrintMatrix(m, n);
    printf("\n");
  ParallelJacobi(m, n, 1e-5);
  elapsed_time += MPI_Wtime();
  if (rank == 0) {
    for (int i = 0; i < n; ++i) {
      printf("%f ", m[i][i]);
    }
    printf("Dimension %i, time elapsed %f\n", n, elapsed_time);
    MPI_Abort(MPI_COMM_WORLD, MPI_SUCCESS);
  }
  MPI_Finalize();
  for (int i = 0; i < n; ++i) {
    delete[] m[i];
  }
  delete[] m;
  return 0;
}

/* Тестирование последовательного алгоритма Якоби*/
/*int main(int argc, char** argv){
  int dimesions[8] = {8, 16, 32, 64, 128, 256, 512, 1024};
  clock_t t;
  double elapsed_secs;
  for (int i = 0; i < 8; ++i) {
    t = clock();
    SerialJacobi(SquareMatrix(dimesions[i]), dimesions[i], 1e-5);
    elapsed_secs = double(clock() - t) / CLOCKS_PER_SEC;
    printf("Dimension %i, elapsed time %f\n", dimesions[i], elapsed_secs);
  }
  return 0;
}*/
