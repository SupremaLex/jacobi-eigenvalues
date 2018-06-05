#include "jacobi.h"
#include "mpi/mpi.h"
#include <unistd.h>
#include <cstdio>
#include <cmath>

using namespace std;

struct Ind2D {
  Ind2D(int _i=0, int _j=0) {
    k = _i;
    j = _j;
  }
  int k;
  int j;
};

Ind2D find_abs_max(Matrix m, const int n) {
  float max = m[1][0];
  Ind2D max_ind(1, 0);
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      if (j != k && abs(m[j][k]) > abs(max)) {
        max = m[j][k];
        max_ind.j = j;
        max_ind.k = k;
      }
    }
  }
  return max_ind;
}

float NormMatrix(Matrix m, const int n) {
  float norm = 0;
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < n; ++k) {
      if (j != k) {
        norm += m[j][k] * m[j][k];
      }
    }
  }
  return std::sqrt(norm);
}

void PrintMatrix(Matrix m, const int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      printf("%2.2f ", m[i][j]);
    }
    printf("\n");
  }
}

int** get_even_and_odd_ranks(int world_ranksize) {
  int** ranks = new int*[2];
  ranks[0] = new int[world_ranksize / 2];
  ranks[1] = new int[world_ranksize / 2];
  for (int i = 1; i < world_ranksize; ++i) {
      if (i % 2 == 1) {
        ranks[0][(i + 1) / 2 - 1] = i;

      } else {
        ranks[1][(i / 2) - 1] = i;
      }
  }
  return ranks;
}

void SerialJacobiRotate(Matrix m, const int j, const int k, const int n) {
  float c, s;
  if (m[j][j] == m[k][k]) {
    c = cos(M_PI / 4);
    s = sin(M_PI / 4);
  }
  else {
    float tau = (m[j][j] - m[k][k]) / (2 * m[j][k]);
    float t = ((tau > 0) ? 1 : -1) / (abs(tau) + sqrt(1 + tau * tau));
    c = 1 / sqrt(1 + t * t);
    s = c * t;
  }
  float tmp_jk = m[j][k];
  float tmp_jj = m[j][j];
  m[j][k] = (c * c - s * s) * tmp_jk + s * c * (m[k][k] - m[j][j]);
  m[k][j] = m[j][k];
  m[j][j] = c * c * tmp_jj + 2 * s * c * tmp_jk + s * s * m[k][k];
  m[k][k] = s * s * tmp_jj - 2 * s * c * tmp_jk + c * c * m[k][k];
  float tmp_jl;
  for (int l = 0; l < n; ++l) {
    if (l != j && l != k) {
      tmp_jl = m[j][l];
      m[j][l] = c * tmp_jl + s * m[k][l];
      m[k][l] = s * tmp_jl - c * m[k][l];
      m[l][j] = m[j][l];
      m[l][k] = m[k][l];
    }
  }
}

void ParallelJacobiRotate(Matrix m, int ind_j, int ind_k, const int n) {
  /*
   * Основные вычисления параллельного алгоритма Якоби для вычисление собственных чисел симметричной матрицы
   * Минимальное количество процессов для данной реализации - 3:
   *  0 процесс производит вычисление значений матрицы поворота и аккумулирует в себе данные из других процессов
   *  1 и 2 процессы вычисляют новые диагональные элементы
   *  3 и 4 процессы вычисляют новые значения строк j и k
   *  Остальные процессы также вычисляют новые значения строк j и k
   * */
  int rank, world_ranksize;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_ranksize);
  float c, s;
  int j, k;
  float* row_j = new float[n];
  float* row_k = new float[n];
  if (rank == 0) {
    j = ind_j;
    k = ind_k;
    //  Вычисление углов матрицы поворота
    if (m[j][j] == m[k][k]) {
      c = cos(M_PI / 4);
      s = sin(M_PI / 4);
    }
    else {
      float tau = (m[j][j] - m[k][k]) / (2 * m[j][k]);
      float t = ((tau > 0) ? 1 : -1) / (abs(tau) + sqrt(1 + tau * tau));
      c = 1 / sqrt(1 + t * t);
      s = c * t;
    }
    //  Копируем j и k строки в соотвествующие буфферы, чтобы переслать их другим потокам
    for (int i = 0; i < n; ++i) {
      row_j[i] = m[j][i];
      row_k[i] = m[k][i];
    }

  }
  // Рассылаем данные по всем потокам
  MPI_Bcast(&j, 1, MPI_INT, 0, MPI_COMM_WORLD);  //  номер строки j
  MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);  //  номер строки k
  MPI_Bcast(&c, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);  // cos(theta)
  MPI_Bcast(&s, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);  // sin(theta)
  MPI_Bcast(row_j, n, MPI_FLOAT, 0, MPI_COMM_WORLD);  // строка j
  MPI_Bcast(row_k, n, MPI_FLOAT, 0, MPI_COMM_WORLD);  // строка k
  //MPI_Barrier(MPI_COMM_WORLD);
  float m_jj, m_kk;
  /*if (rank == 1) {
    //  Процесс ! пересчитывает jj диагональный элемент
    m_jj = c * c * row_j[j] + 2 * s * c * row_j[k] + s * s * row_k[k];
    MPI_Send(&m_jj, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  } else if (rank == 2) {
    //  Процесс ! пересчитывает kk диагональный элемент
    m_kk = s * s * row_j[j] - 2 * s * c * row_j[k] + c * c * row_k[k];
    MPI_Send(&m_kk, 1, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
  }*/
  //  Создаем две группы процессов: одну для пересчета строки j, другую для пересчета строки k
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  // Группа строки k состоит из нечетных процессов ( >= 3)
  // Группа строки j состоит из четных процессов ( >= 4)
  int** odd_even_ranks = get_even_and_odd_ranks(world_ranksize); // получаем номера потоков
  MPI_Group row_j_group;
  MPI_Group row_k_group;
  const int group_k_size = (world_ranksize - 1) / 2;
  const int group_j_size = group_k_size;
  MPI_Group_incl(world_group, group_j_size, odd_even_ranks[0], &row_j_group);
  MPI_Group_incl(world_group, group_k_size, odd_even_ranks[1], &row_k_group);
  // Создаем на основе групп коммуникаторы
  MPI_Comm row_j_comm;
  MPI_Comm row_k_comm;
  MPI_Comm_create_group(MPI_COMM_WORLD, row_j_group, 0, &row_j_comm);
  MPI_Comm_create_group(MPI_COMM_WORLD, row_k_group, 0, &row_k_comm);

  int row_j_rank = -1;
  int row_j_size = -1;
  float* row_j_new = new float[n];
  //  проверяем существование коммуникатора
  if (MPI_COMM_NULL != row_j_comm) {
    MPI_Comm_rank(row_j_comm, &row_j_rank);
    MPI_Comm_size(row_j_comm, &row_j_size);
    //  часть строки, которую пересчитывает один поток из группы row_j_comm
    int size = n / row_j_size;
    //  Выделяем память под буфферы
    float* row_j_part = new float[size];
    float* row_k_part = new float[size];
    float* row_j_new_part = new float[size];
    //  Разбиваем k и j строки между процессами группы row_j_comm
    MPI_Scatter(row_j, size, MPI_FLOAT, row_j_part, size, MPI_FLOAT, 0, row_j_comm);
    MPI_Scatter(row_k, size, MPI_FLOAT, row_k_part, size, MPI_FLOAT, 0, row_j_comm);
    //  Пересчитываем часть строки
    for (int i = 0; i < size; ++i) {
        row_j_new_part[i] = c * row_j_part[i] + s * row_k_part[i];
    }
    //  Собираем новую строку из частей в 0 процессе по отношению к коммуникатору row_j_comm
    //  (3 - по отношению MPI_COMM_WORLD)
    MPI_Gather(row_j_new_part, size, MPI_FLOAT, row_j_new, size, MPI_FLOAT, 0, row_j_comm);
    if (row_j_rank == 0) {
      //  Пересылаем новую строку в 0 процесс (по отношению к MPI_COMM_WORLD)
      MPI_Send(row_j_new, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    // Освобождаем память, группу и коммуникатор
    delete[] row_j_new_part;
    delete[] row_k_part;
    delete[] row_j_part;
    MPI_Group_free(&row_j_group);
    MPI_Comm_free(&row_j_comm);
  }

  int row_k_rank = -1;
  int row_k_size = -1;
  float* row_k_new = new float[n];
  if (MPI_COMM_NULL != row_k_comm) {
    MPI_Comm_rank(row_k_comm, &row_k_rank);
    MPI_Comm_size(row_k_comm, &row_k_size);
    int size = n / row_k_size;
    float* row_j_part = new float[size];
    float* row_k_part = new float[size];
    float* row_k_new_part = new float[size];
    MPI_Scatter(row_j, size, MPI_FLOAT, row_j_part, size, MPI_FLOAT, 0, row_k_comm);
    MPI_Scatter(row_k, size, MPI_FLOAT, row_k_part, size, MPI_FLOAT, 0, row_k_comm);
    for (int i = 0; i < size; ++i) {
        row_k_new_part[i] = s * row_j_part[i] - c * row_k_part[i];
    }
    MPI_Gather(row_k_new_part, size, MPI_FLOAT, row_k_new, size, MPI_FLOAT, 0, row_k_comm);
    if (row_k_rank == 0) {
      MPI_Send(row_k_new, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
    delete[] row_k_new_part;
    delete[] row_k_part;
    delete[] row_j_part;
    MPI_Group_free(&row_k_group);
    MPI_Comm_free(&row_k_comm);
  }
  //  Модифицируем матрицу в главное процессе
  //MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    //  Получаем данные из других потоков
    //  диагональный элемент jj
    //MPI_Recv(&m_jj, 1, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //  диагональный элемент kk
    //MPI_Recv(&m_kk, 1, MPI_FLOAT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //  пересчитанная j строка
    MPI_Recv(row_j_new, n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //  пересчитанная k строка
    MPI_Recv(row_k_new, n, MPI_FLOAT, 2, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    // Заменяем значения оригинальной матрицы
    m[j][k] = (c * c - s * s) * row_j[k] + s * c * (row_k[k] - row_j[j]);
    m[k][j] = m[j][k];
    //m[j][j] = m_jj;
    //m[k][k] = m_kk;
    m[j][j] = c * c * row_j[j] + 2 * s * c * row_j[k] + s * s * row_k[k];
    m[k][k] = s * s * row_j[j] - 2 * s * c * row_j[k] + c * c * row_k[k];;
    for (int i = 0; i < n; ++i) {
      if (i != j && i != k) {
        m[j][i] = row_j_new[i];
        m[k][i] = row_k_new[i];
        m[i][j] = m[j][i];
        m[i][k] = m[k][i];
      }
    }
  }
  //  освобождаем память
  delete[] row_k_new;
  delete[] row_j_new;
  delete[] odd_even_ranks[1];
  delete[] odd_even_ranks[0];
  delete[] odd_even_ranks;
  delete[] row_k;
  delete[] row_j;
}

void SerialJacobi(Matrix mat, const int n, const float eps) {
  Ind2D ind_max;
  ind_max = find_abs_max(mat, n);
  float norm = NormMatrix(mat, n);
  float tol = eps * norm;
  printf("eps = %f, norm = %f, tol = %f\n",eps, norm, tol);
  while (NormMatrix(mat, n) > tol) {
    //printf("%f ", norm);
    SerialJacobiRotate(mat, ind_max.j, ind_max.k, n);
    //PrintMatrix(mat, n);
    ind_max = find_abs_max(mat, n);
  }
}

void ParallelJacobi(Matrix mat, const int n, const float eps) {
  Ind2D ind_max;
  float elapsed_time = 0;
  ind_max = find_abs_max(mat, n);
  float norm = NormMatrix(mat, n);;
  float tol;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    norm = NormMatrix(mat, n);
    tol = eps * norm;
    printf("eps = %f, norm = %f, tol = %f\n",eps, norm, tol);
  }
  MPI_Bcast(&norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&tol, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
  while (norm > tol) {
    elapsed_time -= MPI_Wtime();
    ParallelJacobiRotate(mat, ind_max.j, ind_max.k, n);
    if (rank == 0) {
      norm = NormMatrix(mat, n);
      //printf("\nnorm = %f\n", norm);
    }
    elapsed_time += MPI_Wtime();
    MPI_Bcast(&norm, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    ind_max = find_abs_max(mat, n);
  }
  /*if (rank == 0) {
    for (int i = 0; i < n; ++i) {
      printf("%f ", mat[i][i]);
    }
  }*/
}
