// =================================================================================================
// This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
// CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
//
// Author(s):
//   Rafal Gandecki <rafal.gandecki@pwr.edu.pl>
//
// This example demonstrates the use of OpenMP for matrix-matrix multiplication and
// compares execution time of algorithms.
// The example is set-up to perform single precision matrix-matrix multiplication.
// The example takes a triple input arguments (matrix A rows, matrix A cols, matric B cols),
// specifying the size of the matrices.
// See [http://www.openmp.org/] for the full OpenMP documentation.
//
// =================================================================================================

#include <omp.h>
#include <random>
#include <iostream>
#include <ctime>

void fill_random(float *A, const int &n, const int &m)
{
  std::mt19937 e(static_cast<unsigned int>(std::time(nullptr)));
  std::uniform_real_distribution<float> f;
  for(int i=0; i<n; ++i)
  {
    for(int j=0; j<m; ++j)
    {
      A[i*m+j] = f(e);
    }
  }
}

void gemm(float *A, float *B, float *C, 
          const int &A_rows, const int &A_cols, const int &B_rows)
{
  for(int i=0; i<A_rows; i++)
  {
    for (int k=0; k<A_cols; k++) {
      for(int j=0; j<B_rows; j++) {
        C[i*B_rows + j] += A[i*A_cols+k] * B[k*B_rows+j];
      }
    }
  }
}

void gemm_OpenMP(float *A, float *B, float *C,
                 const int &A_rows, const int &A_cols, const int &B_rows)
{
  int i, j, k;
  #pragma omp parallel for shared(A, B, C, A_rows, A_cols, B_rows) private(i, j, k)
  for (i = 0; i < A_rows; i++)
  {
    for (k=0; k<A_cols; k++)
    {
      for (j = 0; j < B_rows; j++)
	  {
        C[i*B_rows + j] += A[i*A_cols+k] * B[k*B_rows+j];
      }
    }
  }
}

int main(int argc, char **argv)
{
  int A_rows, A_cols, B_rows, B_cols;
    
  if (argc != 4)
  {
    std::cout << "Usage: 3 arguments: matrix A rows, matrix A cols and matrix B cols"<< std::endl;
    return 1;
  }
  else
  {
    A_rows = atoi(argv[1]);
    A_cols = atoi(argv[2]);
    B_rows = atoi(argv[2]);
    B_cols = atoi(argv[3]);
  }

  double dtime;

  float *A = new float[A_rows*A_cols];
  float *B = new float[B_rows*B_cols];
  float *C = new float[A_rows*B_cols](); // value-init to zero

  fill_random(A, A_rows, A_cols);
  fill_random(B, B_rows, B_cols);

  dtime = omp_get_wtime();
  gemm_OpenMP(A, B, C, A_rows, A_cols, B_cols);
  dtime = omp_get_wtime() - dtime;
  std::cout << "Time with OpenMp: " << dtime << std::endl;

  dtime = omp_get_wtime();
  gemm(A,B,C, A_rows, A_cols, B_cols);
  dtime = omp_get_wtime() - dtime;
  std::cout << "Time without OpenMP: " << dtime << std::endl;

  delete[] A;
  delete[] B;
  delete[] C;
  return 0;
}

