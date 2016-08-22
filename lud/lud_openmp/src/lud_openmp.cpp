// =================================================================================================
// This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
// CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
//
// Author(s):
//   Rafal Gandecki <rafal.gandecki@pwr.edu.pl>
//
// This example demonstrates the use of OpenMP for LU decomposition (Doolittle algorithm) and
// compares execution time.
// The example takes a single input argument, specifying the size of the matrices.
//
// See [http://www.openmp.org/] for the full OpenMP documentation.
//
// =================================================================================================

#include <omp.h>
#include <random>
#include <iostream>

void fill_random(float *A, const int &n, const int &m)
{
  std::mt19937 e(static_cast<unsigned int>(time(nullptr)));
  std::uniform_real_distribution<float> f;
  for(int i=0; i<n; ++i)
  {
    for(int j=0; j<m; ++j)
    {
      A[i*m+j] = f(e);
    }
  }
}


void lud(float *A, float *L, float *U, const int &n)
{
  for(int i=0;  i<n; i++)
  {
    for(int j=0; j<n; j++)
    {
      if(j>i)
        U[j*n+i] = 0;
      U[i*n+j] = A[i*n+j];
      for(int k=0; k<i; k++)
      {
        U[i*n+j] -= U[k*n+j] * L[i*n+k];
      }
    }
    for(int j=0; j<n; j++)
    {
      if(i>j)
        L[j*n+i] = 0;
      else if (j==i)
        L[j*n+i] = 1;
      else
      {
        L[j*n+i] = A[j*n+i] / U[i*n+i];
        for(int k=0; k<i; k++)
        {
          L[j*n+i] -= ((U[k*n+i] * L[j*n+k]) / U[i*n+i]);
        }
      }
    }
  }
}

void lud_OpenMP(float *A, float *L, float *U, const int &n)
{
  int i, j, k;
  #pragma omp parallel for shared(A, L, U, n) private(i, j, k)
  for (i=0; i<n; i++)
  {
    for(j=0; j<n; j++)
    {
      if(j>i)
        U[j*n+i] = 0;
      U[i*n+j] = A[i*n+j];
      for(k=0; k<i; k++)
      {
        U[i*n+j] -= U[k*n+j] * L[i*n+k];
      }
     }
    for(j=0; j<n; j++)
    {
      if(i>j)
        L[j*n+i] = 0;
      else if (j==i)
        L[j*n+i] = 1;
      else
      {
        L[j*n+i] = A[j*n+i] / U[i*n+i];
        for(k=0; k<i; k++)
        {
          L[j*n+i] -= ((U[k*n+i] * L[j*n+k]) / U[i*n+i]);
        }
      }
    }
  }
}

int main(int argc, char **argv)
{
  int n;
  float *A, *L, *U;
  if (argc != 2)
  {
    std::cout << "Usage: 1 argument: matrix size" << std::endl;
    return 1;
  }
  else
  {
    n = atoi(argv[1]);
  }

  A = new float[n*n];
  L = new float[n*n];
  U = new float[n*n];

  fill_random(A, n, n);

  double dtime;
  dtime = omp_get_wtime();
  lud(A, L, U, n);
  dtime = omp_get_wtime() - dtime;
  std::cout << "Time without OpenMP: " << dtime << std::endl;

  dtime = omp_get_wtime();
  lud_OpenMP(A, L, U, n);
  dtime = omp_get_wtime() - dtime;
  std::cout << "Time with OpenMP: " << dtime << std::endl;

  delete[] A;
  delete[] L;
  delete[] U;
  return 0;
}
