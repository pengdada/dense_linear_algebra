
// =================================================================================================
// This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
// CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
//
// Author(s):
//   Mariusz Uchronski <mariusz.uchronski@pwr.edu.pl>
//
// This example demonstrates the use of C++ threads for matrix-matrix multiplication.
// The example is set-up to perform single precision matrix-matrix multiplication.
//
// See [http://www.cplusplus.com/reference/thread/thread/] for the full C++ threads documentation.
//
// =================================================================================================

#include <thread>
#include <random>
#include <iostream>
#include <vector>
#include <algorithm>

void fill_random(float *a, const int &n, const int &m)
{
  std::mt19937 e(static_cast<unsigned int>(time(nullptr)));
  std::uniform_real_distribution<float> f;
  for(int i=0; i<n; ++i)
  {
    for(int j=0; j<m; ++j)
    {
      a[i*m+j] = f(e);
    }
  }
}

void gemm_threads(const int &num_threads, const int &id,
                  const int &n, const int &m, const int &k,
                  float *a, float *b, float *c)
{
  const int part = n/num_threads;
  const int begin = id*part;
  const int end = (id+1)*part - 1;
  for(int i=begin; i<=end; ++i)
  {
    for(int j=0; j<k; ++j)
    {
      float sum = 0.0;
      for(int l=0; l<m; ++l)
      {
        sum = sum + a[i*m+l] * b[l*k+j];
      }
      c[i*k+j] = sum;
    }
  }
}

void print_data(float *a, const int &n, const int &m)
{
  for(int i=0; i<n; ++i)
  {
    for(int j=0; j<m; ++j)
    {
      std::cout << a[i*m+j] << " ";
    }
    std::cout << std::endl;
  }
}

int main(int argc, char *argv[])
{
  const int n = 200;
  const int m = 300;
  const int k = 400;
  const int num_threads = 4;
  std::vector<std::thread> threads;

  float *a = new float[n*m];
  float *b = new float[m*k];
  float *c = new float[n*k];

  fill_random(a, n, m);
  fill_random(b, m, k);

  for(int i=0; i<num_threads; ++i)
    threads.push_back(std::thread(gemm_threads, num_threads, i, n, m, k, a, b, c));
  std::for_each(threads.begin(), threads.end(), std::mem_fn(&std::thread::join));

  delete[] a;
  delete[] b;
  delete[] c;
  return 0;
}
