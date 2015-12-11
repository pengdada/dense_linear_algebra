#include <cstdio>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include "../../../common/helper.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>
#include <mkl.h>

int nIters = 2;

int main(int argc, char **argv) {

	int N;

	if (argc !=2)  {
		printf("Usage: ./1_dense_lud_mkl <matrix_size> \n");
		return 1;
	} else {
		N = atoi(argv[1]);
    	}
	// --- Matrix to store the randomly initialized data that will be used throughout the passes
	float *init_A = new float[N*N];
	// --- Randomly initialize input data
	for (int i = 0 ; i < N*N; i++) {
		init_A[i] = (float)rand() / RAND_MAX;
	}

	// --- Initializing buffer for MKL version
	int info;
	float *mkl_A = new float[N*N];
	for (int i = 0 ; i < N*N; i++) {
		mkl_A[i] = init_A[i];
	}

	int ipiv[N];

	// --- MKL version warm-up
	sgetrf(&N, &N, mkl_A, &N, ipiv, &info);

#ifdef _OPENMP
	double mkl_time;
	mkl_time = omp_get_wtime();
#else
	uint64 mkl_time;
	mkl_time = get_time_uint64();
#endif

	// --- Time MKL version
	for (int kk = 0; kk < nIters; kk++) {
		for (int i = 0; i< N*N; i++)
			mkl_A[i] = init_A[i];
		sgetrf(&N, &N, mkl_A, &N, ipiv, &info);
	}

#ifdef _OPENMP
	mkl_time = omp_get_wtime() - mkl_time;
#else
	mkl_time = get_time_uint64() - mkl_time;
#endif

#ifdef _OPENMP
	printf("MKL time for computing LU of matrix size %d is %fs \n",N,mkl_time);
	printf("MKL achieved %f GFLOPS \n",((2.f/3.f)*N*N*N*nIters)/(mkl_time*1024*1024*1024));
#else
	printf("MKL time for computing LU of matrix size %d is %llums \n",N,mkl_time);
	printf("MKL achieved %f GFLOPS \n",((2.f/3.f)*N*N*N*nIters*1000)/(mkl_time*1024*1024*1024));
#endif

	
	// --- Free buffers
	delete(mkl_A);
	delete(init_A);
}
