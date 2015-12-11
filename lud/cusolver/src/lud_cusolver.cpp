#include <cstdio>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include "../../../common/helper.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <math.h>

int nIters = 2;

int main(int argc, char **argv) {

	int N;
	if (argc !=2)  {
		printf("Usage: ./1_dense_lud_cusolver_mkl <matrix_size> \n");
		return 1;
	} else {
		N = atoi(argv[1]);
    	}

	// --- Matrices to be inverted (only one in this example)
	float *h_A = new float[N*N];
	// --- Matrix to store the randomly initialized data that will be used throughout the passes
	float *init_A = new float[N*N];
	// --- Randomly initialize input data
	for (int i = 0 ; i < N*N; i++) {
		init_A[i] = (float)rand() / RAND_MAX;
	}
	// --- Copy input data to host buffer
	for (int i = 0 ; i < N*N; i++) {
		h_A[i] = init_A[i];
	}


	// --- Allocate device buffers 
	float *d_A;	
	cudaMalloc((void**)&d_A, N*N*sizeof(float));

	// --- Move the matrix to be computed (LU) from host to device
	cudaMemcpy(d_A,h_A,N*N*sizeof(float),cudaMemcpyHostToDevice);

	// --- Creating the array of pointers needed as input to the batched getrf

	int *d_PivotArray; cudaMalloc((void**)&d_PivotArray, N*sizeof(int));
	int *d_InfoArray;  cudaMalloc((void**)&d_InfoArray,  sizeof(int));
	
	int *h_PivotArray = (int *)malloc(N*sizeof(int));
	int *h_InfoArray  = (int *)malloc(sizeof(int));


	// --- Creating cuSOLVER handle
	cusolverDnHandle_t handle_cusolver;
	cusolverDnCreate(&handle_cusolver);
	int size;

	// --- Computing cuSOLVER buffer size
	cusolverDnSgetrf_bufferSize(handle_cusolver, N, N, d_A, N, &size);

	// --- Allocate buffer for cuSOLVER scratchpad
	float *Lwork; cudaMalloc((void**)&Lwork,size*sizeof(float));

	// --- cuSOLVER warm-up phase
	cusolverDnSgetrf(handle_cusolver, N, N, d_A, N, Lwork, d_PivotArray, d_InfoArray);
#ifdef _OPENMP
	double cusolver_time;
	cusolver_time = omp_get_wtime();
#else
	uint64 cusolver_time;
	cusolver_time = get_time_uint64();
#endif
	for (int kk = 0; kk < nIters; kk++) {
		cudaMemcpy(d_A,init_A,N*N*sizeof(float),cudaMemcpyHostToDevice);
		cusolverDnSgetrf(handle_cusolver, N, N, d_A, N, Lwork, d_PivotArray, d_InfoArray);
	}
#ifdef _OPENMP
	cusolver_time = omp_get_wtime() - cusolver_time;
#else
	cusolver_time = get_time_uint64() - cusolver_time;
#endif

	// --- copying cuSOLVER status to host
	cudaMemcpy(h_InfoArray,d_InfoArray,sizeof(int),cudaMemcpyDeviceToHost);

	for (int i = 0; i < 1; i++)
		if (h_InfoArray[i]  != 0) {
			fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

	cudaMemcpy(h_A,d_A,N*N*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_PivotArray,d_PivotArray,N*sizeof(int),cudaMemcpyDeviceToHost);

	// --- Print the results
#ifdef _OPENMP
	printf("cuSOLVER time for computing LU of matrix size %d is %fs \n",N,cusolver_time);
	printf("cuSOLVER achieved %f GFLOPS \n",((2.f/3.f)*N*N*N*nIters)/(cusolver_time*1024*1024*1024));
#else
	printf("cuSOLVER time for computing LU of matrix size %d is %fms \n",N,cusolver_time);
	printf("cuSOLVER achieved %f GFLOPS \n",((2.f/3.f)*N*N*N*nIters*1000)/(cusolver_time*1024*1024*1024));
#endif

	
	// --- Free buffers
	delete(h_A);
	delete(init_A);
	cudaFree(d_A);
	cudaFree(d_InfoArray);
	cudaFree(d_PivotArray);
	cudaFree(Lwork);
	free(h_InfoArray);
	free(h_PivotArray);
}
