#include <cstdio>
#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
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
		printf("Usage: ./1_dense_lud_cublas_mkl <matrix_size> \n");
		return 1;
	} else {
		N = atoi(argv[1]);
    	}

	const unsigned int Nmatrices = 1;
	
	cublasHandle_t handle;
	cublasCreate(&handle);

	// --- Matrices to be inverted (only one in this example)
	float *h_A = new float[N*N*Nmatrices];
	// --- Matrix to store the randomly initialized data that will be used throughout the passes
	float *init_A = new float[N*N*Nmatrices];
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
	cudaMalloc((void**)&d_A, N*N*Nmatrices*sizeof(float));

	// --- Move the matrix to be computed (LU) from host to device
	cudaMemcpy(d_A,h_A,N*N*Nmatrices*sizeof(float),cudaMemcpyHostToDevice);

	// --- Creating the array of pointers needed as input to the batched getrf
	float **h_inout_pointers = (float **)malloc(Nmatrices*sizeof(float *));
	for (int i=0; i<Nmatrices; i++) h_inout_pointers[i]=(float *)((char*)d_A+i*((size_t)N*N)*sizeof(float));
 
	float **d_inout_pointers;
	cudaMalloc((void**)&d_inout_pointers, Nmatrices*sizeof(float *));
	cudaMemcpy(d_inout_pointers,h_inout_pointers,Nmatrices*sizeof(float *),cudaMemcpyHostToDevice);
	free(h_inout_pointers);

	int *d_PivotArray; cudaMalloc((void**)&d_PivotArray, N*Nmatrices*sizeof(int));
	int *d_InfoArray;  cudaMalloc((void**)&d_InfoArray,  Nmatrices*sizeof(int));
	
	int *h_PivotArray = (int *)malloc(N*Nmatrices*sizeof(int));
	int *h_InfoArray  = (int *)malloc(  Nmatrices*sizeof(int));



	// --- Warm-up phase for cuBLAS	
	cublasSgetrfBatched(handle, N, d_inout_pointers, N, d_PivotArray, d_InfoArray, Nmatrices);

	// --- Timing cuBLAS
#ifdef _OPENMP
	double cublas_time;
	cublas_time = omp_get_wtime();
#else
	uint64 cublas_time;
	cublas_time = get_time_uint64();
#endif
	for (int kk = 0; kk < nIters; kk++) {
		cudaMemcpy(d_A,init_A,N*N*Nmatrices*sizeof(float),cudaMemcpyHostToDevice);
		cublasSgetrfBatched(handle, N, d_inout_pointers, N, d_PivotArray, d_InfoArray, Nmatrices);
	}
#ifdef _OPENMP
	cublas_time = omp_get_wtime() - cublas_time;
#else
	cublas_time = get_time_uint64() - cublas_time;
#endif

	cudaMemcpy(h_InfoArray,d_InfoArray,Nmatrices*sizeof(int),cudaMemcpyDeviceToHost);

	for (int i = 0; i < Nmatrices; i++)
		if (h_InfoArray[i]  != 0) {
			fprintf(stderr, "Factorization of matrix %d Failed: Matrix may be singular\n", i);
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}

	cudaMemcpy(h_A,d_A,N*N*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_PivotArray,d_PivotArray,N*Nmatrices*sizeof(int),cudaMemcpyDeviceToHost);

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

	// --- Compare cuBLAS/cuSOLVER with MKL for corectness
	bool correct = compareReference(mkl_A, h_A, N*N, 1.0e-3f);

	// --- Print the results
	printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == correct) ? "PASS" : "FAIL");
#ifdef _OPENMP
	printf("MKL time for computing LU of matrix size %d is %fs \n",N,mkl_time);
	printf("cuBLAS time for computing LU of matrix size %d is %fs \n",N,cublas_time);
	printf("MKL achieved %f GFLOPS and cuBLAS achieved %f GFLOPS \n",((2.f/3.f)*N*N*N*nIters)/(mkl_time*1024*1024*1024),((2.f/3.f)*N*N*N*nIters)/(cublas_time*1024*1024*1024));
#else
	printf("MKL time for computing LU of matrix size %d is %llums \n",N,mkl_time);
	printf("cuBLAS time for computing LU of matrix size %d is %llums \n",N,cublas_time);
	printf("MKL achieved %f GFLOPS and cuBLAS achieved %f GFLOPS \n",((2.f/3.f)*N*N*N*nIters*1000)/(mkl_time*1024*1024*1024),((2.f/3.f)*N*N*N*nIters*1000)/(cublas_time*1024*1024*1024));
#endif
	
	// --- Free buffers
	delete(mkl_A);
	delete(h_A);
	delete(init_A);
	cudaFree(d_A);
	cudaFree(d_InfoArray);
	free(h_InfoArray);
	cudaFree(d_PivotArray);
	free(h_PivotArray);
	cudaFree(d_inout_pointers);
}
