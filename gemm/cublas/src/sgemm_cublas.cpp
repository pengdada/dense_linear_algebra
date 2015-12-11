
// =================================================================================================
// This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
// CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
//
// Author(s):
//   Valeriu Codreanu <valeriu.codreanu@surfsara.nl>
//
// This example demonstrates the use of NVIDIA's linear algebra library for CUDA: cuBLAS.
// The example is set-up to perform single precision matrix multiplication and compare the result
// with a naive CPU version for validation. The example takes a single input argument, specifying
// the size of the matrices.
//
// See [http://docs.nvidia.com/cuda/cublas] for the full cuBLAS documentation.
//
// =================================================================================================

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../../../common/helper.h"

void
matrixMulCPU(float *C, const float *A, const float *B, unsigned int hA, unsigned int wA, unsigned int wB)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (unsigned int i = 0; i < hA; ++i)
        for (unsigned int j = 0; j < wB; ++j)
        {
            float sum = 0;

            for (unsigned int k = 0; k < wA; ++k)
            {
                sum += A[i * wA + k] * B[k * wB + j];
            }

            C[i * wB + j] = (float)sum;
        }
}

void gpuCublasMmul(cublasHandle_t &handle,const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda=m,ldb=k,ldc=m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;
    // do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gpuFillRandom(float *A, int nr_rows_A, int nr_cols_A) {
    // create a pseudo-random number generator
    curandGenerator_t prng;
    curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);

    // set the seed for the random number generator using the system clock
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // fill the array with random numbers on the device
    curandGenerateUniform(prng, A, nr_rows_A * nr_cols_A);
}

int main(int argc, char **argv) {

    int matrix_size;

    if (argc !=2)  {
	printf("Usage: ./1_dense_cublas <matrix_size> \n");
	return 1;
    } else {
	matrix_size = atoi(argv[1]);
    }

    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = matrix_size;
    

    // allocate arrays on the CPU
    float *h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    float *h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    float *h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
    float *reference = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

    // allocate arrays on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A,nr_rows_A * nr_cols_A * sizeof(float));
    cudaMalloc(&d_B,nr_rows_B * nr_cols_B * sizeof(float));
    cudaMalloc(&d_C,nr_rows_C * nr_cols_C * sizeof(float));

    // fill the arrays A and B on GPU with random numbers
    gpuFillRandom(d_A, nr_rows_A, nr_cols_A);
    gpuFillRandom(d_B, nr_rows_B, nr_cols_B);

    // copy the random data back on CPU and to perform reference check
    cudaMemcpy(h_A,d_A,nr_rows_A * nr_cols_A * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B,d_B,nr_rows_B * nr_cols_B * sizeof(float),cudaMemcpyDeviceToHost);
    
    cublasHandle_t handle;
    cudaEvent_t start, stop;

    // create cuBlas handler
    cublasCreate(&handle);

    // create cuda timer events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // multiply A and B on GPU warm-up (not timed)
    gpuCublasMmul(handle,d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);

    int n_iter = 10;

    // start the timer
    cudaEventRecord(start, NULL);


    // perform nIter matrix-matrix multiplications
    for (int j = 0; j < n_iter; j++)
    {
      gpuCublasMmul(handle,d_A, d_B, d_C, nr_rows_A, nr_cols_A, nr_cols_B);
    }
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float msec_total = 0.0f;
    cudaEventElapsedTime(&msec_total, start, stop);

    // Compute and print the performance
    float msec_per_matrix_mul = msec_total / n_iter;
    double flops_per_matrix_mul = 2.0 * (double)nr_cols_C * (double)nr_cols_B * (double)nr_cols_A;
    double giga_flops = (flops_per_matrix_mul * 1.0e-9f) / (msec_per_matrix_mul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
        giga_flops,
        msec_per_matrix_mul,
        flops_per_matrix_mul);


    // copy the result in host memory
    cudaMemcpy(h_C,d_C,nr_rows_C * nr_cols_C * sizeof(float),cudaMemcpyDeviceToHost);

    // perform naive matrix multiplication on the CPU for corectness check
    matrixMulCPU(reference, h_B, h_A, nr_rows_A, nr_cols_A, nr_cols_B);
    printf("DONE matMulCPU \n");

    // perform corectness check
    bool correct = compareReference(reference, h_C, nr_cols_A*nr_cols_B, 1.0e-6f);
    printf("Comparing CUBLAS Matrix Multiply with CPU results: %s\n", (true == correct) ? "PASS" : "FAIL");

    // free cuBlas handle
    cublasDestroy(handle);

    // destroy timer events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);  

    // free CPU memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);

    return 0;
}
