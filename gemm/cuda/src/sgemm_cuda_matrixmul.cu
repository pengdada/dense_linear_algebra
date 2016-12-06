#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "sgemm_cuda_kernel.h"
#include "sgemm_cuda_kernel.cu"
#include "dev_array.h"
#include <math.h>

using namespace std;

int main(int argc, char **argv)
{
    int N;

    if (argc !=2)  {
	printf("Usage: ./1_dense_cuda <matrix_size> \n");
	return 1;
    } else {
	N = atoi(argv[1]);
    }

    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int SIZE = N*N;

    cudaEvent_t start, stop;

    // create cuda timer events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start, NULL);

    // Allocate memory on the host
    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = sin(i);
            h_B[i*N+j] = cos(j);
        }
    }

    // Allocate memory on the device
    dev_array<float> d_A(SIZE);
    dev_array<float> d_B(SIZE);
    dev_array<float> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);

    float msec_total = 0.0f;
    cudaEventElapsedTime(&msec_total, start, stop);

    // Compute and print the performance
    float msec_per_matrix_mul = msec_total;
    double flops_per_matrix_mul = 2.0 * (double)N * (double)N * (double)N;
    double giga_flops = (flops_per_matrix_mul * 1.0e-9f) / (msec_per_matrix_mul / 1000.0f);
    printf(
        "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops\n",
        giga_flops,
        msec_per_matrix_mul,
        flops_per_matrix_mul);


    float *cpu_C;
    cpu_C=new float[SIZE];

    // Now do the matrix multiplication on the CPU
    float sum;
    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
            sum = 0.f;
            for (int n=0; n<N; n++){
                sum += h_A[row*N+n]*h_B[n*N+col];
            }
            cpu_C[row*N+col] = sum;
        }
    }

    double err = 0;
    // Check the result and make sure it is correct
    for (int ROW=0; ROW < N; ROW++){
        for (int COL=0; COL < N; COL++){
            err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
        }
    }

    cout << "Error: " << err << endl;

    return 0;
}