
// =================================================================================================
// This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
// CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
//
// Author(s):
//   Mariusz Uchronski <mariusz.uchronski@pwr.edu.pl>
//
// This example demonstrates the use of AMDS's linear algebra library for OpenCL: clBLAS.
// The example is set-up to perform single precision matrix multiplication. The example
// takes a single input argument, specifying the size of the matrices.
//
// See [http://clmathlibraries.github.io/clBLAS/index.html] for the full clBLAS documentation.
//
// =================================================================================================

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <clBLAS.h>

void matrixMulCPU(float *C, const float *A, const float *B,
                  unsigned int hA, unsigned int wA, unsigned int wB) {

    unsigned int i, j, k;
    for (i = 0; i < hA; ++i)
        for (j = 0; j < wB; ++j)
        {
            float sum = 0;
            for (k = 0; k < wA; ++k)
            {
                sum += A[i * wA + k] * B[k * wB + j];
            }
            C[i * wB + j] = (float)sum;
        }
}

void fillZeros(float *A, int nr_rows_A, int nr_cols_A) {

    unsigned int i, j;
    for (i = 0; i < nr_rows_A; ++i)
    {
        for (j = 0; j < nr_cols_A; ++j)
        {
            A[i * nr_rows_A + j] = 0.0f;
        }
    }

}

void fillRandom(float *A, int nr_rows_A, int nr_cols_A) {

    unsigned int i, j;
    for (i = 0; i < nr_rows_A; ++i)
    {
        for (j = 0; j < nr_cols_A; ++j)
        {
            A[i * nr_rows_A + j] = (float)rand()/(float)(RAND_MAX);
        }
    }
}

int main(int argc, char **argv) {

    int matrix_size;

    float *h_A, *h_B, *h_C, *reference;
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;

    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;
    cl_mem d_A, d_B, d_C;
    cl_event event = NULL;
    const clblasOrder order = clblasRowMajor;
    const cl_float alpha = 1;
    const cl_float beta = 0;
    const clblasTranspose transA = clblasNoTrans;
    const clblasTranspose transB = clblasNoTrans;

    if (argc != 2)  {
        printf("Usage: ./1_dense_clblas <matrix_size> \n");
	    return 1;
    } else {
        matrix_size = atoi(argv[1]);
    }

    nr_rows_A = nr_cols_A = nr_rows_B = nr_cols_B = nr_rows_C = nr_cols_C = matrix_size;

    srand(time(NULL));

    /* allocate arrays on the CPU */
    h_A = (float *)malloc(nr_rows_A * nr_cols_A * sizeof(float));
    h_B = (float *)malloc(nr_rows_B * nr_cols_B * sizeof(float));
    h_C = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));
    reference = (float *)malloc(nr_rows_C * nr_cols_C * sizeof(float));

    /* fill the arrays A and B with random numbers */
    fillRandom(h_A, nr_rows_A, nr_cols_A);
    fillRandom(h_B, nr_rows_B, nr_cols_B);
    fillZeros(h_C, nr_rows_C, nr_cols_C);
    matrixMulCPU(reference, h_A, h_B, nr_rows_A, nr_cols_A, nr_cols_B);

    /* setup OpenCL environment */
    err = clGetPlatformIDs(1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetPlatformIDs() failed with %d\n", err );
        return 1;
    }

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf( "clGetDeviceIDs() failed with %d\n", err );
        return 1;
    }

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateContext() failed with %d\n", err );
        return 1;
    }

    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf( "clCreateCommandQueue() failed with %d\n", err );
        clReleaseContext(ctx);
        return 1;
    }

    /* setup clblas */
    err = clblasSetup();
    if (err != CL_SUCCESS) {
        printf("clblasSetup() failed with %d\n", err);
        clReleaseCommandQueue(queue);
        clReleaseContext(ctx);
        return 1;
    }

    /* prepare OpenCL memory objects */
    d_A = clCreateBuffer(ctx, CL_MEM_READ_ONLY, nr_rows_A * nr_cols_A * sizeof(h_A),
                         NULL, &err);
    d_B = clCreateBuffer(ctx, CL_MEM_READ_ONLY, nr_rows_B * nr_cols_B * sizeof(h_B),
                         NULL, &err);
    d_C = clCreateBuffer(ctx, CL_MEM_READ_WRITE, nr_rows_C * nr_cols_C * sizeof(h_C),
                         NULL, &err);

    /* copy data to the device memory */
    err = clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0,
        nr_rows_A * nr_cols_A * sizeof(h_A), h_A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0,
        nr_rows_B * nr_cols_B * sizeof(h_B), h_B, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, d_C, CL_TRUE, 0,
        nr_rows_C * nr_cols_C * sizeof(h_C), h_C, 0, NULL, NULL);

    /* Call clblas extended function. Perform gemm for the lower right sub-matrices */
    err = clblasSgemm(order, transA, transB, nr_rows_A, nr_cols_B, nr_cols_A,
                      alpha, d_A, 0, nr_rows_A,
                      d_B, 0, nr_cols_B, beta,
                      d_C, 0, nr_cols_C,
                      1, &queue, 0, NULL, &event);
    if (err != CL_SUCCESS) {
        printf("clblasSgemmEx() failed with %d\n", err);
    } else {
        /* Wait for calculations to be finished. */
        err = clWaitForEvents(1, &event);

        /* Fetch results of calculations from GPU memory. */
        err = clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0,
                                  nr_rows_C * nr_cols_C * sizeof(h_C),
                                  h_C, 0, NULL, NULL);
    }

    /* free CPU memory */
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);

    /* free OpenCL memory */
    clReleaseMemObject(d_C);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_A);

    /* finalize clblas */
    clblasTeardown();

    /* release OpenCL working objects */
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);

    return 0;
}
