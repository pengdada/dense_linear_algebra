# README - MKL Blas Gemm Example

## Description

Matrix multiplications can be done in different ways which vary in performance. This example should point out how massive these performance differences really are.

This code sample demonstrates:
 * Different implementations to do a matrix multiplication
   * Serial
     * IJK Algorithm
     * IKJ Algorithm
   * Parallel
     * OpenMP parallelized IKJ Algorithm
     * Intel MKL BLAS Library
 * Benchmark implementations with different matrix dimensions.

Benchmark design:

The program runs the multiplication with matrix dimensions from 2-2048 with single precision floats. For timing accuracy multiple repetitions are used and the average time per multiplication is calculated. The program output is a CSV-Text with dimension size and time per multiplication in seconds.

## Release Date

2016-10-24

## Version History

 * 2016-10-24: Initial Release on PRACE CodeVault repository

## Contributors

 * Thomas Steinreiter - [thomas.steinreiter@risc-software.at](mailto:thomas.steinreiter@risc-software.at)


## Copyright

This code is available under Apache License, Version 2.0 - see also the license file in the CodeVault root directory.


## Languages

This sample is entirely written in C++14.

## Parallelisation

This sample uses MPI-3 for parallelization.

## Level of the code sample complexity

Intermediate

## Compiling

Follow the compilation instructions given in the main directory of the kernel samples directory (`/hpc_kernel_samples/README.md`).

## Running

In your current working directory, to run the program you may use something similar to

```
1_dense_mklblas blas
```

either on the command line or in your batch script.


### Command line arguments

 * `<mode>`: Specity the implementation used for the multiplication (obligatory)
   * `serialijk` Use the serial IJK Algorithm 
   * `serialikj` Use the serial IKJ Algorithm
   * `parallel` Use the parallel (OpenMp) IKJ Algorithm
   * `serialijk` Use the parallel Intel MKL Library

### Example

If you run

```
1_dense_mklblas blas
```

the output should look similar to

```
2;6.59846e-06
4;6.6475e-06
8;5.93214e-06
16;6.80383e-06
32;8.44547e-06
64;2.10678e-05
128;0.000102738
256;0.000789877
512;0.00560496
1024;0.0301494
2048;0.338072
```

### Benchmarks

Tested on Intel Core i7-6820HQ @ 2,70GHz (4 physical cores, 8 logical).
Compiler: Intel(R) 64, Version 17.0.0.109 Build 20160721 (MKL 2017.0) on Windows 10 1607.

![chart](hpc_kernel_samples/dense_linear_algebra/gemm/mklblas/benchmarks/Chart.PNG)

Be aware the chart has logarithmic scale.

The benchmarks shows some interesting facts:
 * The *serial IKJ* implementation is brutally faster than IJK (~30x @2048)
 * *Serial IKJ* is the fastest smaller sizes (2-16). *MKL Blas* is the fastest for larger sizes (32-2048).
 * The *Parallel OpenMP IKJ implementation* is never the fastest. **Manually parallelizing matrix multiplication is not recommended!**