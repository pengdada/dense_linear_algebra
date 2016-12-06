=======
README
=======

# 1. Code sample name
gemm

# 2. Description of the code sample package
This example demonstrates a multi-dimensional thread blocks and shared memory implementation of matrix-matrix multiplication using NVIDIA's CUDA. The example is set-up to perform the computation of both CPU and GPU and in the end to verify the results.

Additional pre-requisites:
* CUDA 

See http://docs.nvidia.com/cuda/ about CUDA information

# 3. Release date
6 Dec 2016

# 4. Version history 
1.0

Damian Podareanu <damian.podareanu@surfsara.nl>

# 6. Copyright / License of the code sample
Apache 2.0

# 7. Language(s) 
C++
CUDA

# 8. Parallelisation Implementation(s)
GPU

# 9. Level of the code sample complexity 
Basic level

# 10. Instructions on how to compile the code
Uses the CodeVault CMake infrastructure, see main README.md

# 11. Instructions on how to run the code
Run the executable with a single command-line option, the matrix size

# 12. Sample input(s)
Input-data is generated automatically when running the program by computing sin and cos values.

# 13. Sample output(s)
Output data is verified programmatically using a CPU implementation of matrix-matrix multiplication.
