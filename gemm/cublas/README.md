=======
README
=======

# 1. Code sample name
gemm

# 2. Description of the code sample package
This example demonstrates the use of NVIDIA's linear algebra library for CUDA: cuBLAS. The example is set-up to perform the computation of both CPU and GPU and in the end to verify the results.

Additional pre-requisites:
* CUDA (includes the cuBLAS library)
* clBLAS

See http://docs.nvidia.com/cuda/cublas for the full cuBLAS documentation.
See https://github.com/clMathLibraries/clBLAS for the clBLAS library

# 3. Release date
25 July 2015

# 4. Version history 
1.0

# 5. Contributor (s) / Maintainer(s) 
Valeriu Codreanu <valeriu.codreanu@surfsara.nl>

# 6. Copyright / License of the code sample
Apache 2.0

# 7. Language(s) 
C++
CUDA

# 8. Parallelisation Implementation(s)
GPU

# 9. Level of the code sample complexity 
Basic level, uses library calls only

# 10. Instructions on how to compile the code
Uses the CodeVault CMake infrastructure, see main README.md

# 11. Instructions on how to run the code
Run the executable with a single command-line option, the matrix size

# 12. Sample input(s)
Input-data is generated automatically when running the program.

# 13. Sample output(s)
Output data is verified programmatically using a CPU implementation of GEMM.
