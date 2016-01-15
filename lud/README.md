=======
README
=======

# 1. Code sample name
lud

# 2. Description of the code sample package
This set of examples demonstrates the use of:
* NVIDIA's linear algebra library for CUDA: cuBLAS. 
* NVIDIA's solver library for CUDA: cuSOLVER. 
* Intel's Math Kernel Library: Intel MKL. 

Some examples (cublas_mkl, cusolver_mkl) are set-up to perform the computation of both CPU and GPU and in the end to verify the results.

Additional pre-requisites:
* CUDA (includes the cuBLAS and cuSOLVER libraries)
* Intel MKL

# 3. Release date
30 July 2015

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
CPU

# 9. Level of the code sample complexity 
Basic level, uses library calls only

# 10. Instructions on how to compile the code
Uses the CodeVault CMake infrastructure, see main README.md

# 11. Instructions on how to run the code
Run the executable with a single command-line option, the matrix size

# 12. Sample input(s)
Input-data is generated automatically when running the program.

# 13. Sample output(s)
Output data is verified programmatically using a CPU implementation of LUD. Performance numbers are outputted as well.
