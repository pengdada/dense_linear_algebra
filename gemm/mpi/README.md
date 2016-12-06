=======
README
=======

# 1. Code sample name
gemm

# 2. Description of the code sample package
This example demonstrates an MPI implementation

Additional pre-requisites:
* Open MPI there is an issue with Intel or mvapich. Will be fixed in future releases. For now it works also with those if  something like "mpirun_rsh -np 2 node001 node001 ./runnable" is used

# 3. Release date
6 Dec 2016

# 4. Version history 
1.0

Damian Podareanu <damian.podareanu@surfsara.nl>

# 6. Copyright / License of the code sample
Apache 2.0

# 7. Language(s) 
C++

# 8. Parallelisation Implementation(s)
MPI

# 9. Level of the code sample complexity 
Basic level

# 10. Instructions on how to compile the code
Uses the CodeVault CMake infrastructure, see main README.md

# 11. Instructions on how to run the code
Run the executable. Matrix size is defined in src/gemm_mpi.cpp

# 12. Sample input(s)
Input-data is generated automatically when running the program.