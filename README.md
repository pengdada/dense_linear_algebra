# PRACE HPC Kernels

This project is part of PRACE CodeVault providing
example codes of common HPC kernels such as dense and sparse linear
algebra, spectral and N-body methods, structured and unstructured
grids, Monte Carlo methods and parallel I/O. The code samples are
published as open source and can be used both for educational purposes
and as parts of real application suites (as permitted by particular
license).  

## How to contribute

Any contributions (new code samples, bug fixes, improvements etc.) are
warmly welcome. In order to contribute, please follow the standard
Gitlab workflow:

1. Fork the project into your personal space
2. Create a feature branch
3. Work on your contributions
4. Push the commit(s) to your fork
5. Submit a merge request to the master branch

## Compilation instructions

The pre-requisites are:

* CMake 2.8.10 or higher. CMake >= 3.0.0 if you want to build everything.
* A C++ compiler
* For advanced examples, C++11/14, OpenMP support, MPI, ISPC, CUDA, OpenCL ...

In order to build the code, follow the typical CMake steps for an
out-of-source build:

```bash
mkdir build
cd build
cmake ..
make
make install # optional
```
