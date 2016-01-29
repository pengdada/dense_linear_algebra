
# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#   Valeriu Codreanu <valeriu.codreanu@surfsara.nl>
#
# ==================================================================================================

# CMake project
cmake_minimum_required(VERSION 2.8.10 FATAL_ERROR)
project("1_dense")
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/common.cmake)

# ==================================================================================================

# Dwarf 1: Dense Linear Algebra
message("--------------------")
message("Dwarf 1: Dense Linear Algebra:")
message("--------------------")
set(DWARF_PREFIX 1_dense) # The prefix of the name of the binaries produced

# Add the examples
add_subdirectory(gemm/cublas)
add_subdirectory(gemm/clblas)
add_subdirectory(kmeans/kmeans_cuda)
add_subdirectory(kmeans/kmeans_openmp)
add_subdirectory(kmeans/kmeans_rodinia_opencl)
add_subdirectory(lud/mkl)
add_subdirectory(lud/cublas)
add_subdirectory(lud/cusolver)
add_subdirectory(lud/cublas_mkl)
add_subdirectory(lud/cusolver_mkl)


# ==================================================================================================




