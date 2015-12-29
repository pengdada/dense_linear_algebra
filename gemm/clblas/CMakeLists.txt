
# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#  Mariusz Uchronski <mariusz.uchronski@pwr.edu.nl>
#
# ==================================================================================================

cmake_minimum_required(VERSION 2.8.7 FATAL_ERROR)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/common.cmake)

# ==================================================================================================
if ("${DWARF_PREFIX}" STREQUAL "")
  set(DWARF_PREFIX dense_linear_algebra)
endif()

enable_language(C)

find_package(Common)
find_package(OpenCL)   # Included as ${CMAKE_MODULE_PATH}/FindOpenCL.cmake

# Finds the OpenCL clBLAS library
find_library(OPENCL_CLBLAS_LIBRARIES
  NAMES clBLAS
  PATH_SUFFIXES lib lib64
  PATHS /usr /usr/local
  DOC "OpenCL clBLAS library"
)

# GEMM with the OpenCL clBLAS library
set(NAME ${DWARF_PREFIX}_gemm_clblas)
if (OPENCL_FOUND)
  include_directories(${OPENCL_INCLUDE_DIRS})
  add_executable(${NAME} src/sgemm_clblas.c)
  target_link_libraries(${NAME} ${OPENCL_LIBRARIES})
  target_link_libraries(${NAME} ${OPENCL_CLBLAS_LIBRARIES})
endif()

unset(NAME)
