
# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#  Mariusz Uchronski <mariusz.uchronski@pwr.edu.pl>
#
# ==================================================================================================

cmake_minimum_required(VERSION 2.8.7 FATAL_ERROR)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)

# ==================================================================================================
if ("${DWARF_PREFIX}" STREQUAL "")
  set(DWARF_PREFIX dense_linear_algebra)
endif()

enable_language(CXX)

find_package(Common)
find_package(OpenCL)   # Included as ${CMAKE_MODULE_PATH}/FindOpenCL.cmake
find_package(clBLAS)
find_package(clMAGMA)

# Finds the OpenCL clMAGMA library
find_library(CLMAGMA_LIBRARIES
  NAMES clMAGMA
  PATH_SUFFIXES lib lib64
  PATHS /usr /usr/local
  DOC "OpenCL clMAGMA library"
)

# GEMM with the OpenCL clMAGMA library
set(NAME ${DWARF_PREFIX}_lud_clmagma)
if (OPENCL_FOUND AND CLMAGMA_FOUND)
  include_directories(${OPENCL_INCLUDE_DIRS})
  include_directories(${CLMAGMA_INCLUDE_DIRS})
  add_executable(${NAME} src/lud_clmagma.cpp)
  target_link_libraries(${NAME} ${CLMAGMA_LIBRARIES})
  target_link_libraries(${NAME} ${OPENCL_LIBRARIES})
  target_link_libraries(${NAME} ${CLBLAS_LIBRARIES})
  target_link_libraries(${NAME} blas lapack)
  install(TARGETS ${NAME} DESTINATION bin)
  message("** Enabling '${NAME}': with OpenCL and clMAGMA")

else()
     message("## Skipping '${NAME}': no OpenCL or clMAGMA support found")
endif()


unset(NAME)
