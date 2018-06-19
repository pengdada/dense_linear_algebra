
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

find_package(Common)
find_package(OpenMP)

enable_language (Fortran)

set(NAME ${DWARF_PREFIX}_gemm_openmp_fortran)
if (OPENMP_FOUND)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  add_executable(${NAME} src/gemm.f90)
  install(TARGETS ${NAME} DESTINATION bin)
else()
  message("## Skipping '${NAME}_omp': no OpenMP support found")
  dummy_install(${NAME} "OpenMP")
endif()

unset(NAME)
