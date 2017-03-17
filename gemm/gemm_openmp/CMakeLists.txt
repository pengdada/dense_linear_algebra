# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#  Rafal Gandecki <rafal.gandeci@pwr.edu.pl>
#
# ==================================================================================================
 
cmake_minimum_required(VERSION 2.8.7 FATAL_ERROR)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)
 
# ==================================================================================================
if ("${DWARF_PREFIX}" STREQUAL "")
  set(DWARF_PREFIX gemm_openmp)
endif()
 
find_package(Common)
find_package(OpenMP)
enable_language (C)
 
 
set(NAME ${DWARF_PREFIX})
 
if (OPENMP_FOUND)
  set(CXX_FLAGS "${CXX_FLAGS} -Wall -Wno-comment -std=c++0x -fopenmp")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")
  add_executable(${NAME} src/gemm_openmp.cpp)
  install(TARGETS ${NAME} DESTINATION bin)

else ()
  message("## Skipping '${NAME}': no OpenMP support found")
endif()

unset(NAME)
