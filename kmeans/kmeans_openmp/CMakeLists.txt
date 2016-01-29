
# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#   Valeriu Codreanu <valeriu.codreanu@surfsara.nl>
#
# ==================================================================================================
cmake_minimum_required(VERSION 2.8.10 FATAL_ERROR)

# Packages are optional: if they are not present, certain code samples are not compiled
find_package(OpenMP)   # Built-in in CMake
find_package(MPI)      # Built-in in CMake

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)

# ==================================================================================================

if ("${DWARF_PREFIX}" STREQUAL "")
  set(DWARF_PREFIX 1_dense)
endif()
set(NAME ${DWARF_PREFIX}_kmeans_openmp)

# ==================================================================================================
# C++ compiler settings

find_package(Common)
select_compiler_flags(cxx_flags
  GNU "-march=native"   # I suggest remove "-O3" as this is controlled by the CMAKE_BUILD_TYPE
  CLANG "-march=native" # same here
  Intel "-axavx2,avx")
set(CXX_FLAGS ${cxx_flags})
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CXX_FLAGS "${CXX_FLAGS} -Wall -Wno-comment")
  if(APPLE)
    set(CXX_FLAGS "${CXX_FLAGS} -Wa,-q")
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")
# Kmeans clustering with OpenMP
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    add_executable(${NAME} src/cluster.c src/getopt.c src/kmeans_clustering.c src/kmeans.cpp)
    install(TARGETS ${NAME} DESTINATION bin)
else()
     message("## Skipping '${NAME}_omp': no OpenMP support found")
    dummy_install(${NAME} "OpenMP")
endif()
# ==================================================================================================

unset(NAME)

# ==================================================================================================
