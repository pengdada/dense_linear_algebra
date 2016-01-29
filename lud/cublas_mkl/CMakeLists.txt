
# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#   Valeriu Codreanu <valeriu.codreanu@surfsara.nl>
#
# ==================================================================================================
cmake_minimum_required(VERSION 2.8.10 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")
# Packages are optional: if they are not present, certain code samples are not compiled
find_package(OpenMP)   # Built-in in CMake
find_package(MPI)      # Built-in in CMake
find_package(CUDA)     # Built-in in CMake
find_package(MKL)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)

# ==================================================================================================

if ("${DWARF_PREFIX}" STREQUAL "")
  set(DWARF_PREFIX 1_dense)
endif()
set(NAME ${DWARF_PREFIX}_lud_cublas_mkl)

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
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")

# NVCC compiler settings
if (CUDA_FOUND)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
  set(CUDA_HOST_COMPILER "g++")
endif()

# ==================================================================================================
# LUD with the CUDA cuBLAS and MKL libraries

if (CUDA_FOUND)
   if (NOT ${MKL_FOUND})
      message("## Skipping '${NAME}': no MKL support found")      
      install(CODE "MESSAGE(\"${NAME} can only be built with MKL.\")")
   else (NOT ${MKL_FOUND})
      include_directories(${MKL_INCLUDE_DIR})
      link_directories(${MKL_LIBRARY_DIR})
      cuda_add_executable(${NAME} src/lud_cublas_mkl.cpp)
      cuda_add_cublas_to_target(${NAME})
      target_link_libraries(${NAME} mkl_intel_lp64 mkl_sequential mkl_core)
      install(TARGETS ${NAME} DESTINATION bin)
         if (OPENMP_FOUND)
            message("** Enabling '${NAME}': with OpenMP")
         else()
            message("** Enabling '${NAME}': without OpenMP")
         endif()
   endif()
else(CUDA_FOUND)
  message("## Skipping '${NAME}': no CUDA found")
  install(CODE "MESSAGE(\"${NAME} can only be built with CUDA.\")")
endif()

unset(NAME)

# ==================================================================================================
