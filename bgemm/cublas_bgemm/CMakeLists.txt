
# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#   Damian Podareanu <damian.podareanu@surfsara.nl>
#
# ==================================================================================================

# Packages are optional: if they are not present, certain code samples are not compiled
find_package(OpenMP)   # Built-in in CMake
find_package(MPI)      # Built-in in CMake
find_package(CUDA)     # Built-in in CMake
include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)
# ==================================================================================================
if ("${DWARF_PREFIX}" STREQUAL "")
  set(DWARF_PREFIX 1_dense)
endif()

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
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 -gencode=arch=compute_35,code=\"sm_35,compute_35\"")
  set(CUDA_HOST_COMPILER "g++")
endif()

# ==================================================================================================

# Batch GEMM with the CUDA cuBLAS library
set(NAME ${DWARF_PREFIX}_cublas_bgemm)
if (CUDA_FOUND)
  cuda_add_executable(${NAME} src/cublas_bgemm.cpp)
  target_link_libraries(${NAME} ${CUDA_CUBLAS_LIBRARIES})
  target_link_libraries(${NAME} ${CUDA_curand_LIBRARY})
  install(TARGETS ${NAME} DESTINATION bin)
  if (OPENMP_FOUND)
    message("** Enabling '${NAME}': with OpenMP")
  else()
    message("** Enabling '${NAME}': without OpenMP")
  endif()
else()
  message("** Skipping '${NAME}': no CUDA")
  dummy_install(${NAME} "CUDA")
endif()

unset(NAME)

# ==================================================================================================
