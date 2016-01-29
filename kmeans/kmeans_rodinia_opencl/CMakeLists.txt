
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
find_package(OpenCL )


include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)

# ==================================================================================================

if ("${DWARF_PREFIX}" STREQUAL "")
  set(DWARF_PREFIX 1_dense)
endif()
set(NAME ${DWARF_PREFIX}_kmeans_rodinia_opencl)

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

# C compiler settings
select_compiler_flags(c_flags
  GNU "-march=native"   # I suggest remove "-O3" as this is controlled by the CMAKE_BUILD_TYPE
  CLANG "-march=native" # same here
  Intel "-axavx2,avx")
set(C_FLAGS ${c_flags})
if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
  set(C_FLAGS "${C_FLAGS} -Wall -Wno-comment -DOPENCL_HEADER_CL_CL")
  if(APPLE)
    set(C_FLAGS "${C_FLAGS} -Wa,-q -DOPENCL_HEADER_LONG")
  endif()
endif()
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_FLAGS}")


# NVCC compiler settings
if (CUDA_FOUND)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
  set(CUDA_HOST_COMPILER "g++")
endif()

# ==================================================================================================


# GEMM with the CUDA cuBLAS library
find_package(Common)
if (CUDA_FOUND)
  include_directories(${OPENCL_INCLUDE_DIRS})
  add_executable(${NAME} src/cluster.c src/getopt.c src/kmeans_clustering.c src/kmeans.cpp src/read_input.c src/rmse.c)
  target_link_libraries(${NAME} ${OPENCL_LIBRARIES})
  configure_file(src/kmeans.cl kmeans.cl COPYONLY)
#  target_link_libraries(${NAME} ${CUDA_curand_LIBRARY})
  install(TARGETS ${NAME} DESTINATION bin)
  if (OPENMP_FOUND)
    message("** Enabling '${NAME}': with OpenMP and OpenMP")
    message("** Enabling '${CMAKE_MODULE_PATH}': with OpenMP and OpenMP")
  else()
    message("** Enabling '${NAME}': with OpenCL and without OpenMP")
  endif()
else()
  message("** Skipping '${NAME}': no OpenCL")
  dummy_install(${NAME} "OpenCL")
endif()

unset(NAME)

# ==================================================================================================
