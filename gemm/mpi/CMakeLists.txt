
# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#   Valeriu Codreanu <valeriu.codreanu@surfsara.nl>
#
# ==================================================================================================

# Packages are optional: if they are not present, certain code samples are not compiled
find_package(MPI)     # Built-in in CMake
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)
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
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")
set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
# ==================================================================================================

# GEMM with the CUDA library
set(NAME ${DWARF_PREFIX}_gemm_mpi)
if (MPI_FOUND)
  add_executable(${NAME} src/gemm_mpi.cpp)
  target_link_libraries(${NAME} ${MPI_LIBRARIES})
  install(TARGETS ${NAME} DESTINATION bin)
else()
  message("** Skipping '${NAME}': no MPI")
  dummy_install(${NAME} "MPI")
endif()

unset(NAME)

# ==================================================================================================
