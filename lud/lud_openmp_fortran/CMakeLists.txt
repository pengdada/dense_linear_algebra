cmake_minimum_required(VERSION 2.8.7 FATAL_ERROR)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/common.cmake)
 
# ==================================================================================================
if ("${DWARF_PREFIX}" STREQUAL "")
  set(DWARF_PREFIX dense_linear_algebra)
endif()
 
find_package(Common)
find_package(OpenMP)
 
enable_language (Fortran)
 
set(NAME ${DWARF_PREFIX}_lud_openmp_fortran)
if (OPENMP_FOUND)
  set (CMAKE_Fortran_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
  add_executable(${NAME} src/lud.f90)
  install(TARGETS ${NAME} DESTINATION bin)
else()
  message("## Skipping '${NAME}_omp': no OpenMP support found")
  dummy_install(${NAME} "OpenMP")
endif()
 
unset(NAME)
