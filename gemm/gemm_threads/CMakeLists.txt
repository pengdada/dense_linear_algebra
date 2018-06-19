
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
  set(DWARF_PREFIX gemm_threads)
endif()

find_package(Common)

enable_language (C)

set(NAME ${DWARF_PREFIX}_gemm_threads)

set(CXX_FLAGS "${CXX_FLAGS} -Wall -Wno-comment -std=c++0x -g -pthread")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")
add_executable(${NAME} src/gemm.cpp)
install(TARGETS ${NAME} DESTINATION bin)

unset(NAME)
