# ==================================================================================================
# This file is part of the CodeVault project. The project is licensed under Apache Version 2.0.
# CodeVault is part of the EU-project PRACE-4IP (WP7.3.C).
#
# Author(s):
#  Nikos Nikoloutsakos <nikoloutsa@admin.grnet.gr>
#
# ==================================================================================================
 
cmake_minimum_required(VERSION 2.8.7 FATAL_ERROR)
include(${CMAKE_CURRENT_SOURCE_DIR}/../cmake/common.cmake)
 
# ==================================================================================================
if ("${DWARF_PREFIX}" STREQUAL "")
  set(DWARF_PREFIX 1_dense)
endif()
set(NAME_ICC ${DWARF_PREFIX}_pddp2means_omp)
set(NAME_GNU ${DWARF_PREFIX}_pddp2means_gnu)
set(NAME_MIC ${DWARF_PREFIX}_pddp2means_mic)
 
find_package(Common)
find_package(OpenMP)
enable_language (C)
 
# ==================================================================================================
# Check for Intel Compilers 
if("${CMAKE_C_COMPILER_ID}" STREQUAL "Intel")
    set(C_FLAGS "${C_FLAGS} -no-offload")
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        add_executable(${NAME_ICC} src/pddp_2means.c )
        install(TARGETS ${NAME_ICC} DESTINATION bin)
        message("** Enabling '${NAME_ICC}': with OpenMP")
    else()
        message("## Skipping '${NAME_ICC}': no OpenMP support found")
        #dummy_install(${NAME_ICC} "OpenMP")
    endif()

    # if MIC is found on the system 
    if (EXISTS "/usr/linux-k1om-4.7")
        set(C_FLAGS "${C_FLAGS} -offload")
        add_executable(${NAME_MIC} src/pddp_2means.c )
        install(TARGETS ${NAME_MIC} DESTINATION bin)
        message("** Enabling '${NAME_MIC}': with Offload")
    else()
        message("## Skipping '${NAME_MIC}': no MIC support found")
        #dummy_install(${NAME_MIC} "MIC")
    endif()
# Else use GNU
else()
    message("## Please use Intel Compilers for Offload mode")
    if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -lm")
        add_executable(${NAME_GNU} src/pddp_2means.c )
        install(TARGETS ${NAME_GNU} DESTINATION bin)
        message("** Enabling '${NAME_GNU}': with OpenMP")
    else()
        message("## Skipping '${NAME_GNU}': no OpenMP support found")
        #dummy_install(${NAME_GNU} "OpenMP")
    endif()
endif()
# ==================================================================================================
