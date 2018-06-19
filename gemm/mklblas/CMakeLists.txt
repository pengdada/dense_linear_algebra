# Packages are optional: if they are not present, certain code samples are not compiled
cmake_minimum_required(VERSION 2.8.10 FATAL_ERROR)

find_package(OpenMP)
find_package(Boost 1.58.0)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/common.cmake)

find_package(MKL)

set(MKL_LIBRARIES mkl_intel_lp64 mkl_gnu_thread mkl_core)

# ==================================================================================================

if ("${DWARF_PREFIX}" STREQUAL "")
  set(DWARF_PREFIX 1_dense)
endif()

set(NAME ${DWARF_PREFIX}_mklblas)

if (OPENMP_FOUND AND Boost_FOUND AND MKL_FOUND)
	enable_language(CXX)
    include_directories(${Boost_INCLUDE_DIR} ${MKL_INCLUDE_DIR})
	link_directories(${MKL_LIBRARY_DIR})
    add_executable(${NAME} main.cpp)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
	add_definitions(-DGSL_UNENFORCED_ON_CONTRACT_VIOLATION)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -Wall -Wextra")
    elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -xHost -std=c++14")
    endif()
    set_target_properties(${NAME} PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)
	set_target_properties(${NAME} PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}")
    set_target_properties(${NAME} PROPERTIES LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    target_link_libraries(${NAME} ${Boost_LIBRARIES} ${MKL_LIBRARIES})
    install(TARGETS ${NAME} DESTINATION bin)
    message("** Enabling '${NAME}': with OpenMP and MKL and Boost")
else()
    message("## Skipping '${NAME}': OpenMP or MKL or Boost support missing")
#    dummy_install(${NAME} "MPI")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${C_FLAGS}")

unset(NAME)
# ==================================================================================================
