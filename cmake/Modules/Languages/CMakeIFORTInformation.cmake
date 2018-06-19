# This file sets the basic flags for the IFORT language in CMake.
# It also loads the available platform file for the system-compiler
# if it exists.

GET_FILENAME_COMPONENT(CMAKE_BASE_NAME ${CMAKE_IFORT_COMPILER} NAME_WE)
SET(CMAKE_SYSTEM_AND_IFORT_COMPILER_INFO_FILE
  ${CMAKE_ROOT}/Modules/Platform/${CMAKE_SYSTEM_NAME}-${CMAKE_BASE_NAME}.cmake)
INCLUDE(Platform/${CMAKE_SYSTEM_NAME}-${CMAKE_BASE_NAME} OPTIONAL)

# This should be included before the _INIT variables are
# used to initialize the cache.  Since the rule variables 
# have if blocks on them, users can still define them here.
# But, it should still be after the platform file so changes can
# be made to those values.

IF(CMAKE_USER_MAKE_RULES_OVERRIDE)
   INCLUDE(${CMAKE_USER_MAKE_RULES_OVERRIDE})
ENDIF(CMAKE_USER_MAKE_RULES_OVERRIDE)

IF(CMAKE_USER_MAKE_RULES_OVERRIDE_IFORT)
  INCLUDE(${CMAKE_USER_MAKE_RULES_OVERRIDE_IFORT})
ENDIF(CMAKE_USER_MAKE_RULES_OVERRIDE_IFORT)

# Create a set of shared library variable specific to IFORT
# For 90% of the systems, these are the same flags as the C versions
# so if these are not set just copy the flags from the c version

IF(NOT CMAKE_SHARED_LIBRARY_CREATE_IFORT_FLAGS)
  SET(CMAKE_SHARED_LIBRARY_CREATE_IFORT_FLAGS ${CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS})
ENDIF(NOT CMAKE_SHARED_LIBRARY_CREATE_IFORT_FLAGS)

IF(NOT CMAKE_IFORT_COMPILE_OPTIONS_PIC)
  SET(CMAKE_IFORT_COMPILE_OPTIONS_PIC ${CMAKE_C_COMPILE_OPTIONS_PIC})
ENDIF(NOT CMAKE_IFORT_COMPILE_OPTIONS_PIC)

IF(NOT CMAKE_IFORT_COMPILE_OPTIONS_PIE)
  SET(CMAKE_IFORT_COMPILE_OPTIONS_PIE ${CMAKE_C_COMPILE_OPTIONS_PIE})
ENDIF(NOT CMAKE_IFORT_COMPILE_OPTIONS_PIE)

IF(NOT CMAKE_IFORT_COMPILE_OPTIONS_DLL)
  SET(CMAKE_IFORT_COMPILE_OPTIONS_DLL ${CMAKE_C_COMPILE_OPTIONS_DLL})
ENDIF(NOT CMAKE_IFORT_COMPILE_OPTIONS_DLL)

IF(NOT CMAKE_SHARED_LIBRARY_IFORT_FLAGS)
  SET(CMAKE_SHARED_LIBRARY_IFORT_FLAGS ${CMAKE_SHARED_LIBRARY_C_FLAGS})
ENDIF(NOT CMAKE_SHARED_LIBRARY_IFORT_FLAGS)

IF(NOT DEFINED CMAKE_SHARED_LIBRARY_LINK_IFORT_FLAGS)
  SET(CMAKE_SHARED_LIBRARY_LINK_IFORT_FLAGS ${CMAKE_SHARED_LIBRARY_LINK_C_FLAGS})
ENDIF(NOT DEFINED CMAKE_SHARED_LIBRARY_LINK_IFORT_FLAGS)

IF(NOT CMAKE_SHARED_LIBRARY_RUNTIME_IFORT_FLAG)
  SET(CMAKE_SHARED_LIBRARY_RUNTIME_IFORT_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG}) 
ENDIF(NOT CMAKE_SHARED_LIBRARY_RUNTIME_IFORT_FLAG)

IF(NOT CMAKE_SHARED_LIBRARY_RUNTIME_IFORT_FLAG_SEP)
  SET(CMAKE_SHARED_LIBRARY_RUNTIME_IFORT_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_C_FLAG_SEP})
ENDIF(NOT CMAKE_SHARED_LIBRARY_RUNTIME_IFORT_FLAG_SEP)

IF(NOT CMAKE_SHARED_LIBRARY_RPATH_LINK_IFORT_FLAG)
  SET(CMAKE_SHARED_LIBRARY_RPATH_LINK_IFORT_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_C_FLAG})
ENDIF(NOT CMAKE_SHARED_LIBRARY_RPATH_LINK_IFORT_FLAG)

# repeat for modules
IF(NOT CMAKE_SHARED_MODULE_CREATE_IFORT_FLAGS)
  SET(CMAKE_SHARED_MODULE_CREATE_IFORT_FLAGS ${CMAKE_SHARED_MODULE_CREATE_C_FLAGS})
ENDIF(NOT CMAKE_SHARED_MODULE_CREATE_IFORT_FLAGS)

IF(NOT CMAKE_SHARED_MODULE_IFORT_FLAGS)
  SET(CMAKE_SHARED_MODULE_IFORT_FLAGS ${CMAKE_SHARED_MODULE_C_FLAGS})
ENDIF(NOT CMAKE_SHARED_MODULE_IFORT_FLAGS)

IF(NOT CMAKE_SHARED_MODULE_RUNTIME_IFORT_FLAG)
  SET(CMAKE_SHARED_MODULE_RUNTIME_IFORT_FLAG ${CMAKE_SHARED_MODULE_RUNTIME_C_FLAG}) 
ENDIF(NOT CMAKE_SHARED_MODULE_RUNTIME_IFORT_FLAG)

IF(NOT CMAKE_SHARED_MODULE_RUNTIME_IFORT_FLAG_SEP)
  SET(CMAKE_SHARED_MODULE_RUNTIME_IFORT_FLAG_SEP ${CMAKE_SHARED_MODULE_RUNTIME_C_FLAG_SEP})
ENDIF(NOT CMAKE_SHARED_MODULE_RUNTIME_IFORT_FLAG_SEP)

IF(NOT CMAKE_EXECUTABLE_RUNTIME_IFORT_FLAG)
  SET(CMAKE_EXECUTABLE_RUNTIME_IFORT_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_IFORT_FLAG})
ENDIF(NOT CMAKE_EXECUTABLE_RUNTIME_IFORT_FLAG)

IF(NOT CMAKE_EXECUTABLE_RUNTIME_IFORT_FLAG_SEP)
  SET(CMAKE_EXECUTABLE_RUNTIME_IFORT_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_IFORT_FLAG_SEP})
ENDIF(NOT CMAKE_EXECUTABLE_RUNTIME_IFORT_FLAG_SEP)

IF(NOT CMAKE_EXECUTABLE_RPATH_LINK_IFORT_FLAG)
  SET(CMAKE_EXECUTABLE_RPATH_LINK_IFORT_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_IFORT_FLAG})
ENDIF(NOT CMAKE_EXECUTABLE_RPATH_LINK_IFORT_FLAG)

IF(NOT DEFINED CMAKE_SHARED_LIBRARY_LINK_IFORT_WITH_RUNTIME_PATH)
  SET(CMAKE_SHARED_LIBRARY_LINK_IFORT_WITH_RUNTIME_PATH ${CMAKE_SHARED_LIBRARY_LINK_C_WITH_RUNTIME_PATH})
ENDIF(NOT DEFINED CMAKE_SHARED_LIBRARY_LINK_IFORT_WITH_RUNTIME_PATH)

IF(NOT CMAKE_INCLUDE_FLAG_IFORT)
  SET(CMAKE_INCLUDE_FLAG_IFORT ${CMAKE_INCLUDE_FLAG_C})
ENDIF(NOT CMAKE_INCLUDE_FLAG_IFORT)

IF(NOT CMAKE_INCLUDE_FLAG_SEP_IFORT)
  SET(CMAKE_INCLUDE_FLAG_SEP_IFORT ${CMAKE_INCLUDE_FLAG_SEP_C})
ENDIF(NOT CMAKE_INCLUDE_FLAG_SEP_IFORT)

# Copy C version of this flag which is normally determined in platform file.
IF(NOT CMAKE_SHARED_LIBRARY_SONAME_IFORT_FLAG)
  SET(CMAKE_SHARED_LIBRARY_SONAME_IFORT_FLAG ${CMAKE_SHARED_LIBRARY_SONAME_C_FLAG})
ENDIF(NOT CMAKE_SHARED_LIBRARY_SONAME_IFORT_FLAG)

SET(CMAKE_VERBOSE_MAKEFILE FALSE CACHE BOOL "If this value is on, makefiles will be generated without the .SILENT directive, and all commands will be echoed to the console during the make.  This is useful for debugging only. With Visual Studio IDE projects all commands are done without /nologo.")

SET(CMAKE_IFORT_FLAGS_INIT "$ENV{IFORTFLAGS} ${CMAKE_IFORT_FLAGS_INIT}")
# avoid just having a space as the initial value for the cache 
IF(CMAKE_IFORT_FLAGS_INIT STREQUAL " ")
  SET(CMAKE_IFORT_FLAGS_INIT)
ENDIF(CMAKE_IFORT_FLAGS_INIT STREQUAL " ")
SET (CMAKE_IFORT_FLAGS "${CMAKE_IFORT_FLAGS_INIT}" CACHE STRING
  "Flags for IFORT compiler.")

INCLUDE(CMakeCommonLanguageInclude)

# now define the following rule variables

# CMAKE_IFORT_CREATE_SHARED_LIBRARY
# CMAKE_IFORT_CREATE_SHARED_MODULE
# CMAKE_IFORT_CREATE_STATIC_LIBRARY
# CMAKE_IFORT_COMPILE_OBJECT
# CMAKE_IFORT_LINK_EXECUTABLE

# variables supplied by the generator at use time
# <TARGET>
# <TARGET_BASE> the target without the suffix
# <OBJECTS>
# <OBJECT>
# <LINK_LIBRARIES>
# <FLAGS>
# <LINK_FLAGS>

# IFORT compiler information
# <CMAKE_IFORT_COMPILER>  
# <CMAKE_SHARED_LIBRARY_CREATE_IFORT_FLAGS>
# <CMAKE_SHARED_MODULE_CREATE_IFORT_FLAGS>
# <CMAKE_IFORT_LINK_FLAGS>

# Static library tools
# <CMAKE_AR> 
# <CMAKE_RANLIB>


# create a shared C++ library
if(NOT CMAKE_IFORT_CREATE_SHARED_LIBRARY)
  set(CMAKE_IFORT_CREATE_SHARED_LIBRARY
    "<CMAKE_IFORT_COMPILER> <CMAKE_SHARED_LIBRARY_IFORT_FLAGS> <LANGUAGE_COMPILE_FLAGS> <LINK_FLAGS> <CMAKE_SHARED_LIBRARY_CREATE_IFORT_FLAGS> <SONAME_FLAG><TARGET_SONAME> -o <TARGET> <OBJECTS> <LINK_LIBRARIES>")
endif()

# create a c++ shared module copy the shared library rule by default
if(NOT CMAKE_IFORT_CREATE_SHARED_MODULE)
  set(CMAKE_IFORT_CREATE_SHARED_MODULE ${CMAKE_IFORT_CREATE_SHARED_LIBRARY})
endif()


# Create a static archive incrementally for large object file counts.
# If CMAKE_IFORT_CREATE_STATIC_LIBRARY is set it will override these.
if(NOT DEFINED CMAKE_IFORT_ARCHIVE_CREATE)
  set(CMAKE_IFORT_ARCHIVE_CREATE "<CMAKE_AR> cq <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_IFORT_ARCHIVE_APPEND)
  set(CMAKE_IFORT_ARCHIVE_APPEND "<CMAKE_AR> q  <TARGET> <LINK_FLAGS> <OBJECTS>")
endif()
if(NOT DEFINED CMAKE_IFORT_ARCHIVE_FINISH)
  set(CMAKE_IFORT_ARCHIVE_FINISH "<CMAKE_RANLIB> <TARGET>")
endif()

# compile a C++ file into an object file
if(NOT CMAKE_IFORT_COMPILE_OBJECT)
  set(CMAKE_IFORT_COMPILE_OBJECT
    "<CMAKE_IFORT_COMPILER>  <DEFINES> <FLAGS> -o <OBJECT> -c <SOURCE>")
endif()

if(NOT CMAKE_IFORT_LINK_EXECUTABLE)
  set(CMAKE_IFORT_LINK_EXECUTABLE
    "<CMAKE_IFORT_COMPILER>  <FLAGS> <CMAKE_IFORT_LINK_FLAGS> <LINK_FLAGS> <OBJECTS>  -o <TARGET> <LINK_LIBRARIES>")
endif()

mark_as_advanced(
CMAKE_VERBOSE_MAKEFILE
CMAKE_IFORT_FLAGS
CMAKE_IFORT_FLAGS_RELEASE
CMAKE_IFORT_FLAGS_RELWITHDEBINFO
CMAKE_IFORT_FLAGS_MINSIZEREL
CMAKE_IFORT_FLAGS_DEBUG)

set(CMAKE_IFORT_INFORMATION_LOADED 1)

