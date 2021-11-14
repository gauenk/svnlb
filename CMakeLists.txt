
################################################################################
cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(vnlb
  VERSION 1.0.0
  DESCRIPTION "Vnlb code with Python bindings."
  HOMEPAGE_URL "https://github.com/pariasm/vnlb/"
  LANGUAGES C CXX)

cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)

message(STATUS "Configuring main project: " ${CMAKE_PROJECT_NAME})

################################################################################

option(VNLB_OPT_LEVEL "" "generic")
option(VNLB_ENABLE_PYTHON "Build Python extension." ON)

################################################################################

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ggdb")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_LINK_WHAT_YOU_USE off)

################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")
set(CMAKE_VERBOSE_MAKEFILE OFF)

################################################################################

include_directories("$ENV{HOME}/local/include/")
set(CMAKE_PREFIX_PATH "$ENV{HOME}/local/")
message(STATUS "Setting CMAKE_PREFIX_PATH " ${CMAKE_PREFIX_PATH} " for local libraries")
find_package( BLAS REQUIRED )
find_package( CBLAS REQUIRED )
find_package( LAPACKE REQUIRED )
find_package( SWIG REQUIRED COMPONENTS python )
find_package( PythonLibs REQUIRED )

###############################################################################

FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

################################################################################

include(${SWIG_USE_FILE})
include_directories(${PYTHON_INCLUDE_PATH})
include_directories(${OpenCV_INCLUDE_DIRS})
# include_directories(${PNG_LIBRARIES})

################################################################################

include(GNUInstallDirs)
add_subdirectory(vnlb/)
if(VNLB_ENABLE_PYTHON)
  add_subdirectory(vnlb/python/)
endif()
