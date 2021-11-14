
cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(pyvnlb
  DESCRIPTION "Python bindings for vnlb."
  HOMEPAGE_URL "https://github.com/pariasm/vnlb"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

find_package(SWIG REQUIRED COMPONENTS python)
include(${SWIG_USE_FILE})

set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set(SWIG_SOURCE_FILE_EXTENSIONS swig)

macro(configure_swigvnlb source)
  set_source_files_properties(${source} PROPERTIES
    CPLUSPLUS ON
    USE_TARGET_INCLUDE_DIRECTORIES TRUE
  )
  if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWORDSIZE64
    )
  endif()
  if(WIN32)
    set_source_files_properties(${source} PROPERTIES
      SWIG_FLAGS -DSWIGWIN
    )
  endif()
endmacro()

# CMake's SWIG wrappers only allow tweaking certain settings at source level, so
# we duplicate the source in order to override the module name.
configure_file(swigvnlb.swig ${CMAKE_CURRENT_SOURCE_DIR}/swigvnlb_avx2.swig COPYONLY)

configure_swigvnlb(swigvnlb.swig)
configure_swigvnlb(swigvnlb_avx2.swig)

if(TARGET vnlb)
  # Manually add headers as extra dependencies of swigvnlb.
  set(SWIG_MODULE_swigvnlb_EXTRA_DEPS)
  foreach(h ${VNLB_HEADERS})
    list(APPEND SWIG_MODULE_swigvnlb_EXTRA_DEPS "${vnlb_SOURCE_DIR}/vnlb/${h}")
    list(APPEND SWIG_MODULE_swigvnlb_avx2_EXTRA_DEPS "${vnlb_SOURCE_DIR}/vnlb/${h}")
  endforeach()
else()
  find_package(vnlb REQUIRED)
endif()

swig_add_library(swigvnlb
  TYPE SHARED
  LANGUAGE python
  SOURCES swigvnlb.swig
)

set_property(SOURCE swigvnlb_avx2.swig
  PROPERTY SWIG_MODULE_NAME swigvnlb_avx2)
swig_add_library(swigvnlb_avx2
  TYPE SHARED
  LANGUAGE python
  SOURCES swigvnlb_avx2.swig
)
if(NOT VNLB_OPT_LEVEL STREQUAL "avx2")
  set_target_properties(swigvnlb_avx2 PROPERTIES EXCLUDE_FROM_ALL TRUE)
endif()

if(NOT WIN32)
  # NOTE: Python does not recognize the dylib extension.
  set_target_properties(swigvnlb PROPERTIES SUFFIX .so)
  set_target_properties(swigvnlb_avx2 PROPERTIES SUFFIX .so)
endif()

if(VNLB_ENABLE_GPU)
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(swigvnlb PRIVATE CUDA::cudart)
  target_link_libraries(swigvnlb_avx2 PRIVATE CUDA::cudart)
endif()

find_package(OpenMP REQUIRED)

target_link_libraries(swigvnlb PRIVATE
  vnlb
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

target_link_libraries(swigvnlb_avx2 PRIVATE
  vnlb_avx2
  Python::Module
  Python::NumPy
  OpenMP::OpenMP_CXX
)

# Hack so that python_callbacks.h can be included as
# `#include <vnlb/python/python_callbacks.h>`.
target_include_directories(swigvnlb PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(swigvnlb_avx2 PRIVATE ${PROJECT_SOURCE_DIR}/../..)

find_package(Python REQUIRED
  COMPONENTS Development NumPy
)

add_library(vnlb_python_callbacks EXCLUDE_FROM_ALL
  python_callbacks.cpp
)
set_property(TARGET vnlb_python_callbacks
  PROPERTY POSITION_INDEPENDENT_CODE ON
)

# Hack so that python_callbacks.h can be included as
# `#include <vnlb/python/python_callbacks.h>`.
target_include_directories(vnlb_python_callbacks PRIVATE ${PROJECT_SOURCE_DIR}/../..)
target_include_directories(vnlb_python_callbacks PRIVATE ${Python_INCLUDE_DIRS})

target_link_libraries(swigvnlb PRIVATE vnlb_python_callbacks)
target_link_libraries(swigvnlb_avx2 PRIVATE vnlb_python_callbacks)

configure_file(setup.py setup.py COPYONLY)
configure_file(__init__.py __init__.py COPYONLY)
configure_file(loader.py loader.py COPYONLY)

file(GLOB files "${PROJECT_SOURCE_DIR}/../../pylib/*.py")
file(COPY ${files} DESTINATION pylib/)
